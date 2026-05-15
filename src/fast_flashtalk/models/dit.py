import math
import os
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from einops import rearrange
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from ..layers.rope import VideoRopePosition3DEmb
from ..layers.attention import SingleStreamMutiAttention
from ..kernels.fused_ops import (
    fused_affine,
    fused_residual_add,
    fused_residual_mul_add,
    fused_rms_norm,
)
from ..kernels.rope import fast_rope_apply, sinusoidal_embedding_1d
from ..kernels.attn import attention
from ..utils import get_attn_map_with_target


ENABLE_FUSED_OP_LOGS = os.environ.get("ENABLE_FUSED_OP_LOGS", "0") == "1"
ENABLE_FFN_COMPILE = os.environ.get("ENABLE_FFN_COMPILE", "1") == "1"


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._triton_logged = False
        self._triton_calls = 0

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        if x.is_cuda and x.dim() == 3:
            if ENABLE_FUSED_OP_LOGS and not self._triton_logged:
                logger.info(
                    "WanRMSNorm using Triton RMSNorm path for dim={} dtype={}",
                    self.dim,
                    x.dtype,
                )
                self._triton_logged = True
            self._triton_calls += 1
            return fused_rms_norm(x, self.weight, self.eps)
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(origin_dtype)
        return out


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def fuse_qkv(self):
        if hasattr(self, "qkv"):
            return

        qkv = nn.Linear(self.dim, self.dim * 3, bias=self.q.bias is not None).to(
            device=self.q.weight.device, dtype=self.q.weight.dtype
        )
        with torch.no_grad():
            qkv.weight[: self.dim].copy_(self.q.weight)
            qkv.weight[self.dim : 2 * self.dim].copy_(self.k.weight)
            qkv.weight[2 * self.dim :].copy_(self.v.weight)
            if qkv.bias is not None:
                qkv.bias[: self.dim].copy_(self.q.bias)
                qkv.bias[self.dim : 2 * self.dim].copy_(self.k.bias)
                qkv.bias[2 * self.dim :].copy_(self.v.bias)
        self.qkv = qkv
        del self.q
        del self.k
        del self.v

    def clear_runtime_cache(self):
        return

    def forward(self, x, seq_lens, grid_sizes, freqs, ref_target_masks=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        orig_dtype = x.dtype

        # query, key, value function
        if hasattr(self, "qkv"):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        else:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
        q = self.norm_q(q).reshape(b, s, n, d)
        k = self.norm_k(k).reshape(b, s, n, d)
        v = v.reshape(b, s, n, d)
        q = fast_rope_apply(q, freqs)
        k = fast_rope_apply(k, freqs)

        x = attention(q=q, k=k, v=v)
        if x.dtype != orig_dtype:
            x = x.to(dtype=orig_dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)

        if ref_target_masks is not None:
            with torch.no_grad():
                x_ref_attn_map = get_attn_map_with_target(
                    q,
                    k,
                    grid_sizes[0],
                    ref_target_masks=ref_target_masks,
                )
        else:
            x_ref_attn_map = None

        return x, x_ref_attn_map


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self._kv_img_cache = None

    def clear_runtime_cache(self):
        super().clear_runtime_cache()
        self._kv_img_cache = None

    def _get_cached_image_kv(self, context_img: torch.Tensor):
        cache_key = (
            context_img.data_ptr(),
            tuple(context_img.shape),
            context_img.device.type,
            context_img.device.index,
            context_img.dtype,
        )
        if self._kv_img_cache is not None:
            cached_key, cached_k_img, cached_v_img = self._kv_img_cache
            if cached_key == cache_key:
                return cached_k_img, cached_v_img

        if hasattr(self, "kv_img"):
            context_img_kv = self.kv_img(context_img)
            k_img, v_img = context_img_kv.chunk(2, dim=-1)
        else:
            k_img = self.k_img(context_img)
            v_img = self.v_img(context_img)
        if self.qk_norm:
            k_img = self.norm_k_img(k_img)

        self._kv_img_cache = (cache_key, k_img, v_img)
        return k_img, v_img

    def fuse_image_kv(self):
        if hasattr(self, "kv_img"):
            return

        kv_img = nn.Linear(self.dim, self.dim * 2, bias=self.k_img.bias is not None).to(
            device=self.k_img.weight.device, dtype=self.k_img.weight.dtype
        )
        with torch.no_grad():
            kv_img.weight[: self.dim].copy_(self.k_img.weight)
            kv_img.weight[self.dim :].copy_(self.v_img.weight)
            if kv_img.bias is not None:
                kv_img.bias[: self.dim].copy_(self.k_img.bias)
                kv_img.bias[self.dim :].copy_(self.v_img.bias)
        self.kv_img = kv_img
        del self.k_img
        del self.v_img

    def fuse_context_kv(self):
        if hasattr(self, "kv"):
            return

        kv = nn.Linear(self.dim, self.dim * 2, bias=self.k.bias is not None).to(
            device=self.k.weight.device, dtype=self.k.weight.dtype
        )
        with torch.no_grad():
            kv.weight[: self.dim].copy_(self.k.weight)
            kv.weight[self.dim :].copy_(self.v.weight)
            if kv.bias is not None:
                kv.bias[: self.dim].copy_(self.k.bias)
                kv.bias[self.dim :].copy_(self.v.bias)
        self.kv = kv
        del self.k
        del self.v

    def forward(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        if hasattr(self, "kv"):
            context_kv = self.kv(context)
        else:
            context_kv = self.kv_linear(context)
        k, v = context_kv.chunk(2, dim=-1)
        if self.qk_norm:
            k = self.norm_k(k)
        k = k.reshape(b, -1, n, d)
        v = v.reshape(b, -1, n, d)
        k_img, v_img = self._get_cached_image_kv(context_img)
        k_img = k_img.reshape(b, -1, n, d)
        v_img = v_img.reshape(b, -1, n, d)
        img_x = attention(q, k_img, v_img)
        # compute attention
        x = attention(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(ffn_dim, dim)
        self._compiled = False

        if ENABLE_FFN_COMPILE and hasattr(torch, "compile"):
            try:
                self._compiled_core = torch.compile(
                    self._forward_impl,
                    fullgraph=False,
                    dynamic=False,
                    mode="reduce-overhead",
                )
                self._compiled = True
                if ENABLE_FUSED_OP_LOGS:
                    logger.info(
                        "Compiled Wan FFN with torch.compile for dim={} ffn_dim={}",
                        dim,
                        ffn_dim,
                    )
            except Exception as exc:
                self._compiled_core = None
                logger.warning(
                    "Wan FFN compile failed; falling back to eager path. error={}",
                    exc,
                )
        else:
            self._compiled_core = None

    def _forward_impl(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        if self._compiled_core is not None:
            return self._compiled_core(x)
        return self._forward_impl(x)


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        output_dim=768,
        norm_input_visual=True,
        class_range=24,
        class_interval=4,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFeedForward(dim, ffn_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # init audio module
        self.audio_cross_attn = SingleStreamMutiAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm,
            class_range=class_range,
            class_interval=class_interval,
        )
        self.norm_x = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if norm_input_visual
            else nn.Identity()
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_embedding=None,
        ref_target_masks=None,
        human_num=None,
    ):

        dtype = x.dtype
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y, x_ref_attn_map = self.self_attn(
            fused_affine(self.norm1(x), 1 + e[1], e[0]),
            seq_lens,
            grid_sizes,
            freqs,
            ref_target_masks=ref_target_masks,
        )
        x = fused_residual_mul_add(x, y, e[2])

        x = x.to(dtype)

        # cross-attention of text
        x = fused_residual_add(
            x,
            self.cross_attn(self.norm3(x), context, context_lens),
        )

        # cross attn of audio
        x_a = self.audio_cross_attn(
            self.norm_x(x),
            encoder_hidden_states=audio_embedding,
            shape=grid_sizes[0],
            x_ref_attn_map=x_ref_attn_map,
            human_num=human_num,
        )
        x = fused_residual_add(x, x_a)

        y = self.ffn(fused_affine(self.norm2(x), 1 + e[4], e[3]))
        x = fused_residual_mul_add(x, y, e[5])

        x = x.to(dtype)

        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.to(e.device) + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(fused_affine(self.norm(x), 1 + e[1], e[0]))
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(
            batch_size_vf, window_size_vf * blocks_vf * channels_vf
        )

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c * N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(
            batch_size_c * N_t, self.context_tokens, self.output_dim
        )

        # normalization and reshape
        with amp.autocast("cuda", dtype=torch.float32):
            context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # audio params
        audio_window=5,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        vae_scale=4,  # vae timedownsample scale
        norm_input_visual=True,
        norm_output_audio=True,
        weight_init=True,
    ):
        super().__init__()

        assert model_type == "i2v", "MultiTalk model requires your model_type is i2v."
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    output_dim=output_dim,
                    norm_input_visual=norm_input_visual,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.head_dim = dim // num_heads
        self.rope = VideoRopePosition3DEmb(
            head_dim=self.head_dim, len_h=1024, len_w=1024, len_t=1024
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)
        else:
            raise NotImplementedError("Not supported model type.")

        # init audio adapter
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )
        self._freqs_cache = {}
        self._audio_context_cache = None
        self._time_context_cache = {}

        # initialize weights
        if weight_init:
            self.init_weights()

    def _get_cached_freqs(self, freqs_shape: torch.Size, device: torch.device):
        cache_key = (tuple(freqs_shape), device.type, device.index)
        freqs = self._freqs_cache.get(cache_key)
        if freqs is None:
            freqs = self.rope.generate_embeddings(B_T_H_W_C=freqs_shape).to(device)
            self._freqs_cache[cache_key] = freqs
        return freqs

    def _get_audio_context(
        self, audio: torch.Tensor, device: torch.device, x_dtype: torch.dtype
    ):
        cache_key = (
            audio.data_ptr(),
            tuple(audio.shape),
            device.type,
            device.index,
            x_dtype,
        )
        if self._audio_context_cache is not None:
            cached_key, cached_audio_embedding, cached_human_num = self._audio_context_cache
            if cached_key == cache_key:
                return cached_audio_embedding, cached_human_num

        audio_cond = audio.to(device=device, dtype=x_dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]  # b 1 w s c
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(
            latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale
        )
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[
            :, :, :1, : middle_index + 1, ...
        ]
        latter_first_frame_audio_emb = rearrange(
            latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )  # b n_t (1 3) s c
        latter_last_frame_audio_emb = latter_frame_audio_emb[
            :, :, -1:, middle_index:, ...
        ]
        latter_last_frame_audio_emb = rearrange(
            latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )  # b n_t (1 3) s c
        latter_middle_frame_audio_emb = latter_frame_audio_emb[
            :, :, 1:-1, middle_index : middle_index + 1, ...
        ]
        latter_middle_frame_audio_emb = rearrange(
            latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )  # b n_t (2 1) s c

        latter_frame_audio_emb_s = torch.concat(
            [
                latter_first_frame_audio_emb,
                latter_middle_frame_audio_emb,
                latter_last_frame_audio_emb,
            ],
            dim=2,
        )
        audio_embedding = self.audio_proj(
            first_frame_audio_emb_s, latter_frame_audio_emb_s
        )
        human_num = len(audio_embedding)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x_dtype)

        self._audio_context_cache = (cache_key, audio_embedding, human_num)
        return audio_embedding, human_num

    def _get_time_context(self, t: torch.Tensor):
        cache_key = (t.data_ptr(), t.device.type, t.device.index)
        cached = self._time_context_cache.get(cache_key)
        if cached is not None:
            return cached

        with amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        self._time_context_cache[cache_key] = (e, e0)
        return e, e0

    def fuse_attention_projections(self):
        for block in self.blocks:
            block.self_attn.fuse_qkv()
            block.cross_attn.fuse_context_kv()
            block.cross_attn.fuse_image_kv()
        if ENABLE_FUSED_OP_LOGS:
            logger.info(
                "Fused attention projections across {} Wan blocks",
                len(self.blocks),
            )

    def clear_runtime_caches(self):
        for block in self.blocks:
            block.self_attn.clear_runtime_cache()
            block.cross_attn.clear_runtime_cache()
            block.audio_cross_attn.clear_runtime_cache()
        for module in self.modules():
            if isinstance(module, WanRMSNorm):
                module._triton_calls = 0
                module._triton_logged = False
        if ENABLE_FUSED_OP_LOGS:
            logger.info("Cleared Wan runtime caches before forward pass")

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        audio=None,
        ref_target_masks=None,
    ):
        assert clip_fea is not None and y is not None

        self.clear_runtime_caches()

        # params
        # device = self.patch_embedding.weight.device
        # if self.freqs.device != device:
        #     self.freqs = self.freqs.to(device)
        B = 1
        _, T, H, W = x[0].shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        freqs_shape = torch.Size([B, T, H // 2, W // 2, self.head_dim])
        self.freqs = self._get_cached_freqs(freqs_shape, x[0].device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x[0] = x[0].to(context[0].dtype)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]  # [1 fhw c]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings
        e, e0 = self._get_time_context(t)
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        # clip embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        audio_embedding, human_num = self._get_audio_context(audio, x.device, x.dtype)

        # convert ref_target_masks to token_ref_target_masks
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32)
            token_ref_target_masks = nn.functional.interpolate(
                ref_target_masks, size=(N_h, N_w), mode="nearest"
            )
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = token_ref_target_masks > 0
            token_ref_target_masks = token_ref_target_masks.view(
                token_ref_target_masks.shape[0], -1
            )
            token_ref_target_masks = token_ref_target_masks.to(x.dtype)
        else:
            token_ref_target_masks = None

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            ref_target_masks=token_ref_target_masks,
            human_num=human_num,
        )
        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if ENABLE_FUSED_OP_LOGS:
            fused_self_attn = sum(hasattr(block.self_attn, "qkv") for block in self.blocks)
            fused_context_kv = sum(hasattr(block.cross_attn, "kv") for block in self.blocks)
            fused_image_kv = sum(hasattr(block.cross_attn, "kv_img") for block in self.blocks)
            compiled_ffn_blocks = sum(
                getattr(block.ffn, "_compiled", False) for block in self.blocks
            )
            rmsnorm_triton_calls = sum(
                getattr(module, "_triton_calls", 0)
                for module in self.modules()
                if isinstance(module, WanRMSNorm)
            )
            logger.info(
                "Wan runtime summary: fused_self_attn={}, fused_context_kv={}, fused_image_kv={}, compiled_ffn_blocks={}, rmsnorm_triton_calls={}",
                fused_self_attn,
                fused_context_kv,
                fused_image_kv,
                compiled_ffn_blocks,
                rmsnorm_triton_calls,
            )

        return torch.stack(x).float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
