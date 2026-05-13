# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
import time
import yaml
from typing import Literal
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor
from loguru import logger

from osc_data.video import Video
from osc_data.image import Image
from osc_data.audio import Audio

from .models import WanVAE, CLIPModel, T5EncoderModel, Wav2Vec2Model
from .models.dit import WanModel, WanLayerNorm, WanRMSNorm
from .utils import (
    match_and_blend_colors_torch,
    resize_and_centercrop,
    loudness_norm,
)
from .quantize import quantize_model_a8w8_int8_gemlite, quantize_model_a8w4_hqq_gemlite
from .gemlite.core import GemLiteLinear
from .vram_management import (
    enable_vram_management,
    AutoWrappedLinear,
    AutoWrappedModule,
)
from .configs import multitalk_14B

# compile models to speedup inference
COMPILE_MODEL = True
COMPILE_VAE = True
# use parallel vae to speedup decode/encode
USE_PARALLEL_VAE = True


def to_param_dtype_fp32only(model, param_dtype):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32 and param.__class__.__name__ not in [
                "WeightQBytesTensor"
            ]:
                param.data = param.data.to(param_dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype == torch.float32 and buf.__class__.__name__ not in [
                "WeightQBytesTensor"
            ]:
                module._buffers[name] = buf.to(param_dtype)


def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class FlashTalkPipeline:
    def __init__(
        self,
        checkpoint_dir,
        wav2vec_dir,
        device="cuda",
        num_timesteps=1000,
        use_timestep_transform=True,
        num_persistent_param_in_dit=15_000_000_000,
        keep_dit_on_gpu=False,
        quantize_weights=True,
        weight_bits=8,
    ):
        """FlashTalkPipeline for RTX 4090 GPU.
        Args:
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            wav2vec_dir (`str`):
                Path to directory containing wav2vec checkpoints
            num_timesteps (`int`, *optional*, defaults to 1000):
                Number of timesteps.
            use_timestep_transform (`bool`, *optional*, defaults to True):
                Enable timestep transform.
            num_persistent_param_in_dit (`int`, *optional*, defaults to 15_000_000_000):
                Number of persistent parameters in DIT model.
            quantize_weights (`bool`, *optional*, defaults to True):
                Quantize DiT weights at load time using GemLite. Set to False to keep
                the checkpoint weights in their original dtype and rely on VRAM
                management only.
            weight_bits (`int`, *optional*, defaults to 8):
                DiT weight quantization bit-width. Supported values are 8 and 4.
        """
        self.device = device
        config = multitalk_14B
        self.config = config
        if quantize_weights and weight_bits not in (4, 8):
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}; expected 8 or 4."
            )

        with open(Path(__file__).parent / "configs" / "infer_params.yaml", "r") as f:
            self.infer_params = yaml.safe_load(f)
        self.rank = 0
        self.use_usp = False
        self.param_dtype = config.param_dtype
        self.cpu_offload = True
        self.keep_dit_on_gpu = keep_dit_on_gpu
        self.quantize_weights = quantize_weights
        self.weight_bits = weight_bits

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device="cpu" if self.cpu_offload else self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.base_latent_motion_frames = None
        self.previous_chunk_latent_tail = None
        self.window_anchor_cache = {}
        self.latent_carryover_steps = 0
        self.decoded_anchor_frames = 0
        self.window_memory_period = 0
        self.window_memory_strength = 0.0
        self.context_schedule = "uniform_standard"
        self.context_frames = 81
        self.context_stride = 4
        self.context_overlap = 16
        self.context_fuse_method = "linear"

        self.vae = WanVAE(
            vae_path=os.path.join(checkpoint_dir, config.vae_checkpoint),
            dtype=self.param_dtype,
            device="cpu" if self.cpu_offload else self.device,
            parallel=False,
        )

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device="cpu" if self.cpu_offload else self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer),
        )

        logger.info(f"Creating WanModel from {checkpoint_dir}")

        self.model = WanModel.from_pretrained(
            checkpoint_dir,
            device_map=self.device if self.keep_dit_on_gpu else "cpu",
            torch_dtype=self.param_dtype,
        )
        self.model.eval().requires_grad_(False)
        if self.quantize_weights:
            if self.weight_bits == 8:
                quantize_model_a8w8_int8_gemlite(self.model, device="cuda")
            elif self.weight_bits == 4:
                quantize_model_a8w4_hqq_gemlite(self.model, device="cuda")
            else:
                raise ValueError(
                    f"Unsupported weight_bits={self.weight_bits}; expected 8 or 4."
                )
        else:
            logger.info(
                "Skipping DiT weight quantization; using checkpoint weights as loaded."
            )
        if self.keep_dit_on_gpu:
            logger.info("Keeping DiT fully resident on GPU (no VRAM management).")
            self.vram_management = False
        else:
            self.model.cpu()
            torch.cuda.empty_cache()
            logger.info(
                f"Enable low vram mode with num_persistent_param_in_dit: {num_persistent_param_in_dit}"
            )
            self.vram_management = False
            self.enable_vram_management(
                num_persistent_param_in_dit=num_persistent_param_in_dit
            )

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_dir, local_files_only=True
        ).to("cpu")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_dir, local_files_only=True
        )

        self.model_names = ["model"]

    def enable_vram_management(self, num_persistent_param_in_dit=0):
        dtype = next(iter(self.model.parameters())).dtype
        logger.info(f"Enable vram management with dtype: {dtype}")
        enable_vram_management(
            self.model,
            module_map={
                GemLiteLinear: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()
        self.vram_management = True

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def onload_dit_model(self):
        start_time = time.time()
        if not self.cpu_offload:
            return
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if not isinstance(model, nn.Module):
                model = model.model
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        end_time = time.time()
        logger.info(f"Onload dit model time: {end_time - start_time} seconds")

    def offload_dit_model(self):
        start_time = time.time()
        if not self.cpu_offload:
            return
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if not isinstance(model, nn.Module):
                model = model.model
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    for module in model.modules():
                        if hasattr(module, "offload"):
                            module.offload()
                else:
                    model.to(self.device)
        torch.cuda.empty_cache()
        end_time = time.time()
        logger.info(f"Offload dit model time: {end_time - start_time} seconds")

    @torch.no_grad()
    def prepare_params(
        self,
        input_prompt: str,
        cond_image: Image,
        target_size: tuple[int, int] = (768, 448),
        frame_num: int = 33,
        motion_frames_num: int = 5,
        sampling_steps: int = 4,
        seed: int = 9999,
        shift: int = 5,
        color_correction_strength: float = 1.0,
    ):

        context = self.text_encoder(input_prompt, self.device)[0].to(self.device)

        self.frame_num = frame_num
        self.motion_frames_num = motion_frames_num
        self.target_h, self.target_w = target_size
        cond_image_tensor = torch.from_numpy(cond_image.load().to_rgb().data).permute(
            2, 0, 1
        )  # C, H, W
        # if isinstance(cond_image, str):
        #     cond_image = Image.open(cond_image).convert("RGB")
        cond_image_tensor = resize_and_centercrop(
            cond_image_tensor, (self.target_h, self.target_w)
        ).to(dtype=self.param_dtype, device=self.device)
        cond_image_tensor = (cond_image_tensor / 255 - 0.5) * 2
        self.cond_image_tensor = cond_image_tensor

        self.color_correction_strength = color_correction_strength
        self.original_color_reference = None
        if self.color_correction_strength > 0.0:
            self.original_color_reference = cond_image_tensor.clone()

        if self.cpu_offload:
            self.clip.model.to(self.device)
        clip_context = self.clip.visual(cond_image_tensor[:, :, -1:, :, :]).to(
            self.param_dtype
        )
        if self.cpu_offload:
            self.clip.model.cpu()
            torch.cuda.empty_cache()
        video_frames = torch.zeros(
            1,
            cond_image_tensor.shape[1],
            frame_num - cond_image_tensor.shape[2],
            self.target_h,
            self.target_w,
        ).to(dtype=self.param_dtype, device=self.device)
        padding_frames_pixels_values = torch.concat(
            [cond_image_tensor, video_frames], dim=2
        )
        if self.cpu_offload:
            self.vae.model.to(self.device)
            self.vae.scale[0] = self.vae.scale[0].to(self.device)
            self.vae.scale[1] = self.vae.scale[1].to(self.device)
        y = self.vae.encode(padding_frames_pixels_values)
        common_y = y.unsqueeze(0).to(self.param_dtype)

        # get mask
        self.lat_h, self.lat_w = (
            self.target_h // self.vae_stride[1],
            self.target_w // self.vae_stride[2],
        )
        msk = torch.ones(1, frame_num, self.lat_h, self.lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
        msk = msk.transpose(1, 2).to(self.param_dtype)

        y = torch.concat([msk, common_y], dim=1)

        max_seq_len = (
            ((frame_num - 1) // self.vae_stride[0] + 1)
            * self.lat_h
            * self.lat_w
            // (self.patch_size[1] * self.patch_size[2])
        )
        max_seq_len = int(math.ceil(max_seq_len / 1)) * 1

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # prepare timesteps
        if sampling_steps == 2:
            timesteps = [1000, 500]
        elif sampling_steps == 4:
            timesteps = [1000, 750, 500, 250]
        else:
            timesteps = list(
                np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32)
            )

        timesteps.append(0.0)
        timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps)
                for t in timesteps
            ]
        self.timesteps = timesteps

        self.arg_c = {
            "context": [context],
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": y,
            "ref_target_masks": None,
        }

        self.latent_motion_frames = self.vae.encode(self.cond_image_tensor)
        self.base_latent_motion_frames = self.latent_motion_frames.clone()
        self.window_anchor_cache = {}

        if self.cpu_offload:
            self.vae.model.cpu()
            torch.cuda.empty_cache()

        return

    @torch.no_grad()
    def preprocess_audio(self, speech_array, sr: int = 16000, fps: int = 25):
        video_length = len(speech_array) * fps / sr

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        if self.cpu_offload:
            self.audio_encoder.to(self.device)
        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=int(video_length), output_hidden_states=True
            )
        if self.cpu_offload:
            self.audio_encoder.cpu()
            torch.cuda.empty_cache()

        if len(embeddings) == 0:
            logger.error("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        return audio_emb

    def get_audio_embedding(self, audio_array, audio_start_idx=-1, audio_end_idx=-1):
        audio_array = loudness_norm(audio_array, self.infer_params["sample_rate"])
        audio_embedding = self.preprocess_audio(
            audio_array,
            sr=self.infer_params["sample_rate"],
            fps=self.infer_params["tgt_fps"],
        )

        if audio_start_idx == -1 or audio_end_idx == -1:
            audio_start_idx = 0
            audio_end_idx = audio_embedding.shape[0]

        indices = (torch.arange(2 * 2 + 1) - 2) * 1

        center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(
            1
        ) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx - 1)

        audio_embedding = audio_embedding[center_indices][None, ...].contiguous()
        return audio_embedding

    @torch.no_grad()
    def generate_chunk(self, audio_embedding):
        """
        Generate a chunk of video from the audio embedding.
        Args:
            audio_embedding: The audio embedding.
        Returns:
            The generated video
        """
        # evaluation mode
        with torch.no_grad():
            self.arg_c.update(
                {
                    "audio": audio_embedding,
                }
            )

            # sample videos
            latent = torch.randn(
                16,
                (self.frame_num - 1) // 4 + 1,
                self.lat_h,
                self.lat_w,
                dtype=self.param_dtype,
                device=self.device,
                generator=self.generator,
            )

            latent[:, : self.latent_motion_frames.shape[1]] = self.latent_motion_frames
            latent = self._apply_latent_carryover(latent)

            for i in range(len(self.timesteps) - 1):
                timestep = self.timesteps[i]
                latent_model_input = [latent]
                # onload dit model
                if self.vram_management:
                    self.onload_dit_model()

                # inference without CFG
                start_time = time.perf_counter()
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **self.arg_c
                )[0]
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                logger.info(f"model denoise per step: {end_time - start_time:.2f}s")

                noise_pred = -noise_pred_cond

                # update latent
                t_i = self.timesteps[i][:, None, None, None] / self.num_timesteps
                t_i_1 = self.timesteps[i + 1][:, None, None, None] / self.num_timesteps
                x_0 = latent + noise_pred * t_i

                latent = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(
                    x_0.size(),
                    dtype=x_0.dtype,
                    device=self.device,
                    generator=self.generator,
                )

                latent[:, : self.latent_motion_frames.shape[1]] = (
                    self.latent_motion_frames
                )

            # self.offload_dit_model()

            if self.cpu_offload:
                self.vae.model.to(self.device)
                self.vae.scale[0] = self.vae.scale[0].to(self.device)
                self.vae.scale[1] = self.vae.scale[1].to(self.device)

            torch.cuda.synchronize()
            start_decode_time = time.time()
            videos = self.vae.decode(latent.to(self.param_dtype))
            torch.cuda.synchronize()
            end_decode_time = time.time()
            logger.info(
                f"decode video frames: {end_decode_time - start_decode_time:.2f}s"
            )

        torch.cuda.synchronize()
        start_color_correction_time = time.time()
        if self.color_correction_strength > 0.0:
            videos = match_and_blend_colors_torch(
                videos, self.original_color_reference, self.color_correction_strength
            )

        anchor_frames = max(int(self.motion_frames_num), int(self.decoded_anchor_frames))
        anchor_frames = min(anchor_frames, int(videos.shape[2]))
        cond_frame = videos[:, :, -anchor_frames:].to(self.device)
        torch.cuda.synchronize()
        end_color_correction_time = time.time()
        logger.info(
            f"color correction: {end_color_correction_time - start_color_correction_time:.2f}s"
        )

        torch.cuda.synchronize()
        start_encode_time = time.time()
        self.latent_motion_frames = self.vae.encode(cond_frame)
        torch.cuda.synchronize()
        end_encode_time = time.time()
        logger.info(
            "encode motion frames: {:.2f}s (anchor_frames={}, latent_steps={})",
            end_encode_time - start_encode_time,
            anchor_frames,
            int(self.latent_motion_frames.shape[1]),
        )
        self._update_latent_carryover_cache(latent)

        if self.cpu_offload:
            self.vae.model.cpu()
            torch.cuda.empty_cache()

        gen_video_samples = videos  # [:, :, self.motion_frames_num:]
        gen_video_samples = gen_video_samples[0].to(torch.float32)
        gen_video_samples = (
            ((gen_video_samples + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255
        ).contiguous()
        return gen_video_samples

    def _restore_reference_motion_anchor(self, reason: str) -> None:
        if self.base_latent_motion_frames is None:
            logger.warning("Requested motion anchor restore before reference latent was set.")
            return

        self.latent_motion_frames = self.base_latent_motion_frames.clone()
        self.previous_chunk_latent_tail = None
        logger.info("Restored reference motion anchor and cleared latent carryover cache: {}", reason)

    def _apply_latent_carryover(self, latent: torch.Tensor) -> torch.Tensor:
        carryover_steps = int(self.latent_carryover_steps)
        if carryover_steps <= 0 or self.previous_chunk_latent_tail is None:
            return latent

        motion_seed_steps = int(self.latent_motion_frames.shape[1])
        available_steps = latent.shape[1] - motion_seed_steps
        if available_steps <= 0:
            return latent

        carryover_steps = min(
            carryover_steps,
            available_steps,
            int(self.previous_chunk_latent_tail.shape[1]),
        )
        if carryover_steps <= 0:
            return latent

        latent[:, motion_seed_steps : motion_seed_steps + carryover_steps] = (
            self.previous_chunk_latent_tail[:, -carryover_steps:]
        )
        logger.info(
            "Applied latent carryover with {} temporal step(s) (requested={}, motion_seed_steps={}, available_steps={}).",
            carryover_steps,
            int(self.latent_carryover_steps),
            motion_seed_steps,
            available_steps,
        )
        return latent

    def _update_latent_carryover_cache(self, latent: torch.Tensor) -> None:
        carryover_steps = int(self.latent_carryover_steps)
        if carryover_steps <= 0:
            self.previous_chunk_latent_tail = None
            return

        carryover_steps = min(carryover_steps, int(latent.shape[1]))
        if carryover_steps <= 0:
            self.previous_chunk_latent_tail = None
            return

        self.previous_chunk_latent_tail = latent[:, -carryover_steps:].detach().clone()
        logger.info(
            "Cached latent carryover tail with {} temporal step(s) (requested={}).",
            carryover_steps,
            int(self.latent_carryover_steps),
        )

    def _apply_window_memory_anchor(
        self,
        phase_idx: int,
        window_memory_strength: float,
    ) -> bool:
        if window_memory_strength <= 0.0:
            return False

        cached_anchor = self.window_anchor_cache.get(phase_idx)
        if cached_anchor is None:
            return False

        if cached_anchor.shape != self.latent_motion_frames.shape:
            logger.warning(
                "Skipping window memory anchor for phase {} due to shape mismatch (cached={}, current={}).",
                phase_idx,
                tuple(cached_anchor.shape),
                tuple(self.latent_motion_frames.shape),
            )
            return False

        current_anchor = self.latent_motion_frames
        cached_anchor = cached_anchor.to(current_anchor)

        latent_steps = int(current_anchor.shape[1])
        overlap_steps = min(
            latent_steps,
            max(1, int(round(self.context_overlap / 4.0))),
        )
        if self.context_fuse_method == "pyramid":
            ramp = torch.arange(1, overlap_steps + 1, device=current_anchor.device, dtype=current_anchor.dtype)
            if overlap_steps > 1:
                weights = torch.cat([ramp, torch.flip(ramp[:-1], dims=[0])], dim=0)
            else:
                weights = ramp
            weights = weights / weights.max().clamp_min(1e-8)
            if weights.shape[0] < latent_steps:
                pad = torch.zeros(latent_steps - weights.shape[0], device=current_anchor.device, dtype=current_anchor.dtype)
                weights = torch.cat([weights, pad], dim=0)
            elif weights.shape[0] > latent_steps:
                weights = weights[:latent_steps]
        else:
            weights = torch.linspace(
                0.0,
                1.0,
                steps=latent_steps,
                device=current_anchor.device,
                dtype=current_anchor.dtype,
            )
        weights = weights.view(1, latent_steps, 1, 1)

        mix = float(window_memory_strength)
        self.latent_motion_frames = current_anchor * (1 - weights * mix) + cached_anchor * (weights * mix)
        logger.info(
            "Applied window memory anchor for phase {} with mix {:.3f}, fuse_method={}, overlap_steps={}.",
            phase_idx,
            mix,
            self.context_fuse_method,
            overlap_steps,
        )
        return True

    def _update_window_memory_anchor(self, phase_idx: int) -> None:
        self.window_anchor_cache[phase_idx] = self.latent_motion_frames.detach().clone()
        logger.info(
            "Cached window memory anchor for phase {} with {} latent step(s).",
            phase_idx,
            int(self.latent_motion_frames.shape[1]),
        )

    def _get_context_phase_index(self, chunk_idx: int, total_chunks: int) -> int:
        phase_count = max(1, math.ceil(self.context_frames / max(int(self.frame_num), 1)))
        if self.context_schedule == "uniform_looped":
            stride = max(1, int(round(self.context_stride / 4.0)))
            return (chunk_idx * stride) % phase_count
        if self.context_schedule == "uniform_standard":
            return chunk_idx % phase_count
        return min(chunk_idx, phase_count - 1)

    @staticmethod
    def _to_uint8_image(frame: torch.Tensor) -> np.ndarray:
        image = frame.detach().cpu().numpy()
        if image.dtype != np.uint8:
            image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _phase_correlation_shift(
        reference_frame: np.ndarray, moving_frame: np.ndarray
    ) -> tuple[int, int]:
        reference_gray = reference_frame.astype(np.float32).mean(axis=-1)
        moving_gray = moving_frame.astype(np.float32).mean(axis=-1)

        reference_gray -= reference_gray.mean()
        moving_gray -= moving_gray.mean()

        reference_fft = np.fft.fft2(reference_gray)
        moving_fft = np.fft.fft2(moving_gray)
        cross_power = reference_fft * np.conj(moving_fft)
        cross_power /= np.maximum(np.abs(cross_power), 1e-8)
        correlation = np.fft.ifft2(cross_power)
        peak_y, peak_x = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)

        if peak_y > reference_gray.shape[0] // 2:
            peak_y -= reference_gray.shape[0]
        if peak_x > reference_gray.shape[1] // 2:
            peak_x -= reference_gray.shape[1]

        return int(peak_x), int(peak_y)

    @staticmethod
    def _estimate_boundary_shift(
        previous_tail: torch.Tensor,
        current_head: torch.Tensor,
    ) -> tuple[int, int, str]:
        reference_frame = FlashTalkPipeline._to_uint8_image(previous_tail[-1])
        moving_frame = FlashTalkPipeline._to_uint8_image(current_head[0])

        try:
            import cv2  # type: ignore

            reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_RGB2GRAY)
            moving_gray = cv2.cvtColor(moving_frame, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                reference_gray,
                moving_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            shift_x = int(np.rint(np.median(flow[..., 0])))
            shift_y = int(np.rint(np.median(flow[..., 1])))
            return shift_x, shift_y, "optical_flow"
        except Exception:
            shift_x, shift_y = FlashTalkPipeline._phase_correlation_shift(
                reference_frame, moving_frame
            )
            return shift_x, shift_y, "phase_correlation"

    @staticmethod
    def _translate_frames(frames: torch.Tensor, shift_x: int, shift_y: int) -> torch.Tensor:
        if shift_x == 0 and shift_y == 0:
            return frames
        import torch.nn.functional as F

        frames_bchw = frames.permute(0, 3, 1, 2).contiguous()
        _, _, height, width = frames_bchw.shape
        theta = torch.tensor(
            [
                [1.0, 0.0, -2.0 * float(shift_x) / max(width - 1, 1)],
                [0.0, 1.0, -2.0 * float(shift_y) / max(height - 1, 1)],
            ],
            dtype=frames_bchw.dtype,
            device=frames_bchw.device,
        ).unsqueeze(0).repeat(frames_bchw.shape[0], 1, 1)
        grid = F.affine_grid(theta, frames_bchw.shape, align_corners=False)
        translated = F.grid_sample(
            frames_bchw,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return translated.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _temporal_smooth_frames(
        frames: torch.Tensor,
        smoothing_frames: int,
    ) -> torch.Tensor:
        smoothing_frames = int(smoothing_frames)
        if smoothing_frames <= 0 or frames.shape[0] <= 1:
            return frames

        radius = min(smoothing_frames, frames.shape[0] - 1)
        if radius <= 0:
            return frames

        weights = torch.arange(
            1, radius + 2, device=frames.device, dtype=frames.dtype
        )
        if radius > 0:
            weights = torch.cat([weights, torch.flip(weights[:-1], dims=[0])], dim=0)
        weights = weights / weights.sum()

        padded = torch.cat(
            [frames[:1].repeat(radius, 1, 1, 1), frames, frames[-1:].repeat(radius, 1, 1, 1)],
            dim=0,
        )
        smoothed_frames = []
        for idx in range(frames.shape[0]):
            window = padded[idx : idx + weights.shape[0]]
            smoothed_frames.append((window * weights.view(-1, 1, 1, 1)).sum(dim=0))
        return torch.stack(smoothed_frames, dim=0)

    def _repair_chunk_seams(
        self,
        chunks: list[torch.Tensor],
        seam_frames: int,
        smoothing_frames: int,
        boundary_alignment: bool,
    ) -> list[torch.Tensor]:
        if len(chunks) <= 1 or seam_frames <= 0:
            return chunks

        repaired: list[torch.Tensor] = [chunks[0].clone()]
        for chunk_idx, next_chunk in enumerate(chunks[1:], start=1):
            previous_chunk = repaired[-1]
            repair_frames = min(
                int(seam_frames),
                int(previous_chunk.shape[0]),
                int(next_chunk.shape[0]),
            )
            if repair_frames <= 0:
                repaired.append(next_chunk.clone())
                continue

            previous_tail = previous_chunk[-repair_frames:].clone()
            current_head = next_chunk[:repair_frames].clone()
            alignment_method = "disabled"
            shift_x = 0
            shift_y = 0
            if boundary_alignment:
                shift_x, shift_y, alignment_method = self._estimate_boundary_shift(
                    previous_tail, current_head
                )
                current_head = self._translate_frames(current_head, shift_x, shift_y)

            blended_head = self._apply_temporal_crossfade(
                previous_tail,
                current_head,
                repair_frames,
            )
            blended_head = self._temporal_smooth_frames(blended_head, smoothing_frames)

            repaired[-1] = torch.cat([previous_chunk[:-repair_frames], blended_head], dim=0)
            repaired.append(next_chunk[repair_frames:].clone())
            logger.info(
                "Post-process seam repair for chunk-{}: repair_frames={}, smoothing_frames={}, boundary_alignment={}, alignment_method={}, shift_x={}, shift_y={}.",
                chunk_idx,
                repair_frames,
                smoothing_frames,
                boundary_alignment,
                alignment_method,
                shift_x,
                shift_y,
            )

        return repaired

    @staticmethod
    def _compute_boundary_drift_score(
        previous_tail: torch.Tensor | None,
        current_head: torch.Tensor | None,
        overlap_frames: int,
    ) -> float:
        if previous_tail is None or current_head is None:
            return 0.0

        overlap_frames = min(
            overlap_frames,
            int(previous_tail.shape[0]),
            int(current_head.shape[0]),
        )
        if overlap_frames <= 0:
            return 0.0

        prev = previous_tail[-overlap_frames:].to(torch.float32)
        curr = current_head[:overlap_frames].to(torch.float32)
        return torch.mean(torch.abs(prev - curr)).item() / 255.0

    @staticmethod
    def _apply_temporal_crossfade(
        previous_chunk: torch.Tensor,
        current_chunk: torch.Tensor,
        blend_frames: int,
    ) -> torch.Tensor:
        blend_frames = min(
            blend_frames,
            int(previous_chunk.shape[0]),
            int(current_chunk.shape[0]),
        )
        if blend_frames <= 0:
            return previous_chunk

        blend_weights = torch.linspace(
            0.0,
            1.0,
            steps=blend_frames,
            device=previous_chunk.device,
            dtype=previous_chunk.dtype,
        ).view(-1, 1, 1, 1)
        blended_tail = previous_chunk[-blend_frames:] * (1 - blend_weights) + current_chunk[
            :blend_frames
        ] * blend_weights
        return torch.cat([previous_chunk[:-blend_frames], blended_tail], dim=0)

    def generate(
        self,
        input_prompt: str,
        audio: Audio,
        image: Image,
        audio_encode_mode: Literal["stream", "once"] = "once",
        frame_num: int | None = None,
        motion_frames_num: int | None = None,
        sampling_steps: int | None = None,
        color_correction_strength: float | None = None,
        cached_audio_duration: int | None = None,
        temporal_crossfade_frames: int | None = None,
        reanchor_every_n_chunks: int | None = None,
        adaptive_drift_refresh: bool | None = None,
        drift_refresh_threshold: float | None = None,
        latent_carryover_steps: int | None = None,
        postprocess_seam_repair_frames: int | None = None,
        postprocess_temporal_smoothing_frames: int | None = None,
        postprocess_boundary_alignment: bool | None = None,
        decoded_anchor_frames: int | None = None,
        window_memory_period: int | None = None,
        window_memory_strength: float | None = None,
        context_schedule: str | None = None,
        context_frames: int | None = None,
        context_stride: int | None = None,
        context_overlap: int | None = None,
        context_fuse_method: Literal["linear", "pyramid"] | None = None,
    ) -> Video:
        """
        Generate video from the audio and image prompt.

        This method processes audio input along with a conditioning image to generate
        a talking head video. The audio is processed in chunks and each chunk generates
        corresponding video frames.

        Args:
            input_prompt: The input text prompt describing the desired video generation.
            image: The conditioning image used as visual reference.
            audio: The audio file (wav format) that drives the lip movements.
            audio_encode_mode: Strategy for encoding audio, either "stream" or "once".
                - "stream": Process audio in streaming chunks (default), memory efficient.
                - "once": Encode entire audio at once then split into chunks.
            frame_num: Optional override for the chunk length in frames.
            motion_frames_num: Optional override for how many tail frames carry into the next chunk.
            sampling_steps: Optional override for denoising steps per chunk.
            color_correction_strength: Optional override for chunk color blending strength.
            cached_audio_duration: Optional override for the streaming audio cache window.
            temporal_crossfade_frames: Optional override for temporal crossfade strength at chunk joins.
            reanchor_every_n_chunks: Optional override for periodic latent re-anchoring cadence.
            adaptive_drift_refresh: Optional override to enable adaptive latent refresh when drift is high.
            drift_refresh_threshold: Optional override for the normalized drift score threshold.
            latent_carryover_steps: Optional override for how many latent time steps to carry from the prior chunk.
            postprocess_seam_repair_frames: Optional override for seam repair overlap after generation.
            postprocess_temporal_smoothing_frames: Optional override for temporal smoothing radius in seam repair.
            postprocess_boundary_alignment: Optional override to enable boundary alignment before seam repair.
            decoded_anchor_frames: Optional override for how many decoded frames anchor the next chunk latent.
            window_memory_period: Optional override for the chunk-phase period used to reuse reference latents.
            window_memory_strength: Optional override for how strongly the cached reference latent is blended in.
            context_schedule: Optional InfiniteTalk-style schedule name for window memory policy.
            context_frames: Optional InfiniteTalk-style context window size in pixel frames.
            context_stride: Optional InfiniteTalk-style window stride in pixel frames.
            context_overlap: Optional InfiniteTalk-style window overlap in pixel frames.
            context_fuse_method: Optional InfiniteTalk-style window fuse method.
            video_save_path: Path to save the generated video.
            merge_video_audio: Whether to merge the generated video and the original audio.
            force_9_16: Whether to force the video to be 9:16.

        Returns:
            List[torch.Tensor]: A list of video frame tensors, where each tensor has:
                - Shape: (num_frames, height, width, 3)
                - Dtype: torch.float32
                - Value range: [0, 255]
                - Device: CPU
                - Channel order: RGB (channels last)

                For example, with 768x448 resolution, each tensor shape is:
                - First chunk: (33, 768, 448, 3)
                - Subsequent chunks: (28, 768, 448, 3) or (23, 768, 448, 3) for last chunk
        """
        generate_start_time = time.perf_counter()
        logger.info("Start to generate video...")
        # prepare data
        sample_rate = self.infer_params["sample_rate"]
        tgt_fps = self.infer_params["tgt_fps"]
        cached_audio_duration = (
            self.infer_params["cached_audio_duration"]
            if cached_audio_duration is None
            else cached_audio_duration
        )
        frame_num = self.infer_params["frame_num"] if frame_num is None else frame_num
        motion_frames_num = (
            self.infer_params["motion_frames_num"]
            if motion_frames_num is None
            else motion_frames_num
        )
        sampling_steps = (
            self.infer_params["sample_steps"]
            if sampling_steps is None
            else sampling_steps
        )
        color_correction_strength = (
            self.infer_params["color_correction_strength"]
            if color_correction_strength is None
            else color_correction_strength
        )
        temporal_crossfade_frames = (
            self.infer_params["temporal_crossfade_frames"]
            if temporal_crossfade_frames is None
            else temporal_crossfade_frames
        )
        reanchor_every_n_chunks = (
            self.infer_params["reanchor_every_n_chunks"]
            if reanchor_every_n_chunks is None
            else reanchor_every_n_chunks
        )
        adaptive_drift_refresh = (
            self.infer_params["adaptive_drift_refresh"]
            if adaptive_drift_refresh is None
            else adaptive_drift_refresh
        )
        drift_refresh_threshold = (
            self.infer_params["drift_refresh_threshold"]
            if drift_refresh_threshold is None
            else drift_refresh_threshold
        )
        latent_carryover_steps = (
            self.infer_params["latent_carryover_steps"]
            if latent_carryover_steps is None
            else latent_carryover_steps
        )
        postprocess_seam_repair_frames = (
            self.infer_params["postprocess_seam_repair_frames"]
            if postprocess_seam_repair_frames is None
            else postprocess_seam_repair_frames
        )
        postprocess_temporal_smoothing_frames = (
            self.infer_params["postprocess_temporal_smoothing_frames"]
            if postprocess_temporal_smoothing_frames is None
            else postprocess_temporal_smoothing_frames
        )
        postprocess_boundary_alignment = (
            self.infer_params["postprocess_boundary_alignment"]
            if postprocess_boundary_alignment is None
            else postprocess_boundary_alignment
        )
        decoded_anchor_frames = (
            self.infer_params["decoded_anchor_frames"]
            if decoded_anchor_frames is None
            else decoded_anchor_frames
        )
        window_memory_period = (
            self.infer_params["window_memory_period"]
            if window_memory_period is None
            else window_memory_period
        )
        window_memory_strength = (
            self.infer_params["window_memory_strength"]
            if window_memory_strength is None
            else window_memory_strength
        )
        context_schedule = (
            self.infer_params["context_schedule"]
            if context_schedule is None
            else context_schedule
        )
        context_frames = (
            self.infer_params["context_frames"]
            if context_frames is None
            else context_frames
        )
        context_stride = (
            self.infer_params["context_stride"]
            if context_stride is None
            else context_stride
        )
        context_overlap = (
            self.infer_params["context_overlap"]
            if context_overlap is None
            else context_overlap
        )
        context_fuse_method = (
            self.infer_params["context_fuse_method"]
            if context_fuse_method is None
            else context_fuse_method
        )

        if temporal_crossfade_frames < 0:
            raise ValueError("temporal_crossfade_frames must be >= 0")
        if reanchor_every_n_chunks < 0:
            raise ValueError("reanchor_every_n_chunks must be >= 0")
        if drift_refresh_threshold < 0:
            raise ValueError("drift_refresh_threshold must be >= 0")
        if latent_carryover_steps < 0:
            raise ValueError("latent_carryover_steps must be >= 0")
        if postprocess_seam_repair_frames < 0:
            raise ValueError("postprocess_seam_repair_frames must be >= 0")
        if postprocess_temporal_smoothing_frames < 0:
            raise ValueError("postprocess_temporal_smoothing_frames must be >= 0")
        if decoded_anchor_frames < 0:
            raise ValueError("decoded_anchor_frames must be >= 0")
        if window_memory_period < 0:
            raise ValueError("window_memory_period must be >= 0")
        if not 0.0 <= float(window_memory_strength) <= 1.0:
            raise ValueError("window_memory_strength must be between 0 and 1")
        if context_frames < 0:
            raise ValueError("context_frames must be >= 0")
        if context_stride < 0:
            raise ValueError("context_stride must be >= 0")
        if context_overlap < 0:
            raise ValueError("context_overlap must be >= 0")
        if postprocess_seam_repair_frames > 0 and postprocess_temporal_smoothing_frames == 0:
            logger.info("Post-process seam repair enabled without temporal smoothing; set postprocess_temporal_smoothing_frames > 0 for seam denoising.")
        if postprocess_boundary_alignment and postprocess_seam_repair_frames == 0:
            logger.warning(
                "postprocess_boundary_alignment is enabled but postprocess_seam_repair_frames is 0; alignment will not run without a seam repair window."
            )
        if frame_num <= motion_frames_num:
            raise ValueError("frame_num must be greater than motion_frames_num")

        slice_len = frame_num - motion_frames_num
        self.latent_carryover_steps = latent_carryover_steps
        self.decoded_anchor_frames = decoded_anchor_frames
        self.window_memory_period = window_memory_period
        self.window_memory_strength = float(window_memory_strength)
        self.context_schedule = context_schedule
        self.context_frames = int(context_frames)
        self.context_stride = int(context_stride)
        self.context_overlap = int(context_overlap)
        self.context_fuse_method = context_fuse_method
        self.previous_chunk_latent_tail = None
        self.window_anchor_cache = {}
        self.prepare_params(
            input_prompt=input_prompt,
            cond_image=image,
            frame_num=frame_num,
            motion_frames_num=motion_frames_num,
            sampling_steps=sampling_steps,
            color_correction_strength=color_correction_strength,
        )

        logger.info(
            "Boundary controls: temporal_crossfade_frames={}, reanchor_every_n_chunks={}, adaptive_drift_refresh={}, drift_refresh_threshold={:.4f}, latent_carryover_steps={}, postprocess_seam_repair_frames={}, postprocess_temporal_smoothing_frames={}, postprocess_boundary_alignment={}, decoded_anchor_frames={}",
            temporal_crossfade_frames,
            reanchor_every_n_chunks,
            adaptive_drift_refresh,
            drift_refresh_threshold,
            latent_carryover_steps,
            postprocess_seam_repair_frames,
            postprocess_temporal_smoothing_frames,
            postprocess_boundary_alignment,
            decoded_anchor_frames,
        )
        logger.info(
            "Window memory: period={}, strength={:.3f}",
            window_memory_period,
            window_memory_strength,
        )
        logger.info(
            "InfiniteTalk context defaults: schedule={}, context_frames={}, context_stride={}, context_overlap={}, fuse_method={}",
            context_schedule,
            context_frames,
            context_stride,
            context_overlap,
            context_fuse_method,
        )

        human_speech_array_all = audio.load(
            sample_rate=self.infer_params["sample_rate"], mono=True
        ).data
        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_frame_num = frame_num * sample_rate // tgt_fps
        data_prepare_time = time.perf_counter()
        logger.info(
            f"Data preparation time: {data_prepare_time - generate_start_time:.2f}s"
        )

        generated_list = []
        previous_stitched_tail = None
        force_reanchor_next_chunk = False
        if audio_encode_mode == "once":
            # pad audio with silence to avoid truncating the last chunk
            remainder = (
                len(human_speech_array_all) - human_speech_array_frame_num
            ) % human_speech_array_slice_len
            if remainder > 0:
                pad_length = human_speech_array_slice_len - remainder
                human_speech_array_all = np.concatenate(
                    [
                        human_speech_array_all,
                        np.zeros(pad_length, dtype=human_speech_array_all.dtype),
                    ]
                )

            # encode audio together
            audio_embedding_all = self.get_audio_embedding(human_speech_array_all)

            # split audio embedding into chunks: 33, 28, 28, 28, ...
            audio_embedding_len = audio_embedding_all.shape[1]
            chunk_count = (audio_embedding_len - frame_num + slice_len) // slice_len
            audio_embedding_chunks_list = [
                audio_embedding_all[
                    :, i * slice_len : i * slice_len + frame_num
                ].contiguous()
                for i in range(chunk_count)
            ]

            for chunk_idx, audio_embedding_chunk in enumerate(
                audio_embedding_chunks_list
            ):
                reanchored_this_chunk = False
                if chunk_idx > 0:
                    should_periodic_reanchor = (
                        reanchor_every_n_chunks > 0
                        and chunk_idx % reanchor_every_n_chunks == 0
                    )
                    if force_reanchor_next_chunk or should_periodic_reanchor:
                        reason = (
                            "adaptive drift refresh"
                            if force_reanchor_next_chunk
                            else f"periodic cadence every {reanchor_every_n_chunks} chunk(s)"
                        )
                        self._restore_reference_motion_anchor(reason)
                        force_reanchor_next_chunk = False
                        reanchored_this_chunk = True

                    if (
                        window_memory_period > 0
                        and window_memory_strength > 0.0
                        and not reanchored_this_chunk
                    ):
                        phase_idx = self._get_context_phase_index(
                            chunk_idx, chunk_count
                        )
                        if self._apply_window_memory_anchor(
                            phase_idx, window_memory_strength
                        ):
                            logger.info(
                                "Chunk-{} reused window memory phase {} before generation.",
                                chunk_idx,
                                phase_idx,
                            )

                torch.cuda.synchronize()
                start_time = time.time()

                # inference
                video = self.generate_chunk(audio_embedding_chunk)
                video = video.cpu()
                drift_score = 0.0

                if chunk_idx == 0:
                    previous_stitched_tail = video[-motion_frames_num:].clone()
                else:
                    current_head = video[:motion_frames_num]
                    drift_score = self._compute_boundary_drift_score(
                        previous_stitched_tail,
                        current_head,
                        motion_frames_num,
                    )
                    logger.info(
                        "Chunk-{} boundary drift score: {:.4f}",
                        chunk_idx,
                        drift_score,
                    )
                    if adaptive_drift_refresh and drift_score >= drift_refresh_threshold:
                        force_reanchor_next_chunk = True
                        logger.warning(
                            "Chunk-{} drift score {:.4f} exceeded threshold {:.4f}; scheduling reference latent refresh before chunk-{}.",
                            chunk_idx,
                            drift_score,
                            drift_refresh_threshold,
                            chunk_idx + 1,
                        )

                    if temporal_crossfade_frames > 0:
                        blend_frames = min(temporal_crossfade_frames, motion_frames_num)
                        generated_list[-1] = self._apply_temporal_crossfade(
                            generated_list[-1],
                            video,
                            blend_frames,
                        )
                        logger.info(
                            "Chunk-{} temporal crossfade applied with {} frame(s) (requested={}).",
                            chunk_idx,
                            blend_frames,
                            temporal_crossfade_frames,
                        )

                    video = video[motion_frames_num:]
                    previous_stitched_tail = video[-motion_frames_num:].clone()

                if window_memory_period > 0 and window_memory_strength > 0.0:
                    phase_idx = self._get_context_phase_index(
                        chunk_idx, chunk_count
                    )
                    if chunk_idx > 0 and adaptive_drift_refresh and drift_score >= drift_refresh_threshold:
                        logger.info(
                            "Skipping window memory cache update for phase {} due to elevated drift score {:.4f}.",
                            phase_idx,
                            drift_score,
                        )
                    else:
                        self._update_window_memory_anchor(phase_idx)

                torch.cuda.synchronize()
                end_time = time.time()
                logger.info(
                    f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s"
                )

                generated_list.append(video)
                torch.cuda.empty_cache()

        elif audio_encode_mode == "stream":
            cached_audio_length_sum = sample_rate * cached_audio_duration
            audio_end_idx = cached_audio_duration * tgt_fps
            audio_start_idx = audio_end_idx - frame_num

            audio_dq = deque(
                [0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum
            )

            # pad audio with silence to avoid truncating the last chunk
            remainder = len(human_speech_array_all) % human_speech_array_slice_len
            if remainder > 0:
                pad_length = human_speech_array_slice_len - remainder
                human_speech_array_all = np.concatenate(
                    [
                        human_speech_array_all,
                        np.zeros(pad_length, dtype=human_speech_array_all.dtype),
                    ]
                )

            # split audio embedding into chunks: 28, 28, 28, 28, ...
            human_speech_array_slices = human_speech_array_all.reshape(
                -1, human_speech_array_slice_len
            )
            total_chunks = human_speech_array_slices.shape[0]

            for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
                reanchored_this_chunk = False
                if chunk_idx > 0:
                    should_periodic_reanchor = (
                        reanchor_every_n_chunks > 0
                        and chunk_idx % reanchor_every_n_chunks == 0
                    )
                    if force_reanchor_next_chunk or should_periodic_reanchor:
                        reason = (
                            "adaptive drift refresh"
                            if force_reanchor_next_chunk
                            else f"periodic cadence every {reanchor_every_n_chunks} chunk(s)"
                        )
                        self._restore_reference_motion_anchor(reason)
                        force_reanchor_next_chunk = False
                        reanchored_this_chunk = True

                    if (
                        window_memory_period > 0
                        and window_memory_strength > 0.0
                        and not reanchored_this_chunk
                    ):
                        phase_idx = self._get_context_phase_index(
                            chunk_idx, total_chunks
                        )
                        if self._apply_window_memory_anchor(
                            phase_idx, window_memory_strength
                        ):
                            logger.info(
                                "Chunk-{} reused window memory phase {} before generation.",
                                chunk_idx,
                                phase_idx,
                            )

                # streaming encode audio chunks
                audio_dq.extend(human_speech_array.tolist())
                audio_array = np.array(audio_dq)
                audio_embedding = self.get_audio_embedding(
                    audio_array, audio_start_idx, audio_end_idx
                )

                torch.cuda.synchronize()
                start_time = time.time()

                # inference
                video = self.generate_chunk(audio_embedding)
                video = video.cpu()
                drift_score = 0.0

                if chunk_idx == 0:
                    previous_stitched_tail = video[-motion_frames_num:].clone()
                else:
                    current_head = video[:motion_frames_num]
                    drift_score = self._compute_boundary_drift_score(
                        previous_stitched_tail,
                        current_head,
                        motion_frames_num,
                    )
                    logger.info(
                        "Chunk-{} boundary drift score: {:.4f}",
                        chunk_idx,
                        drift_score,
                    )
                    if adaptive_drift_refresh and drift_score >= drift_refresh_threshold:
                        force_reanchor_next_chunk = True
                        logger.warning(
                            "Chunk-{} drift score {:.4f} exceeded threshold {:.4f}; scheduling reference latent refresh before chunk-{}.",
                            chunk_idx,
                            drift_score,
                            drift_refresh_threshold,
                            chunk_idx + 1,
                        )

                    if temporal_crossfade_frames > 0:
                        blend_frames = min(temporal_crossfade_frames, motion_frames_num)
                        generated_list[-1] = self._apply_temporal_crossfade(
                            generated_list[-1],
                            video,
                            blend_frames,
                        )
                        logger.info(
                            "Chunk-{} temporal crossfade applied with {} frame(s) (requested={}).",
                            chunk_idx,
                            blend_frames,
                            temporal_crossfade_frames,
                        )

                    video = video[motion_frames_num:]
                    previous_stitched_tail = video[-motion_frames_num:].clone()

                if window_memory_period > 0 and window_memory_strength > 0.0:
                    phase_idx = self._get_context_phase_index(
                        chunk_idx, total_chunks
                    )
                    if chunk_idx > 0 and adaptive_drift_refresh and drift_score >= drift_refresh_threshold:
                        logger.info(
                            "Skipping window memory cache update for phase {} due to elevated drift score {:.4f}.",
                            phase_idx,
                            drift_score,
                        )
                    else:
                        self._update_window_memory_anchor(phase_idx)

                torch.cuda.synchronize()
                end_time = time.time()
                logger.info(
                    f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s"
                )

                generated_list.append(video)
                torch.cuda.empty_cache()

        # offload dit model
        if self.vram_management:
            self.offload_dit_model()
        if (
            postprocess_seam_repair_frames > 0
            or postprocess_temporal_smoothing_frames > 0
            or postprocess_boundary_alignment
        ):
            logger.info(
                "Post-process seam repair stage enabled: seam_frames={}, smoothing_frames={}, boundary_alignment={}",
                postprocess_seam_repair_frames,
                postprocess_temporal_smoothing_frames,
                postprocess_boundary_alignment,
            )
            generated_list = self._repair_chunk_seams(
                generated_list,
                seam_frames=postprocess_seam_repair_frames,
                smoothing_frames=postprocess_temporal_smoothing_frames,
                boundary_alignment=postprocess_boundary_alignment,
            )
        video_array = torch.cat(generated_list, dim=0).numpy().astype(np.uint8)
        video = Video(data=video_array, prompt=input_prompt, fps=tgt_fps)
        generate_end_time = time.perf_counter()
        logger.info(
            f"FlashTalk Pipeline Generate time: {generate_end_time - generate_start_time:.2f}s"
        )
        return video
