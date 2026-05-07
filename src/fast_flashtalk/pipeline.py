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
        keep_dit_on_gpu=True,
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
        self.cpu_offload = False
        self.keep_dit_on_gpu = keep_dit_on_gpu
        self.quantize_weights = quantize_weights
        self.weight_bits = weight_bits

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device="cpu" if self.cpu_offload else self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            offload_to_cpu=self.cpu_offload,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

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
        ).to("cpu" if self.cpu_offload else self.device)
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

        cond_frame = videos[:, :, -self.motion_frames_num :].to(self.device)
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
        logger.info(f"encode motion frames: {end_encode_time - start_encode_time:.2f}s")

        if self.cpu_offload:
            self.vae.model.cpu()
            torch.cuda.empty_cache()

        gen_video_samples = videos  # [:, :, self.motion_frames_num:]
        gen_video_samples = gen_video_samples[0].to(torch.float32)
        gen_video_samples = (
            ((gen_video_samples + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255
        ).contiguous()
        return gen_video_samples

    def generate(
        self,
        input_prompt: str,
        audio: Audio,
        image: Image,
        audio_encode_mode: Literal["stream", "once"] = "once",
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
        cached_audio_duration = self.infer_params["cached_audio_duration"]
        frame_num = self.infer_params["frame_num"]
        motion_frames_num = self.infer_params["motion_frames_num"]
        slice_len = frame_num - motion_frames_num
        self.prepare_params(input_prompt=input_prompt, cond_image=image)

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
                torch.cuda.synchronize()
                start_time = time.time()

                # inference
                video = self.generate_chunk(audio_embedding_chunk)

                if chunk_idx != 0:
                    video = video[motion_frames_num:]

                torch.cuda.synchronize()
                end_time = time.time()
                logger.info(
                    f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s"
                )

                generated_list.append(video.cpu())
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

            for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
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
                video = video[motion_frames_num:]

                torch.cuda.synchronize()
                end_time = time.time()
                logger.info(
                    f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s"
                )

                generated_list.append(video.cpu())
                torch.cuda.empty_cache()

        # offload dit model
        if self.vram_management:
            self.offload_dit_model()
        video_array = torch.cat(generated_list, dim=0).numpy().astype(np.uint8)
        video = Video(data=video_array, prompt=input_prompt, fps=tgt_fps)
        generate_end_time = time.perf_counter()
        logger.info(
            f"FlashTalk Pipeline Generate time: {generate_end_time - generate_start_time:.2f}s"
        )
        return video
