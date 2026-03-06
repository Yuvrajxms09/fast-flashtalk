# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import yaml
import torch
from loguru import logger

from flash_talk.src.distributed.usp_device import get_device, get_parallel_degree
from flash_talk.configs import multitalk_14B
from flash_talk.utils import loudness_norm
from flash_talk.pipeline import FlashTalkPipeline

with open("flash_talk/configs/infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)

# TODO: support more resolution
target_size = (infer_params["height"], infer_params["width"])


def get_pipeline(ckpt_dir, wav2vec_dir, cpu_offload=True, device: str = "cuda:0"):
    cfg = multitalk_14B
    pipeline = FlashTalkPipeline(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        wav2vec_dir=wav2vec_dir,
        device=device,
        cpu_offload=cpu_offload,
    )

    return pipeline


def get_base_data(pipeline, input_prompt, cond_image, base_seed):
    pipeline.prepare_params(
        input_prompt=input_prompt,
        cond_image=cond_image,
        target_size=target_size,
        frame_num=infer_params["frame_num"],
        motion_frames_num=infer_params["motion_frames_num"],
        sampling_steps=infer_params["sample_steps"],
        seed=base_seed,
        shift=infer_params["sample_shift"],
        color_correction_strength=infer_params["color_correction_strength"],
    )


def get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
    audio_array = loudness_norm(audio_array, infer_params["sample_rate"])
    audio_embedding = pipeline.preprocess_audio(
        audio_array, sr=infer_params["sample_rate"], fps=infer_params["tgt_fps"]
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


def run_pipeline(pipeline, audio_embedding):
    audio_embedding = audio_embedding.to(pipeline.device)
    sample = pipeline.generate(audio_embedding)
    sample_frames = (
        ((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255
    ).contiguous()
    return sample_frames
