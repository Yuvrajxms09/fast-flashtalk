#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as metadata
import os
import resource
import subprocess
import sys
from pathlib import Path

import psutil
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fast_flashtalk import Audio, FlashTalkPipeline, Image


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value == "1"


def current_ram_gb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / (1024**3)


def peak_ram_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def main() -> None:
    print("python:", subprocess.check_output(["python3", "--version"], text=True).strip())
    print("torch:", torch.__version__)
    print("torchvision (dist):", metadata.version("torchvision"))
    print("torchaudio (dist):", metadata.version("torchaudio"))
    print("cuda available:", torch.cuda.is_available())
    cuda_device_index = env_int("CUDA_DEVICE_INDEX", 0)
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print(f"cuda device {cuda_device_index}:", torch.cuda.get_device_name(cuda_device_index))
        torch.cuda.set_device(cuda_device_index)

    required = ("SoulX-FlashTalk-14B", "chinese-wav2vec2-base")
    candidates = ["/content/flash-talk-models", os.environ.get("VOLUME_ROOT")]
    volume_root = None
    for root in candidates:
        if root and os.path.isdir(root) and all(
            os.path.isdir(os.path.join(root, d)) for d in required
        ):
            volume_root = root
            break
    if volume_root is None:
        raise FileNotFoundError(
            "Set VOLUME_ROOT or place SoulX-FlashTalk-14B and chinese-wav2vec2-base under /content/flash-talk-models."
        )

    checkpoint_dir = os.path.join(volume_root, "SoulX-FlashTalk-14B")
    wav2vec_dir = os.path.join(volume_root, "chinese-wav2vec2-base")
    text_prompt = os.environ.get("TEXT_PROMPT") or (
        "A person is talking. Only the foreground characters are moving, the background remains static."
    )
    image_path = os.environ.get("IMAGE_PATH", "/content/sample_data/girl.png")
    audio_path = os.environ.get("AUDIO_PATH", "/content/sample_data/cantonese_16k_5sec.wav")
    target_size = (env_int("TARGET_HEIGHT", 768), env_int("TARGET_WIDTH", 448))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "sample_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_persistent_param_in_dit = env_int("NUM_PERSISTENT_PARAM_IN_DIT", 15_000_000_000)
    weight_bits = env_int("WEIGHT_BITS", 8)
    keep_dit_on_gpu = env_bool("KEEP_DIT_ON_GPU", True)
    warmup_runs = env_int("WARMUP_RUNS", 1)
    profile_mode = os.environ.get("PROFILE_MODE", "dit").lower()
    audio_encode_mode = os.environ.get("AUDIO_ENCODE_MODE", "once")

    proc = psutil.Process(os.getpid())
    print(f"Volume root: {volume_root}")
    print(f"Checkpoint: {checkpoint_dir} (OK)")
    print(f"Wav2Vec: {wav2vec_dir} (OK)")
    print(f"Target size: {target_size}")
    print(f"Warmup runs: {warmup_runs}")

    pipeline = FlashTalkPipeline(
        checkpoint_dir=checkpoint_dir,
        wav2vec_dir=wav2vec_dir,
        num_persistent_param_in_dit=num_persistent_param_in_dit,
        keep_dit_on_gpu=keep_dit_on_gpu,
        weight_bits=weight_bits,
        device=f"cuda:{cuda_device_index}",
    )

    image = Image(uri=image_path)
    audio = Audio(uri=audio_path)

    for key in ("tgt_fps", "sample_rate"):
        env_key = key.upper()
        if os.environ.get(env_key):
            pipeline.infer_params[key] = int(os.environ[env_key])

    generate_kwargs = dict(
        input_prompt=text_prompt,
        audio=audio,
        image=image,
        audio_encode_mode=audio_encode_mode,
    )

    if profile_mode == "dit":
        pipeline.prepare_params(
            input_prompt=text_prompt, cond_image=image, target_size=target_size
        )
        sample_rate = pipeline.infer_params["sample_rate"]
        tgt_fps = pipeline.infer_params["tgt_fps"]
        frame_num = pipeline.infer_params["frame_num"]
        motion_frames_num = pipeline.infer_params["motion_frames_num"]
        slice_len = frame_num - motion_frames_num

        human_speech_array_all = audio.load(sample_rate=sample_rate, mono=True).data
        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        remainder = len(human_speech_array_all) - frame_num * sample_rate // tgt_fps
        remainder %= human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.pad(
                human_speech_array_all,
                (0, pad_length),
                mode="constant",
                constant_values=0,
            )

        audio_embedding_all = pipeline.get_audio_embedding(human_speech_array_all)
        audio_embedding = audio_embedding_all[:, :frame_num].contiguous()
        pipeline.arg_c["audio"] = audio_embedding

        latent = torch.randn(
            16,
            (frame_num - 1) // 4 + 1,
            pipeline.lat_h,
            pipeline.lat_w,
            dtype=pipeline.param_dtype,
            device=pipeline.device,
            generator=pipeline.generator,
        )
        latent[:, : pipeline.latent_motion_frames.shape[1]] = pipeline.latent_motion_frames
        t = pipeline.timesteps[0]

        def run_step() -> torch.Tensor:
            return pipeline.model([latent], t=t, **pipeline.arg_c)[0]
    elif profile_mode == "e2e":
        if target_size != (768, 448):
            print(
                "Warning: PROFILE_MODE=e2e uses pipeline.generate() and this checkout does not accept target_size there; "
                "TARGET_HEIGHT/TARGET_WIDTH are ignored in e2e mode."
            )

        def run_step() -> torch.Tensor:
            return pipeline.generate(**generate_kwargs)
    else:
        raise ValueError("PROFILE_MODE must be 'dit' or 'e2e'.")

    for _ in range(warmup_runs):
        _ = run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ram_before_gb = current_ram_gb(proc)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        _ = run_step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ram_after_gb = current_ram_gb(proc)
    peak_ram_gb_value = peak_ram_gb()

    trace_path = output_dir / f"{profile_mode}_profiler_trace.json"
    prof.export_chrome_trace(str(trace_path))

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print(f"Trace exported to {trace_path}")
    print(f"RAM before run: {ram_before_gb:.2f} GB")
    print(f"RAM after run:  {ram_after_gb:.2f} GB")
    print(f"Peak RAM RSS:   {peak_ram_gb_value:.2f} GB")
    print(f"Profile mode:   {profile_mode}")


if __name__ == "__main__":
    main()
