import time
from typing import List

import torch
from loguru import logger

from flash_talk.gemlite.helper import (
        patch_model,
        A8W8_int8_dynamic,
    )


def quantize_model_a8w8_int8_gemlite(
    model: torch.nn.Module,
    device: str = "cuda",
    exclude: List[str] = [
        "time_embedding.0",
        "time_embedding.2",
        "time_projection.1",
        "head.head",
        "img_emb.proj.1",
        "img_emb.proj.3",
    ],
) -> None:

    logger.info(f"Quantizing model on {device}...")
    quant_start_time = time.perf_counter()
    patch_model(
        model, processor=A8W8_int8_dynamic(), device=device, skip_modules=exclude
    )
    quant_end_time = time.perf_counter()
    logger.info(f"Quantization time: {quant_end_time - quant_start_time}")
