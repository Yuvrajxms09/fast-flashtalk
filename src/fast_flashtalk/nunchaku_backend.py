"""Helpers for loading Nunchaku-exported Wan checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _ensure_nunchaku_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sibling = repo_root.parent / "nunchaku"
    if sibling.exists() and str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))


_ensure_nunchaku_path()

try:
    from nunchaku.models.linear import SVDQW4A4Linear
    from nunchaku.ops.fused import fused_gelu_mlp
except Exception as exc:  # pragma: no cover
    SVDQW4A4Linear = None  # type: ignore[assignment]
    fused_gelu_mlp = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _nunchaku_linear_forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    squeeze_batch = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_batch = True
    elif x.dim() != 3:
        raise ValueError(f"Expected a 2D or 3D tensor, got shape {tuple(x.shape)}")

    batch_size, seq_len, channels = x.shape
    x = x.reshape(batch_size * seq_len, channels)
    if output is None:
        output = torch.empty(batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device)
    quantized_x, ascales, lora_act_out = self.quantize(x)
    output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
    output = output.reshape(batch_size, seq_len, -1)
    if squeeze_batch:
        return output.squeeze(0)
    return output


if SVDQW4A4Linear is not None and fused_gelu_mlp is not None:
    SVDQW4A4Linear.forward = _nunchaku_linear_forward  # type: ignore[assignment]


def has_nunchaku_backend() -> bool:
    return SVDQW4A4Linear is not None and fused_gelu_mlp is not None


def require_nunchaku_backend() -> None:
    if not has_nunchaku_backend():
        raise ImportError(
            "Nunchaku backend is unavailable. Ensure the sibling `nunchaku` repo is present and importable."
        ) from _IMPORT_ERROR


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(name)] = new_module
    else:
        setattr(parent, name, new_module)


def _download_or_read_safetensors(
    source: str | Path,
    filename: str | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    path = Path(source)

    if path.exists():
        if path.is_file():
            return load_file(path, device="cpu")

        bundle_dir = path if subfolder is None else path / subfolder
        if filename is not None:
            file_path = bundle_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Missing Nunchaku export file: {file_path}")
            return load_file(file_path, device="cpu")

        split_files = [bundle_dir / "transformer_blocks.safetensors", bundle_dir / "unquantized_layers.safetensors"]
        if all(file_path.exists() for file_path in split_files):
            state_dict: dict[str, torch.Tensor] = {}
            for file_path in split_files:
                state_dict.update(load_file(file_path, device="cpu"))
            return state_dict

        safetensors_files = sorted(bundle_dir.glob("*.safetensors"))
        if len(safetensors_files) == 1:
            return load_file(safetensors_files[0], device="cpu")
        raise FileNotFoundError(
            f"Could not find a Nunchaku export bundle under {bundle_dir}. "
            "Expected either a single .safetensors file or split transformer_blocks.safetensors / "
            "unquantized_layers.safetensors files."
        )

    if filename is None:
        raise ValueError("Remote Nunchaku export loading requires a filename.")

    download_kwargs = {
        "repo_id": str(source),
        "filename": filename,
        "subfolder": subfolder,
        "repo_type": "model",
        "revision": revision,
        "token": token,
        "local_files_only": local_files_only,
    }
    file_path = hf_hub_download(**download_kwargs)
    return load_file(file_path, device="cpu")


def load_nunchaku_export_state_dict(
    export_root: str | Path,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a Nunchaku export from a local directory, a merged safetensors file, or a Hugging Face repo."""
    require_nunchaku_backend()

    path = Path(export_root)
    if path.exists() and path.is_file():
        return load_file(path, device="cpu")

    if path.exists():
        return _download_or_read_safetensors(path, subfolder=subfolder)

    transformer_blocks = _download_or_read_safetensors(
        export_root,
        filename="transformer_blocks.safetensors",
        subfolder=subfolder,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    unquantized_layers = _download_or_read_safetensors(
        export_root,
        filename="unquantized_layers.safetensors",
        subfolder=subfolder,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    merged: dict[str, torch.Tensor] = {}
    merged.update(unquantized_layers)
    merged.update(transformer_blocks)
    return merged


def _remap_export_key(key: str) -> str:
    if key.endswith(".lora_down"):
        return key[: -len(".lora_down")] + ".proj_down"
    if key.endswith(".lora_up"):
        return key[: -len(".lora_up")] + ".proj_up"
    if key.endswith(".smooth_orig"):
        return key[: -len(".smooth_orig")] + ".smooth_factor_orig"
    if key.endswith(".smooth"):
        return key[: -len(".smooth")] + ".smooth_factor"
    return key


def _infer_precision_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    for key, value in state_dict.items():
        if key.endswith(".wscales") and getattr(value, "dtype", None) == torch.float8_e4m3fn:
            return "nvfp4"
    return "int4"


def _infer_rank_from_state_dict(state_dict: dict[str, torch.Tensor], default_rank: int = 1) -> int:
    for key, value in state_dict.items():
        if key.endswith(".lora_down") and value.ndim >= 2:
            return int(value.shape[1])
        if key.endswith(".proj_down") and value.ndim >= 2:
            return int(value.shape[1])
    return default_rank


def _replace_quantized_linears(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    precision: str,
    rank: int,
    prefix: str = "",
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and f"{full_name}.qweight" in state_dict:
            _replace_module(
                module,
                child_name,
                SVDQW4A4Linear.from_linear(
                    child,
                    rank=rank,
                    precision=precision,
                    torch_dtype=child.weight.dtype,
                ),
            )
        else:
            _replace_quantized_linears(child, state_dict, precision=precision, rank=rank, prefix=full_name)


def load_nunchaku_export_into_module(
    module: nn.Module,
    export_root: str | Path,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    local_files_only: bool = False,
    precision: str = "auto",
    rank: int | None = None,
) -> nn.Module:
    """Replace exported Wan linears with Nunchaku linears and load a DeepCompressor/Nunchaku export."""
    require_nunchaku_backend()

    export_state_dict = load_nunchaku_export_state_dict(
        export_root,
        subfolder=subfolder,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )

    if precision == "auto":
        precision = _infer_precision_from_state_dict(export_state_dict)
    if precision not in ("int4", "nvfp4"):
        raise ValueError(f"Unsupported precision={precision}; expected int4, nvfp4, or auto.")

    rank = _infer_rank_from_state_dict(export_state_dict, default_rank=rank or 1) if rank is None else rank

    _replace_quantized_linears(module, export_state_dict, precision=precision, rank=rank)

    loadable_state_dict: dict[str, torch.Tensor] = {}
    wtscale_values: dict[str, float] = {}
    for key, value in export_state_dict.items():
        if key.endswith(".wtscale"):
            wtscale_values[key[: -len(".wtscale")]] = float(value.item()) if value.numel() == 1 else float(value.reshape(-1)[0].item())
            continue
        if key.endswith(".wcscales"):
            continue
        if key.endswith(".subscale"):
            continue
        if precision == "nvfp4" and key.endswith(".wscales") and value.dtype != torch.float8_e4m3fn:
            value = value.to(torch.float8_e4m3fn)
        loadable_state_dict[_remap_export_key(key)] = value

    missing_keys, unexpected_keys = module.load_state_dict(loadable_state_dict, strict=False)
    missing_keys = [key for key in missing_keys if not key.endswith(".wcscales")]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load Nunchaku export cleanly. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )

    for name, submodule in module.named_modules():
        if isinstance(submodule, SVDQW4A4Linear):
            if name in wtscale_values:
                submodule.wtscale = wtscale_values[name]
            elif submodule.wtscale is None:
                submodule.wtscale = 1.0

    return module
