import torch
import triton
import triton.language as tl


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected a 3D tensor, got shape={tuple(x.shape)}")
    return x


@triton.jit
def _fused_affine_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    stride_xb,
    stride_xs,
    stride_xc,
    stride_sb,
    stride_ss,
    stride_sc,
    stride_tb,
    stride_ts,
    stride_tc,
    stride_ob,
    stride_os,
    stride_oc,
    n_cols,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < n_cols

    x = tl.load(
        x_ptr + pid_b * stride_xb + pid_s * stride_xs + offs_c * stride_xc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + pid_b * stride_sb + pid_s * stride_ss + offs_c * stride_sc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + pid_b * stride_tb + pid_s * stride_ts + offs_c * stride_tc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out = x * scale + shift
    tl.store(
        out_ptr + pid_b * stride_ob + pid_s * stride_os + offs_c * stride_oc,
        out,
        mask=mask,
    )


@triton.jit
def _fused_residual_add_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    stride_xb,
    stride_xs,
    stride_xc,
    stride_rb,
    stride_rs,
    stride_rc,
    stride_ob,
    stride_os,
    stride_oc,
    n_cols,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < n_cols

    x = tl.load(
        x_ptr + pid_b * stride_xb + pid_s * stride_xs + offs_c * stride_xc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + pid_b * stride_rb + pid_s * stride_rs + offs_c * stride_rc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + pid_b * stride_ob + pid_s * stride_os + offs_c * stride_oc,
        x + residual,
        mask=mask,
    )


@triton.jit
def _fused_residual_mul_add_kernel(
    x_ptr,
    residual_ptr,
    gate_ptr,
    out_ptr,
    stride_xb,
    stride_xs,
    stride_xc,
    stride_rb,
    stride_rs,
    stride_rc,
    stride_gb,
    stride_gs,
    stride_gc,
    stride_ob,
    stride_os,
    stride_oc,
    n_cols,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < n_cols

    x = tl.load(
        x_ptr + pid_b * stride_xb + pid_s * stride_xs + offs_c * stride_xc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + pid_b * stride_rb + pid_s * stride_rs + offs_c * stride_rc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + pid_b * stride_gb + pid_s * stride_gs + offs_c * stride_gc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + pid_b * stride_ob + pid_s * stride_os + offs_c * stride_oc,
        x + residual * gate,
        mask=mask,
    )


def fused_affine(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """
    Compute x * scale + shift with a single Triton launch.

    All tensors must be broadcastable to the same [B, S, C] shape.
    """
    if not x.is_cuda or x.dim() != 3:
        return x * scale + shift

    x = _ensure_3d(x)
    if scale.shape != x.shape:
        scale = scale.expand_as(x)
    if shift.shape != x.shape:
        shift = shift.expand_as(x)

    out = torch.empty_like(x)
    block_c = triton.next_power_of_2(min(x.shape[-1], 1024))
    grid = (x.shape[0], x.shape[1], triton.cdiv(x.shape[-1], block_c))

    _fused_affine_kernel[grid](
        x,
        scale,
        shift,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        shift.stride(0),
        shift.stride(1),
        shift.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        x.shape[-1],
        BLOCK_C=block_c,
    )
    return out


def fused_residual_add(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.dim() != 3:
        return x + residual

    x = _ensure_3d(x)
    if residual.shape != x.shape:
        residual = residual.expand_as(x)

    out = torch.empty_like(x)
    block_c = triton.next_power_of_2(min(x.shape[-1], 1024))
    grid = (x.shape[0], x.shape[1], triton.cdiv(x.shape[-1], block_c))

    _fused_residual_add_kernel[grid](
        x,
        residual,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        x.shape[-1],
        BLOCK_C=block_c,
    )
    return out


def fused_residual_mul_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda or x.dim() != 3:
        return x + residual * gate

    x = _ensure_3d(x)
    if residual.shape != x.shape:
        residual = residual.expand_as(x)
    if gate.shape != x.shape:
        gate = gate.expand_as(x)

    out = torch.empty_like(x)
    block_c = triton.next_power_of_2(min(x.shape[-1], 1024))
    grid = (x.shape[0], x.shape[1], triton.cdiv(x.shape[-1], block_c))

    _fused_residual_mul_add_kernel[grid](
        x,
        residual,
        gate,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        x.shape[-1],
        BLOCK_C=block_c,
    )
    return out
