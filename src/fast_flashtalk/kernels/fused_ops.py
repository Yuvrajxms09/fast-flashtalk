import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected a 3D tensor, got shape={tuple(x.shape)}")
    return x


@triton.jit
def _fused_rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_b,
    stride_s,
    stride_c,
    stride_w,
    n_cols,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < n_cols

    x = tl.load(
        x_ptr + pid_b * stride_b + pid_s * stride_s + offs_c * stride_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    weight = tl.load(weight_ptr + offs_c * stride_w, mask=mask, other=0.0).to(tl.float32)
    out = x * inv_rms * weight
    tl.store(
        out_ptr + pid_b * stride_b + pid_s * stride_s + offs_c * stride_c,
        out,
        mask=mask,
    )


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


@triton.jit
def _fused_ffn_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    out_ptr,
    stride_xm,
    stride_xk,
    stride_w1o,
    stride_w1i,
    stride_b1,
    stride_w2o,
    stride_w2i,
    stride_b2,
    stride_om,
    stride_on,
    n_rows,
    in_dim,
    hidden_dim,
    out_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IN_DIM: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < n_rows
    n_mask = offs_n < OUT_DIM

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    bias2 = tl.load(b2_ptr + offs_n * stride_b2, mask=n_mask, other=0.0).to(
        tl.float32
    )

    num_h_tiles: tl.constexpr = tl.cdiv(HIDDEN_DIM, BLOCK_H)
    num_k_tiles: tl.constexpr = tl.cdiv(IN_DIM, BLOCK_K)

    for h_idx in tl.range(num_h_tiles, num_stages=1):
        h0 = h_idx * BLOCK_H
        offs_h = h0 + tl.arange(0, BLOCK_H)
        h_mask = offs_h < HIDDEN_DIM
        hidden = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

        for k_idx in tl.range(num_k_tiles, num_stages=1):
            k0 = k_idx * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < IN_DIM

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w1 = tl.load(
                w1_ptr + offs_h[None, :] * stride_w1o + offs_k[:, None] * stride_w1i,
                mask=h_mask[None, :] & k_mask[:, None],
                other=0.0,
            )
            hidden = tl.dot(x, w1, acc=hidden, out_dtype=tl.float32)

        bias1 = tl.load(b1_ptr + offs_h * stride_b1, mask=h_mask, other=0.0).to(
            tl.float32
        )
        hidden += bias1[None, :]

        hidden = hidden.to(tl.bfloat16)
        hidden = 0.5 * hidden * (1.0 + tl.sigmoid(1.702 * hidden))

        w2 = tl.load(
            w2_ptr + offs_n[None, :] * stride_w2o + offs_h[:, None] * stride_w2i,
            mask=n_mask[None, :] & h_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        acc = tl.dot(hidden, w2, acc=acc, out_dtype=tl.float32)

    acc += bias2[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
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


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda or x.dim() != 3:
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * weight

    x = _ensure_3d(x)
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match x last dim {x.shape[-1]}"
        )

    n_cols = x.shape[-1]
    block_c = triton.next_power_of_2(n_cols)
    if block_c > 4096:
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * weight

    out = torch.empty_like(x)
    grid = (x.shape[0], x.shape[1])

    _fused_rms_norm_kernel[grid](
        x,
        weight,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        n_cols,
        eps,
        BLOCK_C=block_c,
    )
    return out


def fused_ffn(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda or x.dim() != 3:
        return F.gelu(x.matmul(w1.t()) + b1, approximate="tanh").matmul(w2.t()) + b2

    x = _ensure_3d(x)
    if w1.dim() != 2 or w2.dim() != 2:
        raise ValueError("expected 2D weights for fused_ffn")
    if b1.dim() != 1 or b2.dim() != 1:
        raise ValueError("expected 1D biases for fused_ffn")

    in_dim = x.shape[-1]
    hidden_dim = w1.shape[0]
    out_dim = w2.shape[0]
    if w1.shape[1] != in_dim:
        raise ValueError(
            f"w1 shape {tuple(w1.shape)} does not match x last dim {in_dim}"
        )
    if w2.shape[1] != hidden_dim:
        raise ValueError(
            f"w2 shape {tuple(w2.shape)} does not match hidden dim {hidden_dim}"
        )
    if b1.shape[0] != hidden_dim or b2.shape[0] != out_dim:
        raise ValueError(
            f"bias shapes {tuple(b1.shape)}, {tuple(b2.shape)} do not match weights"
        )
    x2d = x.reshape(-1, in_dim)
    n_rows = x2d.shape[0]
    out = torch.empty((n_rows, out_dim), device=x.device, dtype=x.dtype)

    block_m = 16
    block_n = 64
    block_k = 32
    block_h = 64
    if hidden_dim > 8192 or in_dim > 4096:
        return F.gelu(x.matmul(w1.t()) + b1, approximate="tanh").matmul(w2.t()) + b2

    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(out_dim, block_n))
    _fused_ffn_kernel[grid](
        x2d,
        w1,
        b1,
        w2,
        b2,
        out,
        x2d.stride(0),
        x2d.stride(1),
        w1.stride(0),
        w1.stride(1),
        b1.stride(0),
        w2.stride(0),
        w2.stride(1),
        b2.stride(0),
        out.stride(0),
        out.stride(1),
        n_rows,
        in_dim,
        hidden_dim,
        out_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        IN_DIM=in_dim,
        HIDDEN_DIM=hidden_dim,
        OUT_DIM=out_dim,
        num_warps=4,
    )
    return out.reshape(*x.shape[:-1], out_dim)
