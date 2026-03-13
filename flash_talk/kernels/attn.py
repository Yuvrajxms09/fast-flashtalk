from typing import Tuple
import os

import torch
from flash_attn import flash_attn_func
from sageattention import sageattn_varlen as varlen_func
from sageattention import sageattn
from loguru import logger

USE_SPAS = os.environ.get("USE_SPAS", "false").lower() == "true"
if USE_SPAS:
    logger.info("Using Sparse SageAttention")
else:
    logger.info("Using Sageattn")


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    x = varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
):
    """Attention function
    Attention(o) = softmax(Q @K^T) @ V
    q: [B, L, N, D]
    k: [B, L, N, D]
    v: [B, L, N, D]
    topk: float. Topk value for spas_sage2_attn_meansim_topk_cuda.
    dtype: torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    # print(f"attention: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256
    _B, L_k, _N_k, _D = k.shape

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Convert to half precision if needed
    q = half(q)
    k = half(k)
    v = half(v)

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if L_k < 512:
        return flash_attn_func(q, k, v)
    else:
        if USE_SPAS:
            from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

            return spas_sage2_attn_meansim_topk_cuda(
                q=q, k=k, v=v, tensor_layout="NHD", topk=topk, output_dtype=dtype
            )
        else:
            return sageattn(q=q, k=k, v=v, tensor_layout="NHD", output_dtype=dtype)
