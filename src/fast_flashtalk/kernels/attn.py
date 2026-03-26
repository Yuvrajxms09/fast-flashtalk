import torch
from sageattention import sageattn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
):
    """Attention function
    Attention(o) = softmax(Q @K^T) @ V
    q: [B, L, N, D]
    k: [B, L, N, D]
    v: [B, L, N, D]
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
        # return flash_attn_func(q, k, v)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return scaled_dot_product_attention(
                q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            ).permute(0, 2, 1, 3)
    else:
        return sageattn(q=q, k=k, v=v, tensor_layout="NHD", output_dtype=dtype)
