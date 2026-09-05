"""Fused RMSNorm in Triton.

Reference semantics (HuggingFace LlamaRMSNorm, upcast-to-fp32 variant):
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    y = (x32 * torch.rsqrt(var + eps)).to(x.dtype) * weight

The cast back to the input dtype happens BEFORE the weight multiply,
exactly like HF. Matching the op order keeps max-abs-diff near zero and
makes correctness unambiguous.

Memory model: for a (rows, N) tensor we read rows*N elements and write
rows*N elements. The weight (N elements) is cached in L2 across rows and
is not counted in the bench byte total (dominant error < 1% for N=4096,
rows >= 16; slightly optimistic for batch-1 decode, noted in bench).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride, N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    var = tl.sum(x32 * x32, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    y32 = x32 * rstd
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = y32.to(y_ptr.dtype.element_ty) * w

    tl.store(y_ptr + row * stride + cols, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm over the last dimension of x. x and weight same dtype.

    Returns a new tensor (does not write x in place).
    """
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.shape[-1]
    rows = x.numel() // n
    out = torch.empty_like(x)

    block = triton.next_power_of_2(n)
    num_warps = max(1, min(8, block // 512))

    _rmsnorm_kernel[(rows,)](
        x, weight, out,
        n, n, eps,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out
