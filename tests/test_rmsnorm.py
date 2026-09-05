"""Correctness of kernels.rmsnorm against the HF LlamaRMSNorm formula."""

import pytest
import torch

from kernels.rmsnorm import rmsnorm


def hf_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps)).to(x.dtype) * weight


TOL = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 5e-2}

SHAPES = [(1, 8), (2, 128), (4, 512), (8, 4096), (2, 5, 4096), (3, 1100), (1, 8193), (1, 14336)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton needs a CUDA GPU")


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_matches_hf(shape, dtype):
    torch.manual_seed(0)
    n = shape[-1]
    x = (torch.randn(shape, device="cuda", dtype=torch.float32) * 3.0).to(dtype)
    w = torch.randn(n, device="cuda", dtype=torch.float32).to(dtype)

    want = hf_rmsnorm(x, w)
    got = rmsnorm(x, w)

    atol = TOL[dtype]
    max_diff = (got.float() - want.float()).abs().max().item()
    assert torch.allclose(got.float(), want.float(), atol=atol, rtol=atol), (
        f"shape={shape} dtype={dtype} max_abs_diff={max_diff:.3e}"
    )


@requires_cuda
def test_rmsnorm_input_unchanged():
    x = torch.randn(4, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    before = x.clone()
    rmsnorm(x, w)
    assert torch.equal(x, before)
