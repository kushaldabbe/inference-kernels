"""Bench for kernels.rmsnorm.

Reports, per shape and dtype: median ms, GB/s, % of RTX 4090 peak
(1008 GB/s). The 4090 column is the target hardware; local numbers on
whatever GPU is attached are printed with their own estimated peak and
are context only, never a claim.

Usage: python -m bench.bench_rmsnorm
"""

import torch

from bench.peak import RTX_4090, local_hardware
from bench.timing import bench_gpu_ms, gb_s
from kernels.rmsnorm import rmsnorm
from tests.test_rmsnorm import hf_rmsnorm

SHAPES = [(1, 4096), (2, 4096), (4, 4096), (8, 4096), (16, 4096), (32, 4096),
          (128, 4096), (1024, 4096), (4096, 4096), (16384, 4096)]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def rmsnorm_bytes(rows: int, n: int, itemsize: int) -> int:
    # read x once, write y once; weight is L2-resident, amortized to ~0
    return 2 * rows * n * itemsize


def main():
    assert torch.cuda.is_available(), "bench needs a CUDA GPU"
    local = local_hardware()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if local:
        print(f"Local estimated peak: {local.bandwidth_GBps:.0f} GB/s (ESTIMATE, not a claim)")
    print(f"Target peak: {RTX_4090.bandwidth_GBps:.0f} GB/s ({RTX_4090.name})")

    for dtype in DTYPES:
        itemsize = torch.empty((), dtype=dtype).element_size()
        print(f"\n=== {dtype} ===")
        print(f"{'rows':>6} {'N':>6} {'ms':>9} {'GB/s':>9} {'%4090':>7} {'%local':>7} {'eager GB/s':>10}")
        for rows, n in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(rows, n, device="cuda", dtype=torch.float32).to(dtype)
            w = torch.randn(n, device="cuda", dtype=torch.float32).to(dtype)

            ms = bench_gpu_ms(lambda: rmsnorm(x, w))
            eager_ms = bench_gpu_ms(lambda: hf_rmsnorm(x, w))
            bytes_moved = rmsnorm_bytes(rows, n, itemsize)
            bw = gb_s(bytes_moved, ms)
            row = (f"{rows:>6} {n:>6} {ms:>9.4f} {bw:>9.1f} "
                   f"{RTX_4090.bandwidth_pct(bw):>6.1f}% "
                   f"{(100.0 * bw / local.bandwidth_GBps):>6.1f}% "
                   f"{gb_s(bytes_moved, eager_ms):>10.1f}")
            print(row)


if __name__ == "__main__":
    main()
