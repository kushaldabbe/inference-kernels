"""Bench for kernels.rmsnorm.

Reports, per shape and dtype: median ms, GB/s, % of RTX 4090 peak
(1008 GB/s). The 4090 column is the target hardware; local numbers on
whatever GPU is attached are printed with their own estimated peak and
are context only, never a claim.

Usage:
    python -m bench.bench_rmsnorm
    python -m bench.bench_rmsnorm --out results/run/rmsnorm.json

The --out file contains one JSON document: run metadata (gpu, versions,
git sha, peak constants) plus one row per (dtype, shape) with ms, GB/s
and % of peak, so every published number is machine-checkable.
"""

import argparse
import json
import subprocess

import torch
import triton

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


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        st = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return bool(st.strip())
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="write JSON results to this path")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "bench needs a CUDA GPU"
    local = local_hardware()
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if local:
        print(f"Local estimated peak: {local.bandwidth_GBps:.0f} GB/s (ESTIMATE, not a claim)")
    print(f"Target peak: {RTX_4090.bandwidth_GBps:.0f} GB/s ({RTX_4090.name})")

    meta = {
        "kernel": "rmsnorm",
        "version": "v1",
        "gpu_name": torch.cuda.get_device_name(0),
        "sm": f"{p.major}.{p.minor}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "peak": {
            "bandwidth_GBps": RTX_4090.bandwidth_GBps,
            "fp16_tensor_tflops_fp32acc": RTX_4090.fp16_tensor_tflops_fp32acc,
        },
        "local_est_peak_GBps": None if local is None else round(local.bandwidth_GBps, 1),
        "warmup_iters": 25,
        "timed_iters": 100,
        "timing": "cuda events, median",
    }

    results = []
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
            eager_bw = gb_s(bytes_moved, eager_ms)
            pct_local = None if local is None else round(100.0 * bw / local.bandwidth_GBps, 1)
            pct_local_s = f"{pct_local:>6.1f}%" if pct_local is not None else "     -"
            print(f"{rows:>6} {n:>6} {ms:>9.4f} {bw:>9.1f} "
                  f"{RTX_4090.bandwidth_pct(bw):>6.1f}% "
                  f"{pct_local_s:>7} "
                  f"{eager_bw:>10.1f}")
            results.append({
                "dtype": str(dtype).replace("torch.", ""),
                "rows": rows,
                "n": n,
                "ms": round(ms, 4),
                "gb_s": round(bw, 1),
                "pct_of_4090_peak": round(RTX_4090.bandwidth_pct(bw), 1),
                "pct_of_local_est": pct_local,
                "eager_gb_s": round(eager_bw, 1),
                "speedup_vs_eager": round(bw / eager_bw, 2),
            })

    if args.out:
        doc = {"meta": meta, "results": results}
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"\nJSON results written: {args.out}")


if __name__ == "__main__":
    main()
