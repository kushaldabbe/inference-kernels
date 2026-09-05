"""GPU timing helpers.

Rules (from the spec, non-negotiable):
- CUDA events, not wall clock.
- Warmup before timing.
- torch.cuda.synchronize between iterations.
- Median over 100+ runs.
"""

import statistics

import torch


def bench_gpu_ms(fn, *, warmup: int = 25, iters: int = 100) -> float:
    """Return the median GPU time of fn() in milliseconds."""
    if not torch.cuda.is_available():
        raise RuntimeError("bench_gpu_ms requires a CUDA GPU")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def gb_s(bytes_moved: int, ms: float) -> float:
    """Convert bytes moved in `ms` milliseconds to GB/s."""
    return bytes_moved / (ms * 1e-3) / 1e9
