"""Hardware peak numbers.

Every % of peak number printed by this repo must come from a constant
defined here, with the source written next to it. No magic numbers in
bench scripts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    bandwidth_GBps: float
    fp16_tensor_tflops_fp32acc: float
    fp16_tensor_tflops_fp16acc: float
    fp32_tflops: float

    def bandwidth_pct(self, gb_s: float) -> float:
        return 100.0 * gb_s / self.bandwidth_GBps

    def tflops_pct(self, tflops: float, acc: str = "fp32") -> float:
        denom = self.fp16_tensor_tflops_fp32acc if acc == "fp32" else self.fp16_tensor_tflops_fp16acc
        return 100.0 * tflops / denom


# RTX 4090 (AD102, 24 GB GDDR6X, 384-bit, 21 Gbps).
#   Bandwidth: 384 * 21 / 8 = 1008 GB/s (NVIDIA Ada whitepaper).
#   FP16 tensor, dense, FP32 accumulate: 82.6 TFLOPS (Ada whitepaper).
#   FP16 tensor, dense, FP16 accumulate: 165.2 TFLOPS.
#   FP32 (CUDA cores): 82.6 TFLOPS.
RTX_4090 = Hardware(
    name="RTX 4090",
    bandwidth_GBps=1008.0,
    fp16_tensor_tflops_fp32acc=82.6,
    fp16_tensor_tflops_fp16acc=165.2,
    fp32_tflops=82.6,
)


def local_hardware() -> Hardware | None:
    """Best-effort peak numbers for whatever GPU is attached.

    The bandwidth is derived from torch's device properties and is an
    ESTIMATE (clock reporting differs across cards/drivers). It is good
    enough for smoke-test context and must never be quoted as a
    benchmark denominator. Real numbers come from a verified card like
    RTX_4090.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    est_bw = p.memory_clock_rate * 1e3 * p.memory_bus_width / 8 / 1e9
    return Hardware(
        name=f"{p.name} (local, estimated)",
        bandwidth_GBps=est_bw,
        fp16_tensor_tflops_fp32acc=float("nan"),
        fp16_tensor_tflops_fp16acc=float("nan"),
        fp32_tflops=float("nan"),
    )
