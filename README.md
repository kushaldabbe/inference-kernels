# inference-kernel-pack

Triton LLM inference kernels written from scratch, benchmarked against
production references as % of peak hardware capability. Target hardware:
RTX 4090. Status: in progress.

## Hardware truth

| Machine | GPU | Role |
|---|---|---|
| Local | GTX 1650 Ti (4 GB, sm75, no tensor cores) | correctness tests only |
| Rented 4090 (vast.ai / RunPod) | RTX 4090 (24 GB, 1008 GB/s) | all benchmark numbers |

Every % of peak in this README comes from the 4090. Local numbers are
smoke tests and are labeled as estimates. This rule exists so the project
never publishes a number the hardware cannot back.

## Peak numbers used (sources in bench/peak.py)

- RTX 4090 bandwidth: 1008 GB/s (384-bit GDDR6X at 21 Gbps)
- RTX 4090 FP16 tensor dense, FP32 accumulate: 82.6 TFLOPS
  (FP16-accumulate: 165.2 TFLOPS — stated separately, never mixed)

## Kernels

| Kernel | Status | Correctness | Best result | % of 4090 peak |
|---|---|---|---|---|
| rmsnorm | v1 done | 25/25 vs HF LlamaRMSNorm | 165 GB/s local smoke (1650 Ti, context only) | pending 4090 run |

## Run

```
python -m pytest tests -q          # correctness
python -m bench.bench_rmsnorm      # numbers
```

## Install (Windows, GTX 1650 Ti local box)

```
python -m venv .venv
.venv\Scripts\pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install triton-windows pytest numpy
```

For the 4090 box (Linux): `pip install torch` (CUDA build), `pip install triton pytest numpy`.
