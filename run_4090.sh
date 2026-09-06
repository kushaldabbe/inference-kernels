#!/usr/bin/env bash
# One-command run for the rented RTX 4090 box.
#
# Target environment (pattern proven in inference-bench/run_all.sh):
#   RunPod "runpod/pytorch" template: torch 2.8.0+cu128 preinstalled,
#   driver 570-580, CUDA 13.0, Python 3.12.
#
# Stages: preflight -> d2d sanity -> pytest gate -> bench.
# Every run leaves a self-contained forensics folder in results/<ts>/:
#   console.log     everything (stdout+stderr)
#   env.json        gpu, versions, git sha + dirty state, copy bandwidth
#   nvidia_smi.txt  full nvidia-smi
#   clocks.txt      clock/power/temperature snapshot (throttle evidence)
#   git_state.txt   exact code state (sha, dirty files, diffstat)
#   rmsnorm.json    machine-readable bench rows
# Any failure prints: FAILED: stage=<name> exit=<code> — the console.log
# tail then shows the last lines of that stage.
#
# Usage:
#   bash run_4090.sh
#
# Env overrides:
#   IKP_ALLOW_NON_4090=1   continue on 4090D / other GPU (results marked, not a 4090 claim)
#   IKP_WARN_COPY_GBPS=850 warn below this copy bandwidth
#   IKP_MIN_COPY_GBPS=700  abort below this copy bandwidth (throttled/shared box)
set -euo pipefail
cd "$(dirname "$0")"

OUT="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
exec > >(tee "$OUT/console.log") 2>&1

STAGE="0_setup"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo; echo "FAILED: stage=$STAGE exit=$rc at $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "see $OUT/console.log tail for the last lines of this stage"; fi' EXIT

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "=== run started $(ts) ==="
echo "artifacts: $OUT/"

STAGE="1_preflight"
echo "=== [$(ts)] 1/4 preflight ==="
GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1)
GPU_NAME="${GPU_LINE%%,*}"
echo "gpu line: $GPU_LINE"
nvidia-smi | tee "$OUT/nvidia_smi.txt" >/dev/null
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE > "$OUT/clocks.txt" 2>&1 || true

{
  echo "sha: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "dirty files (git status --porcelain):"
  git status --porcelain 2>/dev/null || true
  echo "diffstat:"
  git diff --stat 2>/dev/null || true
} > "$OUT/git_state.txt"
echo "git state: $(head -n1 "$OUT/git_state.txt")"

case "$GPU_NAME" in
  *"RTX 4090"*) echo "GPU check: OK (RTX 4090)" ;;
 *"4090D"*|*"4090 D"*)
    echo "GPU check: WARN — 4090D variant (lower clocks), not a clean 4090 claim"
    if [ "${IKP_ALLOW_NON_4090:-0}" != "1" ]; then
      echo "abort: set IKP_ALLOW_NON_4090=1 to continue anyway"; exit 1
    fi ;;
  *)
    echo "GPU check: FAIL — expected RTX 4090, got: $GPU_NAME"
    if [ "${IKP_ALLOW_NON_4090:-0}" != "1" ]; then
      echo "abort: set IKP_ALLOW_NON_4090=1 to continue anyway"; exit 1
    fi ;;
esac

VENV=.venv4090
if [ ! -f "$VENV/.ready" ]; then
  echo "provisioning venv (reuses template torch via --system-site-packages)"
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q pytest numpy
  touch "$VENV/.ready"
else
  echo "venv already provisioned"
fi
PY="$VENV/bin/python"

STAGE="2_d2d_sanity"
echo "=== [$(ts)] 2/4 d2d copy sanity ==="
IKP_OUT="$OUT" $PY - <<'EOF'
import json
import os

import torch

assert torch.cuda.is_available(), "CUDA not visible — aborting"
import triton

p = torch.cuda.get_device_properties(0)
name = torch.cuda.get_device_name(0)

n = 512 * 1024 * 1024  # fp16 elements -> 1 GiB per buffer
a = torch.empty(n, dtype=torch.float16, device="cuda")
b = torch.empty_like(a)
for _ in range(3):
    b.copy_(a)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(10):
    b.copy_(a)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 10
gbps = (2 * n * 2) / (ms * 1e-3) / 1e9

warn = float(os.environ.get("IKP_WARN_COPY_GBPS", "850"))
mini = float(os.environ.get("IKP_MIN_COPY_GBPS", "700"))
print(f"d2d copy: {gbps:.0f} GB/s  (RTX 4090 theoretical: 1008 GB/s)")
if gbps < mini:
    print(f"abort: copy bandwidth {gbps:.0f} < {mini:.0f} GB/s — box is throttled/shared")
    raise SystemExit(1)
if gbps < warn:
    print(f"warn: copy bandwidth {gbps:.0f} < {warn:.0f} GB/s — check clocks.txt for throttling")

env = {
    "gpu_name": name,
    "sm": f"{p.major}.{p.minor}",
    "vram_GB": round(p.total_memory / 1e9, 1),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "triton": triton.__version__,
    "d2d_copy_GBps": round(gbps, 1),
}
out = os.environ.get("IKP_OUT")
if out:
    with open(os.path.join(out, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
print(json.dumps(env, indent=2))
EOF

STAGE="3_pytest"
echo "=== [$(ts)] 3/4 pytest (correctness gate) ==="
$PY -m pytest tests -q

STAGE="4_bench"
echo "=== [$(ts)] 4/4 bench ==="
$PY -m bench.bench_rmsnorm --out "$OUT/rmsnorm.json"

STAGE="done"
echo
echo "=== DONE $(ts) ==="
echo "artifacts in $OUT/:"
echo "  console.log    full output of all stages"
echo "  env.json       gpu, versions, copy bandwidth"
echo "  nvidia_smi.txt full nvidia-smi"
echo "  clocks.txt     clock/power/temperature (throttle evidence)"
echo "  git_state.txt  sha + dirty files + diffstat"
echo "  rmsnorm.json   machine-readable bench rows"
