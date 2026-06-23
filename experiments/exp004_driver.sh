#!/usr/bin/env bash
# EXP-004: layered skill-injection ablation. 4 arms × 5 reps, DeepSeek agent loop.
#   base       = no injection            (model's bare ability)
#   facts      = wiki/ptx/facts/*        (verified exact recipes)
#   heuristics = facts + heuristics/*    (+ regime->technique judgment)
#   full       = + menu/*                (+ breadth of what exists)
# Injected content = the REAL forge wiki cards (via --skill-level), not a paraphrase.
# 2 phases of 2 arms, one GPU each, so no two benches share a GPU (timing accuracy).
set -u
source /nvme/share/gucheng/secrets/keys.env
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd /nvme/share/gucheng/anvil
export PYTHONPATH=/nvme/share/gucheng/anvil
mkdir -p runs_exp004

run_arm() {
  local arm="$1" dev="$2"; shift 2
  for r in 1 2 3 4 5; do
    echo "[driver] $arm rep$r start $(date)" >> runs_exp004/driver.log
    .venv/bin/python -m anvil.cli agent --op gemm_bf16_nt \
      --repo /nvme/share/gucheng/OpenKernels --device "$dev" \
      --author "gucheng_${arm}" --max-attempts 12 --out-dir "runs_exp004/$arm" "$@" \
      > "runs_exp004/${arm}_rep${r}.log" 2>&1
    echo "[driver] $arm rep$r done  $(date)" >> runs_exp004/driver.log
  done
}

# phase 1
run_arm base  6 &
run_arm facts 7 --skill-level facts &
wait
# phase 2
run_arm heuristics 6 --skill-level heuristics &
run_arm full       7 --skill-level full &
wait
echo "done $(date)" > runs_exp004/DONE
