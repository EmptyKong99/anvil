#!/usr/bin/env bash
# EXP-005 — GENERALIZATION test: do gemm-derived wiki cards help on a DIFFERENT op?
# op = flash_attention_bf16_fwd_bhsd (different algorithm, SAME tensor-core primitives).
# 3 arms × 5 reps, DeepSeek agent, max-attempts 12 (same instrument as EXP-004):
#   base       = no injection
#   facts      = gemm wiki/ptx/facts/*  (mma/ldmatrix/cp.async/swizzle/stmatrix —
#                reusable for FA's QK^T and PV matmuls). THE generalization signal.
#   heuristics = + gemm heuristics (v-ladder) — control; gemm-specific judgment
#                is expected NOT to transfer (maybe even mislead).
# Correctness for FA is judged by fp32-math (anvil okbench_runner override), NOT the
# cuDNN-bitmatch gate (which is < 1 bf16 ULP and rejects correct kernels). Speed is
# still vs cuDNN. Resilience: wait for DeepSeek before each rep + retry-on-empty (the
# server's DeepSeek link drops intermittently). Agent code unchanged.
set -u
source /nvme/share/gucheng/secrets/keys.env
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd /nvme/share/gucheng/anvil
export PYTHONPATH=/nvme/share/gucheng/anvil
mkdir -p runs_exp005
LOG=runs_exp005/driver.log
OP=flash_attention_bf16_fwd_bhsd

wait_ds() {
  for i in $(seq 1 120); do
    local code
    code=$(curl -s -m 12 -o /dev/null -w "%{http_code}" https://api.deepseek.com/v1/models \
           -H "Authorization: Bearer $DEEPSEEK_API_KEY")
    [ "$code" = "200" ] && return 0
    echo "[wait] $(date) deepseek http=$code" >> "$LOG"; sleep 60
  done
  return 1
}

run_arm() {
  local arm="$1" dev="$2"; shift 2
  for r in 1 2 3 4 5; do
    for try in 1 2 3; do
      wait_ds || { echo "[$arm rep$r] deepseek never returned" >> "$LOG"; break; }
      echo "[driver] $arm rep$r try$try start $(date)" >> "$LOG"
      .venv/bin/python -m anvil.cli agent --op "$OP" \
        --repo /nvme/share/gucheng/OpenKernels --device "$dev" \
        --author "gucheng_${arm}" --max-attempts 12 --out-dir "runs_exp005/$arm" "$@" \
        > "runs_exp005/${arm}_rep${r}_try${try}.log" 2>&1
      local last
      last=$(ls -dt runs_exp005/"$arm"/*/ 2>/dev/null | head -1)
      if [ -n "$last" ] && [ -s "$last/results.jsonl" ]; then
        echo "[driver] $arm rep$r ok ($last) $(date)" >> "$LOG"; break
      fi
      echo "[driver] $arm rep$r try$try EMPTY -> retry $(date)" >> "$LOG"
      [ -n "$last" ] && rm -rf "$last"
    done
  done
}

echo "[exp005] start $(date)" >> "$LOG"
# phase 1: base + facts (one GPU each, clean timing)
run_arm base  6 &
run_arm facts 7 --skill-level facts &
wait
# phase 2: heuristics control
run_arm heuristics 6 --skill-level heuristics &
wait
echo "exp005 done $(date)" > runs_exp005/DONE
echo "[exp005] done $(date)" >> "$LOG"
