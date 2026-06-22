#!/usr/bin/env bash
# EXP-006 — OP-SPECIFIC CARD value: does flash_attention's OWN wiki card help the
# FA agent, on top of the generic instruction primitives it already gets?
# op = flash_attention_bf16_fwd_bhsd. 3 arms x 5 reps, DeepSeek agent, max-attempts 12
# (same instrument as EXP-004/005):
#   base     = no injection                                   (model's bare ability)
#   instr    = facts MINUS flash-attention-forward.md         (generic PTX primitives
#              only: mma/ldmatrix/cp.async/stmatrix/swizzle — the cross-op transferable
#              facts; this re-runs EXP-005's condition contemporaneously)
#   instrfa  = full facts bundle (instr + the FA-specific card)
# The headline number is instrfa - instr = the MARGINAL value of the op-specific card.
# Together with EXP-004 (within-op layering) and EXP-005 (gemm cards -> FA, ~1.7x weak)
# this closes the generalization story: cards transfer strongly only when they're the
# RIGHT op's cards; otherwise only the generic instruction facts carry over.
#
# Correctness for FA is judged by fp32-math (anvil okbench_runner override), NOT the
# cuDNN-bitmatch gate (< 1 bf16 ULP, rejects correct kernels). Speed is still vs cuDNN.
# Resilience: wait for DeepSeek before each rep + retry-on-empty (the server's DeepSeek
# link drops intermittently). Agent code unchanged; only the injected bundle differs.
set -u
source /nvme/share/gucheng/secrets/keys.env
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd /nvme/share/gucheng/anvil
export PYTHONPATH=/nvme/share/gucheng/anvil
mkdir -p runs_exp006
LOG=runs_exp006/driver.log
OP=flash_attention_bf16_fwd_bhsd
FA_CARD=flash-attention-forward.md

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
        --author "gucheng_${arm}" --max-attempts 12 --out-dir "runs_exp006/$arm" "$@" \
        > "runs_exp006/${arm}_rep${r}_try${try}.log" 2>&1
      local last
      last=$(ls -dt runs_exp006/"$arm"/*/ 2>/dev/null | head -1)
      if [ -n "$last" ] && [ -s "$last/results.jsonl" ]; then
        echo "[driver] $arm rep$r ok ($last) $(date)" >> "$LOG"; break
      fi
      echo "[driver] $arm rep$r try$try EMPTY -> retry $(date)" >> "$LOG"
      [ -n "$last" ] && rm -rf "$last"
    done
  done
}

echo "[exp006] start $(date)" >> "$LOG"
# phase 1: base + instr (one GPU each, clean timing)
run_arm base  6 &
run_arm instr 7 --skill-level facts --wiki-exclude "$FA_CARD" &
wait
echo "[exp006] phase1 done $(date)" >> "$LOG"
# phase 2: instrfa (full facts) — run after phase1 frees the GPUs
run_arm instrfa 6 --skill-level facts &
wait
echo "[exp006] all done $(date)" >> "$LOG"
