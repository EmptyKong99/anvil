#!/usr/bin/env bash
# EXP-007 — does the ptxas PROFILER feedback help the agent climb?
# Both arms inject the full facts bundle (= EXP-006's instrfa knowledge condition,
# the strong config we'd actually ship); the ONLY variable is the profiler:
#   noprof = --skill-level facts --no-profile   (= v1-feedback-loop behaviour: bench
#            gives speed only, no "why slow")
#   prof   = --skill-level facts                (profiler on: each CORRECT bench also
#            returns ptxas regs/smem/spill -> occupancy/pressure diagnosis)
# headline = prof - noprof (median %cuDNN). The profiler should help most here because
# the FA-card kernels are already tensor-core + correct but register-bound (the 42.9%
# champion hit 255 regs/thread) -> evidence about regs/smem/spill is directly actionable.
# op = flash_attention_bf16_fwd_bhsd, n=5/arm, max-attempts 12, same instrument as
# EXP-006. FA correctness gated on fp32-math. ptxas compile is CPU-only (no GPU
# contention with bench). Resilience: wait-for-DeepSeek + retry-on-empty per rep.
set -u
source /nvme/share/gucheng/secrets/keys.env
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd /nvme/share/gucheng/anvil
export PYTHONPATH=/nvme/share/gucheng/anvil
mkdir -p runs_exp007
LOG=runs_exp007/driver.log
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
        --author "gucheng_${arm}" --max-attempts 12 --skill-level facts \
        --out-dir "runs_exp007/$arm" "$@" \
        > "runs_exp007/${arm}_rep${r}_try${try}.log" 2>&1
      local last
      last=$(ls -dt runs_exp007/"$arm"/*/ 2>/dev/null | head -1)
      if [ -n "$last" ] && [ -s "$last/results.jsonl" ]; then
        echo "[driver] $arm rep$r ok ($last) $(date)" >> "$LOG"; break
      fi
      echo "[driver] $arm rep$r try$try EMPTY -> retry $(date)" >> "$LOG"
      [ -n "$last" ] && rm -rf "$last"
    done
  done
}

echo "[exp007] start $(date)" >> "$LOG"
run_arm noprof 6 --no-profile &
run_arm prof   7 &
wait
echo "[exp007] all done $(date)" >> "$LOG"
