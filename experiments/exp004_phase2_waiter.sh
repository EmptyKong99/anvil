#!/usr/bin/env bash
# Re-run EXP-004 phase 2 (heuristics + full), n=5 each, ROBUST to DeepSeek's
# intermittent unreachability from the server. base/facts (phase 1) are untouched.
# Agent code is unchanged (same instrument as base/facts); resilience lives here in
# the driver: before each rep wait until DeepSeek is reachable, and if a rep dies on
# APITimeout (empty results.jsonl) retry just that rep (≤3×). Guarantees clean 5/5
# whenever the network eventually cooperates.
set -u
source /nvme/share/gucheng/secrets/keys.env
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd /nvme/share/gucheng/anvil
export PYTHONPATH=/nvme/share/gucheng/anvil
mkdir -p runs_exp004
LOG=runs_exp004/phase2.log

wait_ds() {                                   # block until DeepSeek reachable (~2h cap)
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
      echo "[driver] $arm rep$r try$try start $(date)" >> runs_exp004/driver.log
      .venv/bin/python -m anvil.cli agent --op gemm_bf16_nt \
        --repo /nvme/share/gucheng/OpenKernels --device "$dev" \
        --author "gucheng_${arm}" --max-attempts 12 --out-dir "runs_exp004/$arm" "$@" \
        > "runs_exp004/${arm}_rep${r}_try${try}.log" 2>&1
      local last
      last=$(ls -dt runs_exp004/"$arm"/*/ 2>/dev/null | head -1)
      if [ -n "$last" ] && [ -s "$last/results.jsonl" ]; then
        echo "[driver] $arm rep$r ok ($last) $(date)" >> runs_exp004/driver.log; break
      fi
      echo "[driver] $arm rep$r try$try EMPTY -> retry $(date)" >> runs_exp004/driver.log
      [ -n "$last" ] && rm -rf "$last"
    done
  done
}

echo "[waiter v2] start $(date)" >> "$LOG"
rm -rf runs_exp004/heuristics/* runs_exp004/full/*    # clean full re-run of both arms
run_arm heuristics 6 --skill-level heuristics &
run_arm full       7 --skill-level full &
wait
echo "phase2 v2 done $(date)" > runs_exp004/DONE2
echo "[waiter v2] done $(date)" >> "$LOG"
