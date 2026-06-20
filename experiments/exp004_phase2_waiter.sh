#!/usr/bin/env bash
# Re-run EXP-004 phase 2 (heuristics + full) — these failed with APITimeoutError
# because DeepSeek was unreachable from the server during the first phase-2 window.
# Poll DeepSeek until it's back (up to ~2h), then run the two arms. base/facts from
# phase 1 are already good and untouched.
set -u
source /nvme/share/gucheng/secrets/keys.env
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd /nvme/share/gucheng/anvil
export PYTHONPATH=/nvme/share/gucheng/anvil
mkdir -p runs_exp004

echo "[waiter] start $(date)" >> runs_exp004/phase2.log
for i in $(seq 1 120); do
  code=$(curl -s -m 12 -o /dev/null -w "%{http_code}" https://api.deepseek.com/v1/models \
         -H "Authorization: Bearer $DEEPSEEK_API_KEY")
  echo "[waiter] $(date) deepseek http=$code" >> runs_exp004/phase2.log
  [ "$code" = "200" ] && { echo "[waiter] reachable, launching phase 2" >> runs_exp004/phase2.log; break; }
  sleep 60
done

# clear the failed (empty) phase-2 run dirs so reps stay clean at 5
rm -rf runs_exp004/heuristics/* runs_exp004/full/*

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

run_arm heuristics 6 --skill-level heuristics &
run_arm full       7 --skill-level full &
wait
echo "phase2 done $(date)" > runs_exp004/DONE2
echo "[waiter] phase2 done $(date)" >> runs_exp004/phase2.log
