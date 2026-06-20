#!/usr/bin/env bash
set -u
source /nvme/share/gucheng/secrets/keys.env
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd /nvme/share/gucheng/anvil
export PYTHONPATH=/nvme/share/gucheng/anvil
mkdir -p runs_exp003
run_arm() {
  local arm="$1" dev="$2"; shift 2
  for r in 1 2 3 4 5; do
    echo "[driver] $arm rep$r start $(date)" >> runs_exp003/driver.log
    .venv/bin/python -m anvil.cli agent --op gemm_bf16_nt \
      --repo /nvme/share/gucheng/OpenKernels --device "$dev" \
      --author "gucheng_${arm}" --max-attempts 12 --out-dir "runs_exp003/$arm" "$@" \
      > "runs_exp003/${arm}_rep${r}.log" 2>&1
    echo "[driver] $arm rep$r done  $(date)" >> runs_exp003/driver.log
  done
}
run_arm base  6 &
run_arm skill 7 --inject-skill &
wait
echo "done $(date)" > runs_exp003/DONE
