#!/usr/bin/env python3
"""Render each experiment as a compact run × attempt markdown matrix.

Rows = runs (arm + rep) — each row is one run's trajectory read left→right.
Columns = step (Route-B iter / agent attempt). Cell:
  <number>  geomean speedup vs cuBLAS (correct kernel)
  xC        build failed (compile / validate)
  xW        compiled but wrong result
  .         no attempt at that step
Reads runs_exp*/<arm>/<run-ts>/results.jsonl (reps = run dirs in time order).
One-shot generator: run it, then paste each experiment's matrix into the matching
section of EXPERIMENTS.md (completed experiments are frozen — no re-gen needed):
    python collect_results.py [root]
"""
from __future__ import annotations

import glob
import json
import os
import sys


def cell(r: dict) -> str:
    if r.get("correct") and r.get("geomean_speedup") is not None:
        return f"{r['geomean_speedup']:.3f}"
    if r.get("stage") in ("compile", "validate"):
        return "xC"
    if r.get("stage") == "bench":
        return "xW"
    return "x"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out = []
    out.append("> 格子:数字=加速比(geomean vs cuBLAS,正确) · `xC`=编译/校验失败 · "
               "`xW`=结果错 · `.`=该步未跑。列名 b=base(no-skill)/s=skill,后接 rep 号。\n")
    for exp_dir in sorted(glob.glob(os.path.join(root, "runs_exp*/"))):
        exp = os.path.basename(exp_dir.rstrip("/")).replace("runs_", "")
        cols, grid, steps = [], {}, set()
        for arm_dir in sorted(glob.glob(exp_dir + "*/")):
            arm = os.path.basename(arm_dir.rstrip("/"))
            for rep, run in enumerate(sorted(glob.glob(arm_dir + "*/")), 1):
                rj = os.path.join(run, "results.jsonl")
                if not os.path.exists(rj):
                    continue
                ci = len(cols)
                cols.append(f"{arm[:1]}{rep}")
                for line in open(rj):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    step = r.get("attempt", r.get("iter"))
                    steps.add(step)
                    grid[(ci, step)] = cell(r)
        if not cols:
            continue
        out.append(f"### {exp}")
        ss = sorted(steps)
        out.append("| run | " + " | ".join(str(s) for s in ss) + " |")
        out.append("|" + "---|" * (len(ss) + 1))
        for ci, name in enumerate(cols):
            cells = [grid.get((ci, s), ".") for s in ss]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
