#!/usr/bin/env python3
"""Render each experiment as a compact run × attempt markdown matrix.

Rows = runs (arm + rep) — each row is one run's trajectory read left→right.
Columns = step (Route-B iter / agent attempt). Cell:
  <number>  geomean speedup vs cuBLAS as a percent (100 = parity); bold = row max
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


def cell(r: dict) -> tuple[str, float | None]:
    """(display string, numeric speedup or None) — None marks a non-number cell
    so the row-max bolding skips failures."""
    if r.get("correct") and r.get("geomean_speedup") is not None:
        v = r["geomean_speedup"]
        return (f"{v * 100:.1f}%", v)           # percent of cuBLAS, 1 decimal
    if r.get("stage") in ("compile", "validate"):
        return ("xC", None)
    if r.get("stage") == "bench":
        return ("xW", None)
    return ("x", None)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out = []
    out.append("> 格子:数字=对 cuBLAS 的百分比(100=持平),**加粗=该 run 峰值** · "
               "`xC`=编译/校验失败 · `xW`=结果错 · `.`=未跑。行=run(base/skill + rep 号),列=step。\n")
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
                cols.append(f"{arm}-{rep}")
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
            pairs = [grid.get((ci, s), (".", None)) for s in ss]
            nums = [(i, p[1]) for i, p in enumerate(pairs) if p[1] is not None]
            best_i = max(nums, key=lambda t: t[1])[0] if nums else -1   # bold row max
            cells = [f"**{t}**" if i == best_i else t for i, (t, _) in enumerate(pairs)]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
