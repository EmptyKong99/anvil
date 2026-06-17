"""Anvil CLI — run on the GPU server (in the venv with torch+triton+okbench).

    # smoke: hand-written kernel through okbench, NO LLM (proves the pipeline)
    python -m anvil.cli smoke --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels --device 6

    # run: full LLM agent loop (DeepSeek by default; needs DEEPSEEK_API_KEY)
    python -m anvil.cli run --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels \
        --device 6 --max-iters 8
    #   ...or Claude:  --provider claude   (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .op import load_op
from .okbench_runner import OKBenchRunner
from .generator import HumanGenerator, make_generator
from .orchestrator import Orchestrator


def _runner(args, op, out_dir: Path | None = None) -> OKBenchRunner:
    return OKBenchRunner(
        op, hardware=args.hardware, platform=args.platform, arch=args.arch,
        device=args.device, author=args.author, python=args.python, suite=args.suite,
        out_dir=out_dir,
    )


def _cmd_smoke(args):
    op = load_op(args.repo, args.op)
    runner = _runner(args, op)
    cand = HumanGenerator().propose(op, [])
    res = runner.evaluate(cand, f"{args.author}_smoke")
    print(f"[smoke] {op.name}: {res.summary()}")
    for p in res.per_shape:
        print(f"   {p}")
    if not res.correct and res.error:
        print("--- error ---\n" + res.error)


def _cmd_run(args):
    op = load_op(args.repo, args.op)
    # absolute: okbench runs with cwd=OpenKernels, so a relative --output would
    # land under the repo, not here, and we'd misread it as "compile failed".
    run_dir = (Path(args.out_dir).expanduser() / f"{op.name}_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    generator = make_generator(args.provider, args.model, inject_skill=args.inject_skill)
    report = Orchestrator(
        op, generator, _runner(args, op, out_dir=run_dir),
        variant_prefix=args.author, max_iters=args.max_iters,
        target_speedup=args.target_speedup, run_dir=run_dir,
    ).run()
    print(f"\n[anvil] artifacts archived in {run_dir}")
    if report.best:
        print(f"[anvil] best: {report.best.summary()}  (best.cu in run dir)")


def main():
    p = argparse.ArgumentParser(prog="anvil")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--op", default="gemm_bf16_nt")
    common.add_argument("--repo", required=True, help="path to the OpenKernels repo")
    common.add_argument("--hardware", default="5090")
    common.add_argument("--platform", default="sm120_rtx5090")
    common.add_argument("--arch", default="sm_120a")
    common.add_argument("--device", type=int, default=6)
    common.add_argument("--author", default="gucheng")
    common.add_argument("--suite", default="required_5")
    common.add_argument("--python", default=None, help="python running okbench (default: this one)")

    ps = sub.add_parser("smoke", parents=[common], help="hand-written kernel, no LLM")
    ps.set_defaults(func=_cmd_smoke)

    pr = sub.add_parser("run", parents=[common], help="full LLM agent loop")
    pr.add_argument("--provider", default="deepseek", choices=["deepseek", "claude"],
                    help="LLM backend (default: deepseek, OpenAI-compatible API)")
    pr.add_argument("--model", default=None, help="override the provider's default model")
    pr.add_argument("--max-iters", type=int, default=8)
    pr.add_argument("--target-speedup", type=float, default=1.0)
    pr.add_argument("--out-dir", default="runs",
                    help="archive each run under <out-dir>/<op>_<timestamp>/ (default: runs/)")
    pr.add_argument("--inject-skill", action="store_true",
                    help="inject the distilled PTX GEMM skill into the system prompt (A/B test)")
    pr.set_defaults(func=_cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
