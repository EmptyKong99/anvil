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
from . import wiki


def _skill_bundle(args) -> str:
    """Assemble the injected knowledge bundle. --skill-level (layered wiki cards)
    wins; else the legacy --inject-skill boolean; else nothing."""
    level = getattr(args, "skill_level", "none")
    if level and level != "none":
        return wiki.load_bundle(level, getattr(args, "wiki_dir", None))
    return ""  # _cmd_* pass inject_skill through for the legacy path


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


def _run_dir(args, op) -> Path:
    # absolute: okbench runs with cwd=OpenKernels, so a relative --output would
    # land under the repo, not here, and we'd misread it as "compile failed".
    return (Path(args.out_dir).expanduser() / f"{op.name}_{datetime.now():%Y%m%d_%H%M%S}").resolve()


def _cmd_run(args):
    op = load_op(args.repo, args.op)
    run_dir = _run_dir(args, op)
    generator = make_generator(args.provider, args.model,
                               inject_skill=args.inject_skill,
                               extra_system=_skill_bundle(args))
    report = Orchestrator(
        op, generator, _runner(args, op, out_dir=run_dir),
        variant_prefix=args.author, max_iters=args.max_iters,
        target_speedup=args.target_speedup, run_dir=run_dir,
    ).run()
    print(f"\n[anvil] artifacts archived in {run_dir}")
    if report.best:
        print(f"[anvil] best: {report.best.summary()}  (best.cu in run dir)")


def _cmd_agent(args):
    from .agent import AgentRunner          # local import: only agent mode needs openai tools
    op = load_op(args.repo, args.op)
    run_dir = _run_dir(args, op)
    report = AgentRunner(
        op, _runner(args, op, out_dir=run_dir),
        model=args.model, max_attempts=args.max_attempts,
        target_speedup=args.target_speedup, variant_prefix=args.author,
        inject_skill=args.inject_skill, extra_system=_skill_bundle(args),
        run_dir=run_dir,
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
                    help="legacy: inject the single distilled PTX skill string (EXP-001..003)")
    pr.add_argument("--skill-level", default="none", choices=list(wiki.LEVELS),
                    help="layered wiki bundle to inject (EXP-004): none|facts|heuristics|full")
    pr.add_argument("--wiki-dir", default=None,
                    help="path to forge wiki/ptx (default: sibling forge repo)")
    pr.set_defaults(func=_cmd_run)

    # agent: Route-AVO-lite — model drives a bench_kernel tool loop (DeepSeek v1)
    pa = sub.add_parser("agent", parents=[common],
                        help="autonomous tool-loop agent (model compiles+iterates itself)")
    pa.add_argument("--model", default=None, help="override the default DeepSeek model")
    pa.add_argument("--max-attempts", type=int, default=8,
                    help="budget of bench_kernel (compile+bench) calls the agent may make")
    pa.add_argument("--target-speedup", type=float, default=1.0)
    pa.add_argument("--out-dir", default="runs",
                    help="archive each run under <out-dir>/<op>_<timestamp>/ (default: runs/)")
    pa.add_argument("--inject-skill", action="store_true",
                    help="legacy: inject the single distilled PTX skill string (EXP-001..003)")
    pa.add_argument("--skill-level", default="none", choices=list(wiki.LEVELS),
                    help="layered wiki bundle to inject (EXP-004): none|facts|heuristics|full")
    pa.add_argument("--wiki-dir", default=None,
                    help="path to forge wiki/ptx (default: sibling forge repo)")
    pa.set_defaults(func=_cmd_agent)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
