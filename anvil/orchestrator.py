"""The agent loop: generate -> okbench -> feed back -> repeat, keep the best."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .op import Op
from .candidate import EvalResult
from .generator import Generator
from .okbench_runner import OKBenchRunner


@dataclass
class RunReport:
    best: EvalResult | None
    history: list[EvalResult]


class Orchestrator:
    def __init__(self, op: Op, generator: Generator, runner: OKBenchRunner,
                 *, variant_prefix: str = "anvil", max_iters: int = 8,
                 target_speedup: float = 1.0, verbose: bool = True,
                 run_dir: Path | None = None):
        self.op = op
        self.generator = generator
        self.runner = runner
        self.variant_prefix = variant_prefix
        self.max_iters = max_iters
        self.target_speedup = target_speedup
        self.verbose = verbose
        self.run_dir = run_dir          # archive kernels + results here (under gucheng)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[anvil] {msg}", flush=True)

    def run(self) -> RunReport:
        history: list[EvalResult] = []
        best: EvalResult | None = None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"archiving run to {self.run_dir}")

        for it in range(1, self.max_iters + 1):
            self._log(f"iter {it}/{self.max_iters}: generating…")
            try:
                candidate = self.generator.propose(self.op, history)
            except Exception as e:
                self._log(f"iter {it}: generation failed, skipping — {e}")
                self._record(it, None, None, gen_error=str(e))
                continue
            variant = f"{self.variant_prefix}_v{it:02d}"

            result = self.runner.evaluate(candidate, variant)
            history.append(result)
            self._log(f"iter {it} [{variant}]: {result.summary()}")
            self._record(it, variant, result)

            if result.correct and result.geomean_speedup is not None:
                if best is None or result.geomean_speedup > best.geomean_speedup:
                    best = result
                if result.geomean_speedup >= self.target_speedup:
                    self._log(f"target {self.target_speedup:.2f}x reached — stopping")
                    break

        if best is not None:
            self._log(f"BEST: {best.summary()}")
        else:
            self._log("no correct kernel produced")
        self._finalize(best, history)
        return RunReport(best=best, history=history)

    # --- artifact archiving (under run_dir, e.g. anvil/runs/<op>_<ts>/) ------

    def _record(self, it: int, variant: str | None, result: EvalResult | None,
                *, gen_error: str | None = None) -> None:
        if not self.run_dir:
            return
        if result is not None and variant is not None:
            (self.run_dir / f"iter{it:02d}_{variant}.cu").write_text(result.candidate.kernel_cu)
        row = {
            "iter": it, "variant": variant,
            "stage": (result.stage if result else "generate"),
            "correct": (result.correct if result else False),
            "geomean_speedup": (result.geomean_speedup if result else None),
            "error": (gen_error if gen_error else (result.error[:500] if result and result.error else None)),
        }
        with (self.run_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _finalize(self, best: EvalResult | None, history: list[EvalResult]) -> None:
        if not self.run_dir:
            return
        if best is not None:
            (self.run_dir / "best.cu").write_text(best.candidate.kernel_cu)
        summary = {
            "op": self.op.name,
            "iters": len(history),
            "best_geomean_speedup": (best.geomean_speedup if best else None),
            "best_notes": (best.candidate.notes if best else None),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
