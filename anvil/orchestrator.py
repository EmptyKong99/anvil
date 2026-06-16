"""The agent loop: generate -> okbench -> feed back -> repeat, keep the best."""
from __future__ import annotations

from dataclasses import dataclass

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
                 target_speedup: float = 1.0, verbose: bool = True):
        self.op = op
        self.generator = generator
        self.runner = runner
        self.variant_prefix = variant_prefix
        self.max_iters = max_iters
        self.target_speedup = target_speedup
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[anvil] {msg}", flush=True)

    def run(self) -> RunReport:
        history: list[EvalResult] = []
        best: EvalResult | None = None

        for it in range(1, self.max_iters + 1):
            self._log(f"iter {it}/{self.max_iters}: generating…")
            try:
                candidate = self.generator.propose(self.op, history)
            except Exception as e:
                self._log(f"iter {it}: generation failed, skipping — {e}")
                continue
            variant = f"{self.variant_prefix}_v{it:02d}"

            result = self.runner.evaluate(candidate, variant)
            history.append(result)
            self._log(f"iter {it} [{variant}]: {result.summary()}")

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
        return RunReport(best=best, history=history)
