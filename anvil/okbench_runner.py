"""Bridge to the judge: deploy a submission, run okbench, parse the verdict.

The deploy->okbench->parse mechanics live in `okeval` (the canonical eval that
forge's `bench.sh` also calls). This wraps that outcome in an `EvalResult` for
the agent loop. Anvil runs ON the GPU box, in the venv that has okbench.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import okeval
from .op import Op
from .candidate import Candidate, EvalResult


# Ops whose okbench `correct` gate is unsound → judge correctness by a fp32-truth
# field in the shape's `correctness` dict instead. flash_attention's default gate
# bit-matches cuDNN's bf16 rounding at atol=0.002 (< 1 bf16 ULP) and rejects
# correct kernels; gate on the fp32-math reference (atol 0.008) instead.
_CORRECT_FIELD_BY_OP = {
    "flash_attention_bf16_fwd_bhsd": "sampled_vs_fp32_math_allclose",
}


class OKBenchRunner:
    def __init__(self, op: Op, *, hardware: str = "5090",
                 platform: str = "sm120_rtx5090", arch: str = "sm_120a",
                 device: int = 6, author: str = "gucheng",
                 python: str | None = None, suite: str = "required_5",
                 timeout: int = 1800, out_dir: Path | None = None):
        self.op = op
        self.repo = op.repo_root
        self.hardware = hardware
        self.platform = platform
        self.arch = arch
        self.device = device
        self.author = author
        self.python = python or sys.executable    # the venv python running okbench
        self.suite = suite
        self.timeout = timeout
        self.out_dir = out_dir                      # where to keep okbench JSONs (not /tmp)

    def evaluate(self, candidate: Candidate, variant: str) -> EvalResult:
        out_root = self.out_dir or Path(tempfile.gettempdir())
        out_json = out_root / f"{self.op.name}_{variant}.json"

        outcome = okeval.evaluate(
            self.repo, self.op.name, variant, candidate.kernel_cu,
            out_json=out_json, hardware=self.hardware, platform=self.platform,
            arch=self.arch, author=self.author, suite=self.suite,
            device=self.device, python=self.python, timeout=self.timeout,
            notes=candidate.notes,
        )
        if not outcome.ok:
            return EvalResult(candidate, stage=outcome.stage, error=outcome.error)
        return EvalResult.from_okbench(candidate, outcome.result,
                                       correct_field=_CORRECT_FIELD_BY_OP.get(self.op.name))
