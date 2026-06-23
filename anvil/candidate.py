"""Data passed between generator, runner, and orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import geometric_mean
from typing import Any


@dataclass
class Candidate:
    """A proposed kernel: a complete pure-CUDA kernel.cu implementing the op's
    stable C ABI. The model owns the full source."""
    kernel_cu: str
    notes: str = ""                 # one-line rationale from the model
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    candidate: Candidate
    stage: str                      # "validate" | "compile" | "bench"
    correct: bool = False           # all target shapes correct
    error: str | None = None        # validator / nvcc / launch error → fed back
    per_shape: list[dict] = field(default_factory=list)  # name, correct, speedup, ms
    geomean_speedup: float | None = None   # geomean(ref_ms / pure_ms) over correct shapes
    raw: dict[str, Any] = field(default_factory=dict)    # full okbench result JSON
    resource: dict[str, Any] = field(default_factory=dict)  # ptxas regs/smem/spill (profile.py)

    @property
    def ok(self) -> bool:
        return self.correct

    @staticmethod
    def from_okbench(candidate: Candidate, result: dict,
                     *, correct_field: str | None = None) -> "EvalResult":
        """Build from a parsed okbench result JSON.

        `correct_field`: when set, per-shape correctness is read from that key in
        the shape's `correctness` dict instead of okbench's own `correct` gate.
        Used for ops whose default gate is unsound — e.g. flash_attention judges
        `correct` by bit-matching cuDNN's bf16 rounding at atol=0.002 (< 1 bf16 ULP),
        which rejects mathematically-correct kernels; we instead gate on
        `sampled_vs_fp32_math_allclose` (vs the fp32 ground truth). Timing/speedup
        is unaffected (still vs the same reference). Default = okbench's gate.
        """
        per_shape = []
        speedups = []
        all_correct = bool(result.get("shapes"))
        for s in result.get("shapes", []):
            cd = s.get("correctness") or {}
            if correct_field is not None and correct_field in cd:
                correct = bool(cd.get(correct_field))
            else:
                correct = bool(s.get("correct"))
            por = s.get("pure_over_reference")
            speedup = (1.0 / por) if por else None   # ref_ms / pure_ms
            per_shape.append({
                "name": s.get("name"),
                "supported": s.get("supported"),
                "correct": correct,
                "pure_median_ms": s.get("pure_median_ms"),
                "speedup_vs_ref": speedup,
                "max_abs": (cd.get("max_abs") or cd.get("sampled_max_abs_vs_fp32")
                            or cd.get("max_abs_vs_cudnn")),
            })
            all_correct = all_correct and correct
            if correct and speedup:
                speedups.append(speedup)
        geo = geometric_mean(speedups) if speedups else None
        err = None
        if not all_correct:
            bad = [p for p in per_shape if not p["correct"]]
            err = "incorrect/unsupported shapes: " + ", ".join(
                f"{p['name']}(max_abs={p['max_abs']})" for p in bad
            )
        return EvalResult(candidate, stage="bench", correct=all_correct,
                          error=err, per_shape=per_shape, geomean_speedup=geo, raw=result)

    def summary(self) -> str:
        if self.stage == "validate":
            return f"validate failed: {(self.error or '')[:160]}"
        if self.stage == "compile":
            return f"compile failed: {(self.error or '')[:160]}"
        if not self.correct:
            return f"incorrect: {(self.error or '')[:160]}"
        g = self.geomean_speedup
        return f"correct, geomean {g:.4f}x vs reference" if g else "correct (no timing)"
