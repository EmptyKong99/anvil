"""Offline sanity checks for the agent plumbing (no GPU, no okbench, no LLM).

These run anywhere (incl. the Mac) and just verify the pieces wire together.
The real end-to-end check is `python -m anvil.cli smoke` on the GPU server.
"""
from anvil.candidate import Candidate, EvalResult
from anvil.baselines import SMOKE_KERNELS


def test_baseline_kernel_has_entry_symbol():
    src = SMOKE_KERNELS["gemm_bf16_nt"]
    assert 'openkernels_launch_gemm_bf16_nt' in src
    assert '__global__' in src
    assert '#include "ops/gemm_bf16_nt/interface.h"' in src


def test_eval_result_from_okbench_parses_speedup_and_correctness():
    fake = {
        "shapes": [
            {"name": "s1", "supported": True, "correct": True,
             "pure_median_ms": 0.20, "pure_over_reference": 1.25,   # 25% slower -> 0.8x
             "correctness": {"max_abs": 0.0}},
            {"name": "s2", "supported": True, "correct": True,
             "pure_median_ms": 0.10, "pure_over_reference": 0.50,   # 2x faster
             "correctness": {"max_abs": 0.0}},
        ]
    }
    r = EvalResult.from_okbench(Candidate(kernel_cu="x"), fake)
    assert r.correct
    # geomean(0.8, 2.0) = 1.264...
    assert abs(r.geomean_speedup - (0.8 * 2.0) ** 0.5) < 1e-6


def test_eval_result_flags_incorrect_shape():
    fake = {"shapes": [
        {"name": "s1", "correct": True, "pure_over_reference": 1.0, "correctness": {"max_abs": 0.0}},
        {"name": "s2", "correct": False, "pure_over_reference": None, "correctness": {"max_abs": 9.9}},
    ]}
    r = EvalResult.from_okbench(Candidate(kernel_cu="x"), fake)
    assert not r.correct
    assert "s2" in r.error
