"""Prompt construction for the kernel generator."""
from __future__ import annotations

from .op import Op
from .candidate import EvalResult

SYSTEM = """You are an expert CUDA kernel engineer. You write release-quality, \
PURE CUDA kernels for an NVIDIA GeForce RTX 5090 (Blackwell, compute capability \
sm_120a, compiled with nvcc 13). You will be given an operator's C ABI and \
reference, and — on later turns — the compiler/correctness/benchmark feedback \
from your previous attempt.

Emit one complete `kernel.cu`. Hard rules:
- PURE CUDA ONLY. Do NOT call or include cuBLAS, cuDNN, CUTLASS runtime, \
DeepGEMM, FlashKDA, or any TileLang runtime. Hand-write the kernel. (A validator \
rejects those tokens.)
- The file MUST `#include "{header_path}"` and define the exact extern "C" \
entry symbol `{entry_symbol}` with the signature from that header. Return \
`cudaSuccess` on success, `cudaErrorNotSupported` only for shapes you truly \
don't handle.
- It must be correct within atol={atol}, rtol={rtol} for every target shape, \
checked against the reference. Correctness first — a fast wrong kernel scores 0.
- Then make it fast. The score is geomean speedup vs the reference across the \
target shapes; aim to match or beat it.

Performance guidance for sm_120 BF16 GEMM: block/warp tiling with shared memory, \
fp32 accumulation, vectorized 128-bit loads, register blocking, avoid bank \
conflicts, enough occupancy. Use mma/tensor-core PTX if you can do so correctly; \
otherwise a well-tiled SIMT kernel is an acceptable first step.

The kernel.cu must compile and run as-is."""


def build_user(op: Op) -> str:
    return (
        f"Operator: {op.name}\n{op.description}\n\n"
        f"Target shapes (all must pass), name: m,n,k =\n" +
        "\n".join(f"  {s['name']}: {s['m']},{s['n']},{s['k']}" for s in op.shapes) +
        f"\n\nC ABI you must implement (ops/{op.name}/interface.h):\n"
        f"```c\n{op.interface_h}\n```\n\n"
        f"Reference (ground truth the kernel is compared against):\n"
        f"```python\n{op.reference_src}\n```\n"
    )


def build_feedback(history: list[EvalResult]) -> str:
    if not history:
        return "This is your first attempt. Write a correct, reasonably tiled kernel."
    last = history[-1]
    lines = [f"Previous attempt #{len(history)}: {last.summary()}"]
    if last.candidate.notes:
        lines.append(f"Your stated approach: {last.candidate.notes}")

    if last.stage in ("validate", "compile") and last.error:
        label = "Validator rejected it" if last.stage == "validate" else "nvcc / launch error"
        lines.append(f"{label} — fix this exactly:\n{last.error}")
    elif not last.correct:
        lines.append("It compiled but was INCORRECT. Fix the math/indexing:\n" + (last.error or ""))
        lines.append("Reminder: C = A[M,K] @ B[N,K]^T (row-major), fp32 accumulate.")
    else:
        rows = "\n".join(
            f"  {p['name']}: {p['speedup_vs_ref']:.3f}x  ({p['pure_median_ms']:.3f} ms)"
            for p in last.per_shape if p.get("speedup_vs_ref")
        )
        lines.append(f"It was CORRECT. Per-shape speedup vs reference:\n{rows}\n"
                     f"geomean {last.geomean_speedup:.4f}x. Now make it faster — "
                     f"attack the slowest shapes.")

    best = _best(history)
    if best and best is not last:
        lines.append(f"Best correct so far: geomean {best.geomean_speedup:.4f}x.")
    return "\n".join(lines)


def _best(history: list[EvalResult]) -> EvalResult | None:
    correct = [h for h in history if h.correct and h.geomean_speedup]
    return max(correct, key=lambda h: h.geomean_speedup) if correct else None
