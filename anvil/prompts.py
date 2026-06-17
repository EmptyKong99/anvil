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


# Distilled PTX skill, injected into the system prompt only when --inject-skill.
# Source (provenance): our OWN kernels gemm_bf16_nt v7_mma.cu / v8_pipe.cu, verified
# on RTX 5090 (sm_120) by okbench at ~0.92x cuBLAS. This is distilled from real
# practice + measurement, NOT from the model's latent knowledge. A/B target.
PTX_GEMM_SKILL = """\
EXPERT PTX RECIPE for fast BF16 GEMM on sm_120 (distilled from our own kernels,
measured on RTX 5090 via okbench: the wmma path plateaus ~0.88x cuBLAS; raw
mma.sync is REQUIRED to beat it — our v8 reaches ~0.92x).

Instructions (sm_80+, valid on sm_120 — NOT wgmma / tcgen05):
  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0..d3},{a0..a3},{b0,b1},{c0..c3};
  ldmatrix.sync.aligned.m8n8.x4.shared.b16  {r0..r3},[addr];  // A 16x16
  ldmatrix.sync.aligned.m8n8.x2.shared.b16  {r0,r1},[addr];   // B 16x8
Per thread: A=4xb32(8 bf16), B=2xb32(4 bf16), C/D=4xf32. addr = u32 from
__cvta_generic_to_shared(ptr) (a SHARED address).

C/D accumulator layout for the epilogue store (group=lane/4, tidg=lane%4):
  d0->(group,2*tidg) d1->(group,2*tidg+1) d2->(group+8,2*tidg) d3->(group+8,2*tidg+1)

ldmatrix addressing (each lane supplies one 8-wide row of an 8x8 tile):
  A row-major MxK in shared: addr=&As[rowbase+lane%16][k+(lane/16)*8]
  B col-major KxN in shared:  addr=&Bs[nbase+lane%8][k+((lane/8)&1)*8]

CRITICAL CORRECTNESS GOTCHA (cost us a debug iteration): for NT GEMM C=A·B^T,
input B[N,K] row-major stored as Bs[n][k] is ALREADY column-major K×N — exactly
what the .col B operand wants. Load it with PLAIN ldmatrix, NOT ldmatrix.trans.
.trans compiles and runs fast but is NUMERICALLY WRONG. (Use .trans only if B is
stored row-major N×K.)

Recipe that worked: 128x128 tile, 8 warps, cp.async double-buffered global->shared,
register-resident f32 accumulators, ldmatrix feeding mma.sync per 16-wide k substep;
software-pipeline (issue both k-substep loads, then the mma) for a final ~1pp."""


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
