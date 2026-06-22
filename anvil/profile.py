"""Poor-man's profiler: occupancy/pressure signal WITHOUT ncu counters.

The 5090 box has `RmProfilingAdminOnly=1`, so a non-root `ncu` can't read the
performance counters. But the most actionable optimization signal — register
pressure, shared-memory footprint, and (the big red flag) register spilling —
falls straight out of a `--ptxas-options=-v` compile, which needs NO special
permission. We run that as a standalone `nvcc -c` pass (okbench compiles the
kernel its own way; this is a cheap second compile just to harvest the numbers),
parse the `ptxas info` lines, and turn them into a short diagnosis fed back to the
agent so it optimizes on evidence instead of guessing.

Strictly best-effort: every entry point swallows its own failures and returns an
empty dict / "" so a profiling hiccup can never break a bench.

Upgrade path: once an admin sets `RmProfilingAdminOnly=0`, add a real `ncu`
pass here for achieved occupancy / DRAM throughput / L2 — the feedback wiring
(EvalResult.resource + feedback_for_result) stays the same.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# sm_120 (RTX 5090, consumer Blackwell): 64K 32-bit registers per SM. Used only
# for a coarse "register file is ~full" flag, never as a hard occupancy claim.
_REGS_PER_SM = 65536

_RE = {
    "registers": re.compile(r"Used (\d+) registers"),
    "smem": re.compile(r"(\d+) bytes smem"),
    "spill_stores": re.compile(r"(\d+) bytes spill stores"),
    "spill_loads": re.compile(r"(\d+) bytes spill loads"),
    "stack": re.compile(r"(\d+) bytes stack frame"),
    "barriers": re.compile(r"(\d+) barriers"),
}


def _nvcc() -> str | None:
    return shutil.which("nvcc") or next(
        (p for p in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc")
         if os.path.exists(p)), None)


def ptxas_resources(repo: Path | str, arch: str, kernel_src: str,
                    *, timeout: int = 120) -> dict:
    """Compile `kernel_src` with `--ptxas-options=-v` and harvest per-kernel
    resource usage. Returns {} on any failure (never raises).

    Across multiple `__global__`s ptxas prints one block each; we keep the MAX
    registers and MAX smem (the occupancy-binding kernel) and the SUM of spills.
    """
    nvcc = _nvcc()
    if not nvcc:
        return {}
    try:
        with tempfile.TemporaryDirectory() as td:
            cu = Path(td) / "k.cu"
            cu.write_text(kernel_src)
            cp = subprocess.run(
                [nvcc, f"-arch={arch}", f"-I{repo}", "--ptxas-options=-v",
                 "-c", str(cu), "-o", str(Path(td) / "k.o")],
                capture_output=True, text=True, timeout=timeout)
            text = cp.stderr + "\n" + cp.stdout
    except Exception:
        return {}

    # ptxas emits one "Used N registers ... bytes smem" line per kernel.
    regs = [int(m) for m in _RE["registers"].findall(text)]
    smem = [int(m) for m in _RE["smem"].findall(text)]
    if not regs and not smem:
        return {}
    out = {
        "registers": max(regs) if regs else None,
        "smem_bytes": max(smem) if smem else None,
        "spill_stores": sum(int(m) for m in _RE["spill_stores"].findall(text)),
        "spill_loads": sum(int(m) for m in _RE["spill_loads"].findall(text)),
        "n_kernels": max(len(regs), len(smem)),
    }
    return out


def diagnose(resource: dict) -> str:
    """One compact, actionable line per signal. '' if nothing worth saying."""
    if not resource:
        return ""
    regs = resource.get("registers")
    smem = resource.get("smem_bytes")
    spill = (resource.get("spill_stores") or 0) + (resource.get("spill_loads") or 0)
    parts = []
    facts = []
    if regs is not None:
        facts.append(f"{regs} regs/thread")
    if smem is not None:
        facts.append(f"{smem} B smem/block")
    if spill:
        facts.append(f"{spill} B spill")
    head = "PROFILE (ptxas): " + ", ".join(facts) if facts else ""

    if spill:
        parts.append(f"⚠️ REGISTER SPILLING ({spill} B) — the kernel spills to local "
                     "memory, which serializes on slow loads/stores. Cut per-thread "
                     "state (smaller fragment/accumulator tiles, fewer live vars).")
    if regs is not None and regs >= 64:
        parts.append(f"⚠️ high register pressure ({regs}/thread) — with a large block "
                     f"this can pin you to 1 block/SM (≈{_REGS_PER_SM // regs} threads' "
                     "worth of regs per SM); on sm_120 occupancy usually dominates.")
    if smem is not None and smem >= 48 * 1024:
        parts.append(f"⚠️ heavy shared memory ({smem} B/block) — fewer blocks/SM = lower "
                     "occupancy; consider a smaller tile or swizzle-instead-of-pad.")
    if not parts:
        parts.append("resource usage looks moderate; the bottleneck is likely "
                     "algorithmic (tiling / pipelining / memory coalescing), not "
                     "occupancy.")
    return (head + "\n" + "\n".join(parts)).strip() if head else "\n".join(parts)
