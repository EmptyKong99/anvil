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
    """Facts-first feedback: report what ptxas MEASURED + a neutral arithmetic
    derivation. NO prescriptions ('reduce X', 'smaller tile', 'occupancy dominates').

    Those prescriptions encode a prior that has been wrong: e.g. forge's FA v7 hit
    56% via cp.async pipelining at LOW occupancy, not by cutting registers — so
    'reduce regs for occupancy' can be actively misleading here. Whether occupancy
    or latency-hiding is the bottleneck is the model's call (it has the wiki), and
    okbench is the only judge. We hand it the numbers, not our opinion.
    """
    if not resource:
        return ""
    regs = resource.get("registers")
    smem = resource.get("smem_bytes")
    spill = (resource.get("spill_stores") or 0) + (resource.get("spill_loads") or 0)
    facts = []
    if regs is not None:
        facts.append(f"{regs} regs/thread")
    if smem is not None:
        facts.append(f"{smem} B smem/block")
    facts.append(f"{spill} B spill→local")
    if regs is None and smem is None:
        return ""
    out = ["PROFILE (ptxas, measured facts): " + ", ".join(facts) + "."]
    # Neutral arithmetic, not advice — let the model weigh it against the wiki.
    notes = []
    if regs:
        notes.append(f"the 64K-reg/SM file fits ≤{_REGS_PER_SM // regs} threads/SM at "
                     "this reg count")
    if spill:
        notes.append(f"{spill} B of register state lives in (slower) local memory")
    if notes:
        out.append("(derivation, not advice: " + "; ".join(notes)
                   + ". Whether occupancy or latency-hiding bounds you is yours to "
                   "judge — okbench decides.)")
    return "\n".join(out)
