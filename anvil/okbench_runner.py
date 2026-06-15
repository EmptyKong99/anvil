"""Bridge to the judge: write a submission, run okbench, parse the verdict.

Runs okbench as a local subprocess, so Anvil is expected to run ON the GPU box
(in the venv that has torch+triton+okbench), alongside the OpenKernels repo.
The agent loop (LLM calls) and this evaluation both run there.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from .op import Op
from .candidate import Candidate, EvalResult


class OKBenchRunner:
    def __init__(self, op: Op, *, hardware: str = "5090",
                 platform: str = "sm120_rtx5090", arch: str = "sm_120a",
                 device: int = 6, author: str = "gucheng",
                 python: str | None = None, suite: str = "required_5",
                 timeout: int = 1800):
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

    # --- submission files ---------------------------------------------------

    def _variant_dir(self, variant: str) -> Path:
        return self.repo / "submissions" / self.hardware / self.op.name / variant

    def _write_submission(self, candidate: Candidate, variant: str) -> Path:
        vdir = self._variant_dir(variant)
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "kernel.cu").write_text(candidate.kernel_cu)
        metadata = {
            "author": self.author,
            "op": self.op.name,
            "variant": variant,
            "status": "draft",
            "entry_symbol": self.op.entry_symbol,
            "pure_cuda": True,
            "arch": [self.arch.replace("sm_", "sm")],
            "features": ["bf16"],
            "notes": candidate.notes[:200],
        }
        (vdir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        return vdir

    # --- okbench invocation -------------------------------------------------

    def _okbench(self, *args: str) -> subprocess.CompletedProcess:
        cmd = [self.python, "-m", "okbench.cli", *args]
        return subprocess.run(cmd, cwd=self.repo, capture_output=True,
                              text=True, timeout=self.timeout)

    def evaluate(self, candidate: Candidate, variant: str) -> EvalResult:
        self._write_submission(candidate, variant)

        # 1. validate (cheap gate: forbidden tokens, missing symbol, pure_cuda)
        v = self._okbench("validate", "--op", self.op.name,
                          "--hardware", self.hardware, "--variant", variant)
        if v.returncode != 0:
            return EvalResult(candidate, stage="validate",
                              error=(v.stdout + v.stderr).strip()[-2000:])

        # 2. compile + correctness + timing through the stable ABI
        out_json = Path(tempfile.gettempdir()) / f"anvil_{self.op.name}_{variant}.json"
        b = self._okbench(
            self.op.bench_cmd,
            "--op", self.op.name, "--variant", variant,
            "--hardware", self.hardware, "--platform", self.platform,
            "--arch", self.arch, "--runner-id", f"{self.author}_{self.hardware}_dev",
            "--status", "community_reported", "--suite", self.suite,
            "--device", str(self.device), "--output", str(out_json),
        )
        if b.returncode != 0 or not out_json.exists():
            # nvcc compile error or a launch crash both land here
            return EvalResult(candidate, stage="compile",
                              error=(b.stdout + b.stderr).strip()[-3000:])

        result = json.loads(out_json.read_text())
        return EvalResult.from_okbench(candidate, result)
