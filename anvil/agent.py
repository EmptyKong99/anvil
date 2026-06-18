"""Route-AVO-lite: an autonomous tool-loop agent (vs the one-shot Route B).

The model is given ONE tool, `bench_kernel`, that compiles + benchmarks a kernel
through okbench (the same `OKBenchRunner` Route B uses) and hands back the real
verdict. The MODEL drives the loop — it decides when to retry and reads the nvcc
error / speedups itself — instead of the outer python deciding each step. That is
the only thing that changes vs Route B; the op, judge, skill injection
(`--inject-skill`) and archive format are all reused, so an agent run stays
directly comparable to the Route-B ablation (EXP-001).

v1 backs DeepSeek (OpenAI-compatible function calling); Claude can be added later.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .op import Op
from .candidate import Candidate, EvalResult
from .okbench_runner import OKBenchRunner
from .orchestrator import RunReport
from . import prompts
from .generator import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DEEPSEEK_MAX_TOKENS


BENCH_TOOL = {
    "type": "function",
    "function": {
        "name": "bench_kernel",
        "description": (
            "Compile this pure-CUDA kernel.cu and benchmark it on a real RTX 5090 "
            "(sm_120) via okbench against the reference. Returns the nvcc compiler "
            "error if it fails to build, a correctness report if the result is "
            "wrong, or per-shape speedups vs the reference if it works. Call it "
            "repeatedly to iterate on the kernel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kernel_cu": {
                    "type": "string",
                    "description": "Complete contents of kernel.cu: #includes the op "
                                   "header, defines the extern \"C\" entry symbol, "
                                   "compiles as-is. PURE CUDA only (no cuBLAS/cuDNN/CUTLASS).",
                },
                "notes": {"type": "string", "description": "One line: what you changed/tried this time."},
            },
            "required": ["kernel_cu", "notes"],
        },
    },
}


def _assistant_dict(msg) -> dict:
    """Re-encode an OpenAI-compat assistant message (with any tool calls) as a
    plain dict to append back into the conversation for the next turn."""
    d: dict = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d


class AgentRunner:
    def __init__(self, op: Op, runner: OKBenchRunner, *,
                 model: str | None = None, base_url: str = DEEPSEEK_BASE_URL,
                 api_key_env: str = "DEEPSEEK_API_KEY",
                 max_tokens: int = DEEPSEEK_MAX_TOKENS,
                 max_attempts: int = 8, target_speedup: float = 1.0,
                 variant_prefix: str = "gucheng", inject_skill: bool = False,
                 run_dir: Path | None = None, verbose: bool = True,
                 client=None):
        self.op = op
        self.runner = runner
        self.model = model or DEEPSEEK_MODEL
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.target_speedup = target_speedup
        self.variant_prefix = variant_prefix
        self.inject_skill = inject_skill
        self.run_dir = run_dir
        self.verbose = verbose
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError("openai not installed — `pip install openai`") from e
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"{api_key_env} not set in the environment")
            self.client = OpenAI(api_key=key, base_url=base_url)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[anvil-agent] {msg}", flush=True)

    def _messages(self) -> list[dict]:
        header_path = f"ops/{self.op.name}/interface.h"
        system = prompts.SYSTEM.format(
            header_path=header_path, entry_symbol=self.op.entry_symbol,
            atol=self.op.atol, rtol=self.op.rtol,
        )
        if self.inject_skill:
            system += "\n\n" + prompts.PTX_GEMM_SKILL
        user = prompts.build_user(self.op) + prompts.AGENT_TASK.format(budget=self.max_attempts)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def run(self) -> RunReport:
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"archiving run to {self.run_dir}")
        messages = self._messages()
        history: list[EvalResult] = []
        best: EvalResult | None = None
        n_bench = 0
        # +3 turns of slack so the model can reason/finalize around its bench calls.
        for _turn in range(self.max_attempts + 3):
            if n_bench >= self.max_attempts:
                break
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=messages, tools=[BENCH_TOOL], tool_choice="auto",
            )
            msg = resp.choices[0].message
            messages.append(_assistant_dict(msg))
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                self._log("model stopped calling bench_kernel — ending episode")
                break
            for tc in tool_calls:
                if n_bench >= self.max_attempts:
                    messages.append(_tool_result(tc.id, "Budget exhausted; no more bench_kernel calls."))
                    continue
                content = self._handle_bench(tc, n_bench + 1, history)
                if content is not None:                       # a real bench happened
                    n_bench += 1
                    best = _better(best, history[-1])
                    messages.append(_tool_result(tc.id, content))
                else:                                          # malformed call
                    messages.append(_tool_result(tc.id, "Call rejected: send a `bench_kernel` "
                                                          "tool call with a complete `kernel_cu` string."))
            if best and best.geomean_speedup and best.geomean_speedup >= self.target_speedup:
                self._log(f"target {self.target_speedup:.2f}x reached — stopping")
                break

        if best is not None:
            self._log(f"BEST: {best.summary()}")
        else:
            self._log("no correct kernel produced")
        self._finalize(best, history)
        return RunReport(best=best, history=history)

    def _handle_bench(self, tc, attempt: int, history: list[EvalResult]) -> str | None:
        if tc.function.name != "bench_kernel":
            return None
        try:
            args = json.loads(tc.function.arguments)
        except (ValueError, TypeError):
            return None
        kernel = args.get("kernel_cu")
        if not kernel:
            return None
        notes = args.get("notes", "")
        variant = f"{self.variant_prefix}_a{attempt:02d}"
        self._log(f"attempt {attempt}/{self.max_attempts} [{variant}]: benching…")
        result = self.runner.evaluate(Candidate(kernel_cu=kernel, notes=notes), variant)
        history.append(result)
        self._log(f"attempt {attempt} [{variant}]: {result.summary()}")
        self._record(attempt, variant, result)
        prev_best = _better(None, *history[:-1])
        return prompts.feedback_for_result(
            result, best_geomean=prev_best.geomean_speedup if prev_best else None)

    # --- archiving (mirrors orchestrator's format: attempt == iter) ----------

    def _record(self, attempt: int, variant: str, result: EvalResult) -> None:
        if not self.run_dir:
            return
        (self.run_dir / f"attempt{attempt:02d}_{variant}.cu").write_text(result.candidate.kernel_cu)
        row = {
            "attempt": attempt, "variant": variant, "stage": result.stage,
            "correct": result.correct, "geomean_speedup": result.geomean_speedup,
            "error": (result.error[:500] if result.error else None),
        }
        with (self.run_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _finalize(self, best: EvalResult | None, history: list[EvalResult]) -> None:
        if not self.run_dir:
            return
        if best is not None:
            (self.run_dir / "best.cu").write_text(best.candidate.kernel_cu)
        summary = {
            "op": self.op.name, "mode": "agent", "attempts": len(history),
            "best_geomean_speedup": (best.geomean_speedup if best else None),
            "best_notes": (best.candidate.notes if best else None),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _tool_result(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _better(current: EvalResult | None, *results: EvalResult) -> EvalResult | None:
    """Return whichever of current + results has the highest correct geomean."""
    best = current
    for r in results:
        if r is not None and r.correct and r.geomean_speedup is not None:
            if best is None or best.geomean_speedup is None or r.geomean_speedup > best.geomean_speedup:
                best = r
    return best
