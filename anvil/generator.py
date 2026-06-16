"""Kernel generators.

Providers:
- "deepseek" — DeepSeek via its OpenAI-compatible API (cheap; the default).
- "claude"   — Anthropic Claude, forced `emit_kernel` tool call.
- HumanGenerator — a fixed hand-written kernel for smoke tests (no LLM).

All LLM providers receive the same system+user prompt; only the transport and
the way the kernel.cu is extracted differ.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from .op import Op
from .candidate import Candidate, EvalResult
from . import prompts
from .baselines import SMOKE_KERNELS

# --- provider defaults ------------------------------------------------------

CLAUDE_MODEL = "claude-opus-4-8"
DEEPSEEK_MODEL = "deepseek-v4-pro"            # live strong model (old "deepseek-reasoner" alias is gone)
DEEPSEEK_MAX_TOKENS = 32768                    # reasoning models spend budget on thinking; leave room for the kernel too
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Appended for OpenAI-compatible providers that return free text (no tool schema).
_FENCED_CONTRACT = """

Output format (STRICT): return the complete kernel.cu as a single fenced code block:
```cuda
// ...the entire file...
```
After the code block, add exactly one line starting with `NOTES:` (one or two
sentences on what you changed and why). Output nothing else."""


# --- shared prompt building -------------------------------------------------

def _build_prompt(op: Op, history: list[EvalResult]) -> tuple[str, str]:
    header_path = f"ops/{op.name}/interface.h"
    system = prompts.SYSTEM.format(
        header_path=header_path, entry_symbol=op.entry_symbol,
        atol=op.atol, rtol=op.rtol,
    )
    user = prompts.build_user(op) + "\n" + prompts.build_feedback(history)
    return system, user


def _parse_fenced_kernel(text: str, *, finish_reason: str | None = None) -> tuple[str, str]:
    """Pull the kernel.cu and the NOTES line out of free text, tolerating sloppy output."""
    notes_m = re.search(r"NOTES:\s*(.+)", text)
    notes = notes_m.group(1).strip() if notes_m else ""

    # 1. properly-closed fenced blocks → take the largest
    blocks = re.findall(r"```(?:[A-Za-z0-9_+\-]*)\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip(), notes

    # 2. a single unclosed fence (output truncated before the closing ```)
    m = re.search(r"```(?:[A-Za-z0-9_+\-]*)\n(.*)\Z", text, re.DOTALL)
    if m and ("#include" in m.group(1) or "__global__" in m.group(1)):
        return m.group(1).strip(), notes

    # 3. no fence at all, but the whole reply looks like the file
    if "#include" in text or 'extern "C"' in text:
        return text.strip(), notes

    snippet = text[:400].replace("\n", "\\n")
    raise RuntimeError(
        f"model returned no usable kernel (finish_reason={finish_reason}); "
        f"first 400 chars: {snippet!r}"
    )


# --- generators -------------------------------------------------------------

class Generator(ABC):
    @abstractmethod
    def propose(self, op: Op, history: list[EvalResult]) -> Candidate: ...


class HumanGenerator(Generator):
    """Replays a fixed hand-written kernel (the smoke baseline). No LLM."""
    def __init__(self, kernel_cu: str | None = None):
        self._kernel = kernel_cu

    def propose(self, op: Op, history: list[EvalResult]) -> Candidate:
        src = self._kernel or SMOKE_KERNELS.get(op.name)
        if src is None:
            raise ValueError(f"no baseline kernel for op {op.name!r}")
        return Candidate(kernel_cu=src, notes="hand-written tiled baseline")


class OpenAICompatGenerator(Generator):
    """Any OpenAI-compatible chat API (DeepSeek by default). Parses a fenced block."""
    def __init__(self, *, model: str = DEEPSEEK_MODEL,
                 base_url: str = DEEPSEEK_BASE_URL,
                 api_key_env: str = "DEEPSEEK_API_KEY",
                 max_tokens: int = DEEPSEEK_MAX_TOKENS):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai not installed — `pip install openai`") from e
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} not set in the environment")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens

    def propose(self, op: Op, history: list[EvalResult]) -> Candidate:
        system, user = _build_prompt(op, history)
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system + _FENCED_CONTRACT},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        kernel_cu, notes = _parse_fenced_kernel(text, finish_reason=choice.finish_reason)
        return Candidate(kernel_cu=kernel_cu, notes=notes,
                         meta={"model": self.model, "finish_reason": choice.finish_reason})


class ClaudeGenerator(Generator):
    """Anthropic Claude with a forced emit_kernel tool call."""
    _EMIT_TOOL = {
        "name": "emit_kernel",
        "description": "Return one complete pure-CUDA kernel.cu and a one-line rationale.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_cu": {
                    "type": "string",
                    "description": "Complete contents of kernel.cu: includes the op header, "
                                   "defines the extern \"C\" entry symbol, compiles as-is.",
                },
                "notes": {"type": "string", "description": "One or two sentences: what you changed and why."},
            },
            "required": ["kernel_cu", "notes"],
            "additionalProperties": False,
        },
    }

    def __init__(self, model: str = CLAUDE_MODEL, effort: str = "high"):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic not installed — `pip install anthropic`") from e
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def propose(self, op: Op, history: list[EvalResult]) -> Candidate:
        system, user = _build_prompt(op, history)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=32000,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=system,
            tools=[self._EMIT_TOOL],
            tool_choice={"type": "tool", "name": "emit_kernel"},
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()

        tool_use = next((b for b in msg.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise RuntimeError(f"model did not call emit_kernel (stop={msg.stop_reason})")
        data = tool_use.input
        return Candidate(kernel_cu=data["kernel_cu"], notes=data.get("notes", ""),
                         meta={"model": self.model})


# Backwards-compat alias (older callers / tests used LLMGenerator for Claude).
LLMGenerator = ClaudeGenerator


def make_generator(provider: str = "deepseek", model: str | None = None) -> Generator:
    provider = provider.lower()
    if provider == "deepseek":
        return OpenAICompatGenerator(model=model or DEEPSEEK_MODEL)
    if provider == "claude":
        return ClaudeGenerator(model=model or CLAUDE_MODEL)
    raise ValueError(f"unknown provider {provider!r}; choose 'deepseek' or 'claude'")
