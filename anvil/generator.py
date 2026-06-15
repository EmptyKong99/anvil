"""Kernel generators: LLM (Claude) and a hand-written one for smoke tests."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .op import Op
from .candidate import Candidate, EvalResult
from . import prompts
from .baselines import SMOKE_KERNELS

try:
    import anthropic
except ImportError:
    anthropic = None

MODEL = "claude-opus-4-8"

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


class LLMGenerator(Generator):
    def __init__(self, model: str = MODEL, effort: str = "high"):
        if anthropic is None:
            raise RuntimeError("anthropic not installed — `pip install anthropic`")
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def propose(self, op: Op, history: list[EvalResult]) -> Candidate:
        header_path = f"ops/{op.name}/interface.h"
        system = prompts.SYSTEM.format(
            header_path=header_path, entry_symbol=op.entry_symbol,
            atol=op.atol, rtol=op.rtol,
        )
        user = prompts.build_user(op) + "\n" + prompts.build_feedback(history)

        with self.client.messages.stream(
            model=self.model,
            max_tokens=32000,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=system,
            tools=[_EMIT_TOOL],
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
