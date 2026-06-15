"""Anvil — an LLM agent that writes high-performance CUDA kernels.

It does NOT judge kernels itself. Correctness and performance are owned by the
OpenKernels Foundry (`okbench`); Anvil only generates kernel.cu and reads back
okbench's verdict to rewrite a faster version.

    Op (from ops/<op>/) ─▶ LLMGenerator ─▶ Candidate(kernel.cu)
            ▲                                     │
            │                          OKBenchRunner.evaluate
            │                          (writes submission, runs okbench)
            └──────── history ◀── EvalResult (per-shape correct + speedup)
"""

__version__ = "0.1.0"
