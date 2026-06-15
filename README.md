# Anvil

An LLM agent that **writes** high-performance CUDA kernels for the
[OpenKernels Foundry](../OpenKernels说明.md). Anvil is only the *generator*; it
does **not** judge kernels — correctness and performance are owned by
OpenKernels' `okbench`. Anvil generates a `kernel.cu`, hands it to okbench, reads
the verdict, and rewrites a faster version.

```
ops/<op>/  ──▶  LLMGenerator (Claude)  ──▶  Candidate (kernel.cu)
     ▲                                            │
     │                                   OKBenchRunner.evaluate
     │                          (write submission → okbench validate + bench)
     └──────────── history ◀── EvalResult (per-shape correct + speedup vs ref)
```

## Design

- **Generator (LLM)** writes a complete pure-CUDA `kernel.cu` implementing the
  op's stable C ABI (e.g. `openkernels_launch_gemm_bf16_nt` from
  `ops/gemm_bf16_nt/interface.h`). It reads okbench's compiler errors / per-shape
  correctness / speedups to improve.
- **Judge = okbench** (not reinvented): `OKBenchRunner` writes the submission,
  runs `okbench validate` then `okbench bench-<op>`, and parses the result JSON
  (`shapes[].correct`, `pure_over_reference`, geomean speedup).
- We target the **stable-ABI path** first (LLM writes any kernel behind the C
  ABI; flexible, torch reference). The kernel-body ranking path (which needs a
  `tma_abc`-style metadata launch block) comes later.

## Where it runs

Anvil runs **on the GPU server**, in the venv that has `torch` (cu128, with
`sm_120`) + `triton` + `okbench` installed, next to the OpenKernels repo. The
LLM calls go out from there (needs `ANTHROPIC_API_KEY` + internet). The Mac is
just for editing — rsync the `anvil/` package over to run it.

## Files

```
anvil/
  op.py            load ops/<op>/ (interface.h, reference, shapes, tolerance)
  candidate.py     Candidate (kernel.cu) + EvalResult (parses okbench JSON)
  baselines.py     hand-written correct kernels for the smoke path
  okbench_runner.py write submission → okbench validate + bench → EvalResult
  generator.py     LLMGenerator (Claude, forced emit_kernel tool) + HumanGenerator
  prompts.py       system prompt + feedback formatting
  orchestrator.py  generate → evaluate → feed back → repeat, keep best
  cli.py
tests/test_smoke.py  offline plumbing checks (no GPU/LLM)
```

## Run

On the **server** (venv active, repo at `/nvme/share/gucheng/OpenKernels`):

```bash
pip install anthropic pyyaml          # into the venv that has torch+triton+okbench
export ANTHROPIC_API_KEY=...

# 1) prove the pipeline with a hand-written kernel (NO LLM)
python -m anvil.cli smoke --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels --device 6

# 2) full agent loop
python -m anvil.cli run --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels \
    --device 6 --max-iters 8 --target-speedup 0.9
```

Defaults: `--hardware 5090 --platform sm120_rtx5090 --arch sm_120a --author gucheng`.

On the **Mac** you can only run the offline checks: `pytest tests/`.

## Status

Generator + okbench bridge + loop are written. Not yet run on a GPU — the first
real check is the `smoke` command above (compiles the baseline kernel for
`sm_120a` and runs it through okbench on card 6).
