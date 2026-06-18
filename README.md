# anvil — the automated LLM kernel-writing loop

anvil is the **fully-automated** sibling of `forge`. An LLM writes a CUDA kernel
for an [OpenKernels](../OpenKernels说明.md) op; `okbench` judges it (compile +
correctness vs cuBLAS + timing); the verdict drives the next attempt — keeping the
best, archiving every attempt.

- **forge** = the *agent* (Claude Code) writes kernels by hand (semi-auto). → 0.92× cuBLAS.
- **anvil** = an *LLM* writes them in a loop, unattended (full auto).
- Same judge (`okbench`); anvil does **not** reinvent correctness/perf.

## Two modes (the only difference is whether the model has "hands")

| mode | CLI | the model… | loop driven by |
|---|---|---|---|
| **Route B** | `anvil run` | emits one kernel per turn; sees the last verdict as text | the outer python (`orchestrator.py`) |
| **Route-AVO-lite (agent)** | `anvil agent` | calls a `bench_kernel` tool, reads the real verdict, retries itself | the **model**, via tool-use (`agent.py`) |

```
Route B:    ops/<op>/ ─▶ Generator(LLM) ─▶ kernel.cu ─▶ OKBenchRunner.evaluate
                ▲                                            │
                └────── feedback (text) ◀── EvalResult ◀─────┘   python decides each step

agent:      ops/<op>/ ─▶ model ⇄ bench_kernel tool ⇄ OKBenchRunner.evaluate
                              (model reads the verdict & decides to retry itself)
```

Both share op loading, `okeval`/`okbench_runner`, skill injection (`--inject-skill`)
and the run-archive format, so an agent run stays directly comparable to a Route-B
ablation (see `EXPERIMENTS.md`).

## Generator (Route B, pluggable)
`make_generator(provider, model, inject_skill=…)`:
- `deepseek` (default) — OpenAI-compatible API, default model **`deepseek-v4-pro`**
  (reachable from the campus server; key in `secrets/keys.env`).
- `claude` — Anthropic API (forced `emit_kernel` tool).

`--inject-skill` prepends the distilled PTX GEMM recipe (`prompts.PTX_GEMM_SKILL`,
from forge v7/v8) to the system prompt — the A/B variable for the skill ablation.
Output is parsed tolerantly; `max_tokens=65536` so a reasoning model's long
chain-of-thought doesn't crowd out the kernel.

> The **agent** mode (`agent.py`, DeepSeek / OpenAI-compat function-calling) gives
> the model one tool — `bench_kernel(kernel_cu, notes)` — and lets it iterate on a
> `--max-attempts` budget. Claude agent loop is deferred to v2.

## Where it runs
On the **GPU server** as a git clone (`/nvme/share/gucheng/anvil`), in the venv
with torch (cu128/sm_120) + okbench, next to the OpenKernels repo. The LLM API must
be reachable from the server (DeepSeek is). **nvcc must be on PATH** — a login shell
(`bash -lc`) or an explicit `export PATH=/usr/local/cuda-13.0/bin:$PATH` +
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64`. Edit on the Mac → push → `git pull` on
the server.

## Run
```bash
source /nvme/share/gucheng/secrets/keys.env        # DEEPSEEK_API_KEY
cd /nvme/share/gucheng/anvil
export PATH=/usr/local/cuda-13.0/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64

# smoke: hand-written baseline through okbench, NO LLM (proves the pipeline)
.venv/bin/python -m anvil.cli smoke --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels --device 6

# Route B: one-shot generator loop
.venv/bin/python -m anvil.cli run   --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels \
    --device 6 --max-iters 8 [--inject-skill]

# agent: model drives a bench_kernel tool loop (Route-AVO-lite)
.venv/bin/python -m anvil.cli agent --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels \
    --device 6 --max-attempts 12 [--inject-skill]
#   long runs: launch detached so they survive disconnects —
#   setsid bash <driver>.sh </dev/null >run.log 2>&1 &
```
Defaults: `--hardware 5090 --platform sm120_rtx5090 --arch sm_120a --device 6 --author gucheng`.

## Run artifacts
Each run archives to **`runs*/<op>_<timestamp>/`** (gitignored, kept locally + on
the server as experiment evidence; the analysis lives in `EXPERIMENTS.md`):
- `iterNN_<variant>.cu` / `attemptNN_<variant>.cu` — every kernel tried
- `results.jsonl` — per-step stage / correct / geomean / error
- `best.cu` + `summary.json`

(The kernel is also deployed into `OpenKernels/submissions/...` because okbench
requires it there; that copy is transient — the run archive is the record.)

## Files
```
anvil/op.py             load ops/<op>/ (interface.h, reference, shapes, tolerance)
anvil/candidate.py      Candidate + EvalResult (parse okbench JSON)
anvil/generator.py      make_generator: OpenAICompat (DeepSeek) / Claude / Human
anvil/agent.py          Route-AVO-lite: model-driven bench_kernel tool loop
anvil/prompts.py        system prompt + feedback formatting + PTX_GEMM_SKILL
anvil/okeval.py         deploy→okbench→parse (forge has its own vendored copy)
anvil/okbench_runner.py thin adapter: okeval.evaluate → EvalResult
anvil/orchestrator.py   the Route-B loop; archives to run_dir
anvil/baselines.py      hand-written correct kernels for the smoke path
anvil/cli.py            smoke / run / agent
tests/test_smoke.py     offline plumbing checks (no GPU/LLM): parsers, trim_error, agent loop
```

## Status (2026-06-18)
Runs end-to-end on RTX 5090 in both modes.
- **Feedback fixed:** `okeval.trim_error` now surfaces the real nvcc error first
  (unwraps okbench's `RuntimeError(json)`; always hoists `error:` lines) instead of
  a python traceback — the model can finally see what to fix.
- **EXP-001 (Route B):** injecting the PTX skill took DeepSeek 0.03×→0.58× (n=1).
- **EXP-002 (agent):** the agent reaches 0.78–0.88× autonomously; +skill median
  0.746× vs no-skill 0.113×, and roughly halves compile-fail thrashing. Strong-
  suggestive at n=3 (each arm has one dud). See `EXPERIMENTS.md`.
