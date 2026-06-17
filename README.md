# anvil — the automated LLM kernel-writing loop

anvil is the **fully-automated** sibling of `forge`. An LLM API **writes** a CUDA
kernel for an [OpenKernels](../OpenKernels说明.md) op; `okbench` judges it
(compile + correctness vs cuBLAS + timing); anvil feeds the verdict back and the
LLM rewrites — keeping the best, archiving every attempt.

- **forge** = the *agent* (Claude Code) writes kernels by hand (semi-auto). → 0.92× cuBLAS.
- **anvil** = an *LLM API* writes them in a loop, unattended (full auto).
- Same judge (`okbench`); anvil does **not** reinvent correctness/perf.

```
ops/<op>/ ─▶ Generator (LLM API) ─▶ Candidate(kernel.cu) ─▶ OKBenchRunner.evaluate
    ▲                                                          │  (submission → okbench)
    └─────────────── history (feedback) ◀── EvalResult (per-shape correct + speedup)
```

## Generator (pluggable)
`make_generator(provider, model)`:
- `deepseek` (default) — OpenAI-compatible API, default model **`deepseek-v4-pro`**
  (reachable from the campus server; key in `secrets/keys.env`).
- `claude` — Anthropic API.

Output is parsed tolerantly (fenced block or whole reply); `max_tokens=65536` so a
reasoning model's long chain-of-thought doesn't crowd out the kernel.

## Where it runs
On the **GPU server** as a git clone (`/nvme/share/gucheng/anvil`), in the venv
with torch (cu128/sm_120) + okbench, next to the OpenKernels repo. The generator
API must be reachable from the server (DeepSeek is; OpenAI/Gemini are blocked).
Edit on the Mac → push → `git pull` on the server.

## Run
```bash
source /nvme/share/gucheng/secrets/keys.env        # DEEPSEEK_API_KEY
cd /nvme/share/gucheng/anvil

# smoke: hand-written baseline through okbench, NO LLM (proves the pipeline)
.venv/bin/python -m anvil.cli smoke --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels --device 0

# run: full auto loop (default provider deepseek / model deepseek-v4-pro)
.venv/bin/python -m anvil.cli run --op gemm_bf16_nt --repo /nvme/share/gucheng/OpenKernels \
    --device 0 --max-iters 8
#   long runs: launch detached so they survive disconnects —
#   setsid ... > run.log 2>&1 </dev/null &
```
Defaults: `--hardware 5090 --platform sm120_rtx5090 --arch sm_120a --author gucheng`.

## Run artifacts
Each run archives to **`runs/<op>_<timestamp>/`** (gitignored, kept on the server):
- `iterNN_<variant>.cu` — every kernel tried
- `results.jsonl` — per-iter stage / correct / geomean / error
- `best.cu` + `summary.json`

(The kernel is also deployed into `OpenKernels/submissions/...` because okbench
requires it there; that copy is transient — the run archive is the record.)

## Files
```
anvil/op.py             load ops/<op>/ (interface.h, reference, shapes, tolerance)
anvil/candidate.py      Candidate + EvalResult (parse okbench JSON)
anvil/generator.py      make_generator: OpenAICompat (DeepSeek) / Claude / Human
anvil/prompts.py        system prompt + feedback formatting
anvil/okeval.py         deploy→okbench→parse (forge has its own vendored copy)
anvil/okbench_runner.py thin adapter: okeval.evaluate → EvalResult
anvil/orchestrator.py   the loop; archives to run_dir
anvil/baselines.py      hand-written correct kernels for the smoke path
anvil/cli.py
tests/test_smoke.py     offline plumbing checks (no GPU/LLM)
```

## Status (2026-06)
Runs end-to-end on RTX 5090. `deepseek-v4-pro` autonomously produced a *correct*
kernel (~0.19× cuBLAS, basic wmma). The gap vs forge's hand-written 0.92× is real
but expected — the levers are generator quality, iteration budget, and feeding the
forge `wiki/` + `skills/` knowledge into the prompt (next).
