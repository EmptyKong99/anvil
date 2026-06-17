# anvil — code walkthrough

How the package fits together, module by module, plus the known rough edges.
(See `README.md` for what anvil *is* and how to run it.)

## Data flow (one iteration)
```
load_op(repo, op)              op.py        → Op (ABI text, reference, shapes, tol)
        │
generator.propose(op, history) generator.py → Candidate(kernel_cu, notes)
        │   (prompt = prompts.SYSTEM + build_user(op) + build_feedback(history))
        ▼
runner.evaluate(candidate, variant) okbench_runner.py
        │   write submission → okbench validate → okbench bench-<op> → JSON
        ▼
EvalResult (candidate.py)  ←─ from_okbench(json): per-shape correct + speedup
        │
orchestrator.py: append to history, keep best, archive to run_dir, loop
```
The **generator** is the only swappable brain; **okbench** is the judge (not
reinvented); the **orchestrator** is the loop.

## Modules

### `op.py` — the problem
`load_op(repo, name)` reads `ops/<op>/` from the OpenKernels repo: `op.yaml`
(entry_symbol, description), `interface.h` (the C ABI text), `reference.py`
(ground truth), `shapes.yaml` (the suite), `tolerance.yaml` (atol/rtol). Returns a
frozen-ish `Op`. `Op.bench_cmd` delegates to `okeval.bench_cmd` (the op →
okbench-subcommand map lives once, in `okeval`). We read specs, never redefine them.

### `okeval.py` — the canonical eval (shared with forge)
The one place that does **deploy → okbench → parse**, so the logic can't drift
between anvil and forge. Holds `OKBENCH_BENCH_CMD` (single source), `evaluate()`
(write submission → `okbench validate` → `okbench bench-<op>` → parsed JSON, as a
plain `EvalOutcome`), and `format_summary()`. Low layer: stdlib + yaml only, no
dependency on `candidate`/`op`. Has a `__main__` CLI (`python -m anvil.okeval
--repo … --op … --variant … --src … --out …`) that **forge's `tools/bench.sh`
shells out to** (it already runs on anvil's venv; PYTHONPATH adds the anvil repo).

### `candidate.py` — the data passed around
- `Candidate`: the proposal — `kernel_cu` (str), `notes`, `meta`.
- `EvalResult`: the verdict — `stage` (validate/compile/bench), `correct`,
  `per_shape`, `geomean_speedup`, `error`, plus `from_okbench(json)` which parses
  okbench's result schema (`shapes[].correct`, `pure_over_reference`→speedup) and
  `summary()` for one-line logging.

### `generator.py` — the brains (swappable)
- `Generator` ABC with `propose(op, history) -> Candidate`.
- `HumanGenerator` — replays a fixed baseline kernel (smoke path, no LLM).
- `OpenAICompatGenerator` — DeepSeek (default `deepseek-v4-pro`) via the OpenAI
  SDK; parses the kernel out of the reply (`_parse_fenced_kernel`, tolerant of
  truncation / no fence). `max_tokens=65536` so reasoning doesn't crowd the kernel.
- `ClaudeGenerator` — Anthropic, forced `emit_kernel` tool.
- `make_generator(provider, model)` — factory; `_build_prompt` assembles
  system+user from `prompts.py`.

### `prompts.py` — what the model sees
`SYSTEM` (role, hardware, hard rules, perf hints), `build_user(op)` (op +
shapes + full interface.h + full reference.py), `build_feedback(history)` (last
attempt's stage/error/per-shape speedups + best-so-far).

### `okbench_runner.py` — the bridge to the judge
`evaluate(candidate, variant)` is now a thin adapter: it calls `okeval.evaluate`
(deploy + validate + bench) and maps the `EvalOutcome` onto an `EvalResult`
(`from_okbench` on success, or stage/error on failure). All the okbench mechanics
live in `okeval`.
> ⚠️ Gotcha (fixed): okbench's `--output` is resolved against its cwd
> (OpenKernels). `out_json` must be **absolute** (okeval `.resolve()`s it), or
> okbench writes under `OpenKernels/…` while anvil looks under `anvil/…` and every
> working kernel reads as "compile failed". `cli.py` also `.resolve()`s the run dir.

### `orchestrator.py` — the loop
`run()`: for each iter — propose → evaluate → log → `_record` (archive) → track
best → stop at `target_speedup`. `_record`/`_finalize` write the run archive:
`iterNN_<variant>.cu`, `results.jsonl`, `best.cu`, `summary.json` under `run_dir`.
`results.jsonl` is the real per-run record (run.log is just redundant console echo).

### `cli.py` — entrypoint
`smoke` (HumanGenerator, prove the pipeline) and `run` (full loop). Builds an
**absolute** timestamped `run_dir = <out-dir>/<op>_<ts>/`, wires generator+runner+
orchestrator. Flags: `--provider/--model/--max-iters/--target-speedup/--out-dir`
+ common `--op/--repo/--hardware/--platform/--arch/--device/--author/--suite`.

### `baselines.py` — smoke data
A hand-written correct (slow) kernel string for `smoke`. Really test data living
in the package.

## Known rough edges (optimization points)
1. ~~Eval duplicated with forge.~~ **Fixed (Tier 2):** deploy→okbench→parse lives
   once in `okeval.py`; `okbench_runner.py` and forge's `tools/bench.sh` both call it.
2. ~~op→okbench-subcommand mapping duplicated.~~ **Fixed:** single
   `OKBENCH_BENCH_CMD` in `okeval`; `op.py` imports it.
3. **Prompt is hard-coded / not composable.** Injecting the forge `wiki/`+`skills/`
   knowledge needs a clean "extra context" hook in `_build_prompt`.
4. **Config sprawl.** hardware/platform/arch/device/author/suite are threaded by
   hand cli→runner; group into a `Target`/`RunConfig` dataclass.
5. **`baselines.py` is data in the package** — could move to `tests/`/a data dir.
6. **Tests are smoke-only** (offline plumbing); no unit tests for the parsers/loop.
7. **Feedback truncates errors** to ~2–3K chars — the model may not see the real
   nvcc error; consider smarter (head+tail) truncation or full errors.
