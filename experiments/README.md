# experiments/ — exact launch recipes (reproducibility)

Driver scripts used to launch each experiment on the 5090 server: CUDA env
(`PATH`/`LD_LIBRARY_PATH` — non-login shells lack nvcc), per-arm GPU + `--author`
isolation, n reps, both arms in parallel, detached via `setsid`. Run artifacts
(gitignored) land in `runs_expNNN/{base,skill}/<ts>/`; analysis in `EXPERIMENTS.md`.

- `exp002_driver.sh` — EXP-002 (agent ablation, n=3, max-attempts 12)
- `exp003_driver.sh` — EXP-003 (fixed agent + n=5)
