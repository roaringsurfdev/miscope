# REQ_116: Regression Folder Cleanup

**Status:** Completed (by user)
**Priority:** Low
**Branch:** TBD
**Dependencies:** None. Follows REQ_115 (monorepo layout) — this finishes the
root-level cleanup that REQ_115 deliberately scoped out.
**Attribution:** Engineering Claude (drafted from collaborative discussion 2026-05-07)

---

## Problem Statement

After REQ_115, the repo root is mostly clean (`packages/`, `apps/`, `scripts/`,
`docs/`, `results/`, `tests/`), but five directories related to regression
testing still sit at the root:

```
regression/                  # committed reference checksums (2 JSON files)
results_regression/          # gitignored generated regression artifacts
results_regression_fresh/    # gitignored
results_regression_req112/   # gitignored
results_regression_req114/   # gitignored
```

These are visually noisy on the GitHub project landing and don't communicate
what they're for at a glance. The committed reference data is *test* reference
data; the generated outputs are *results*. Both have natural homes in the
existing layout.

---

## Goal Layout

```
tests/regression/                    # committed reference checksums
  reference_checksums.json
  reference_checksums_req112.json
results/regressions/regression/                  # gitignored generated results/regressions/regression_fresh/            # gitignored
results/regressions/regression_req112/           # gitignored
results/regressions/regression_req114/           # gitignored
```

`tests/regression/` makes the role visible: this is reference data the test
suite compares against. `results/regressions/regression*/` keeps all generated artifacts
under one tree (`results/regressions/`).

---

## Conditions of Satisfaction

- [ ] `regression/` moved to `tests/regression/` via `git mv` so blame/log
  walks through the move.
- [ ] `results_regression/`, `results_regression_fresh/`,
  `results_regression_req112/`, `results_regression_req114/` moved under
  `results/` (plain `mv` — they are gitignored).
- [ ] `.gitignore` pattern `/results_regression*` updated to
  `/results/regressions/regression*`.
- [ ] `scripts/run_regression_check.py` and
  `scripts/generate_regression_checksums.py` default paths updated:
  - `--output-dir` default: `results_regression/` → `results/regressions/regression/`
  - `--checksums` default: `regression/reference_checksums.json` →
    `tests/regression/reference_checksums.json`
  - The argument-driven nature of these scripts is preserved — only the
    defaults change.
- [ ] Repo root no longer contains `regression/` or any `results_regression*`
  directory.
- [ ] Regression-check workflow still runs end-to-end against the new defaults
  (verified by invoking `run_regression_check.py` with no args, or however the
  user normally exercises it).

---

## Constraints

**Must:**
- Use `git mv` for the committed `regression/` directory. The generated
  directories are gitignored — plain `mv` is fine for them.
- Preserve the argument-passing interface on both scripts. The defaults are
  what change; explicit `--output-dir` / `--checksums` invocations continue to
  work.

**Must avoid:**
- Renaming or restructuring the JSON checksum files themselves. This is a move,
  not a content change.
- Touching the regression-check logic (analyzer set, comparison rules,
  exclusion list). Out of scope.

---

## Notes

- Branch name suggestion: `feature/regression-cleanup`.
- trivial commit.
