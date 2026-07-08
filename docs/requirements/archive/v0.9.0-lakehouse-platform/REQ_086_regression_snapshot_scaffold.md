# REQ_086: Regression Snapshot Scaffold

**Status:** Active
**Priority:** High
**Branch:** feature/req-086-regression-snapshot-scaffold
**Attribution:** Engineering Claude

---

## Problem Statement

The upcoming ActivationBundle refactor (REQ_087) and 2L MLP architecture support (REQ_088/089)
will touch the analysis pipeline's core model interface. Before that work begins, we need a
way to verify that the refactored code produces numerically identical artifacts to the current
implementation — and to do so without any risk of contaminating the existing `results/` directory.

The scaffold consists of two things: a fixed reference set of artifact checksums from the current
codebase, and a comparison script that re-runs analysis for those variants into a secondary
directory and validates the outputs. If the comparison passes, the refactor is validated. If it
fails, the original `results/` is untouched.

---

## Conditions of Satisfaction

### Reference Set
- [ ] Four reference variants selected and documented:
  - `p113/seed999/dseed598` — canon model, 95 checkpoints
  - `p109/seed485/dseed598` — fast clean grokker, 131 checkpoints
  - `p101/seed485/dseed42` — late grokker, 196 checkpoints (largest checkpoint count)
  - `p59/seed485/dseed999` — no_second_descent (most degraded classification), 95 checkpoints
- [ ] For each reference variant, SHA-256 checksums of all per-epoch and cross-epoch `.npz`
  artifacts are computed and saved to `regression/reference_checksums.json`
- [ ] Checksum file includes: variant_id, analyzer_name, artifact_path (relative), sha256, n_arrays,
  and total_bytes — enough to diagnose where a mismatch occurred

### Comparison Script
- [ ] `scripts/run_regression_check.py` re-runs the full analysis pipeline for all four reference
  variants into `results_regression/` (or a path override via `--output-dir`)
- [ ] Script compares all output `.npz` files against `regression/reference_checksums.json`
- [ ] On mismatch: reports which variant, which analyzer, which file diverged — then exits nonzero
- [ ] On full match: prints a summary and exits zero
- [ ] Script accepts `--variants` flag to run a subset (for faster iteration during the refactor)

### Safety
- [ ] Script never writes to `results/` — output dir is always separate
- [ ] Script is idempotent: re-running it into the same output dir is safe (overwrites cleanly)
- [ ] `results_regression/` is listed in `.gitignore`

---

## Constraints

**Must:**
- Use the existing `AnalysisPipeline` and family machinery — not a reimplementation
- Checksums must be file-level (whole `.npz`), not array-level, for simplicity

**Must not:**
- Modify any existing variant results or artifacts
- Require GPU (CPU analysis is sufficient for regression purposes)

**Flexible:**
- Whether the script supports parallel variant execution in v1
- Exact format of the mismatch report beyond the required fields

---

## Architecture Notes

**Checksum generation (one-time):**
Walk `results/modulo_addition_1layer/{variant_id}/artifacts/` for each reference variant.
Hash each `.npz` file with `hashlib.sha256`. Save to `regression/reference_checksums.json`.

**Re-run path:**
`AnalysisPipeline.run(variant, config)` already accepts a variant object. The variant's
`variant_dir` determines where artifacts are written. To redirect output, either:
- Construct a variant pointing at `results_regression/{variant_id}/` instead of `results/`, or
- Pass an `output_root` override if the pipeline supports it (check; add minimally if not)

**Key insight:** The pipeline is already deterministic given the same checkpoints and the same
analyzer code. Checksums should match exactly on the same hardware. If they don't after the
refactor, something changed in the computation.

**MISCOPE_PROJECT_ROOT:** The env var redirects the results root. The script can set this to
`results_regression/` before constructing the family/variant objects — this may be the
zero-infrastructure path. (`TDW_PROJECT_ROOT` is still accepted as a legacy alias.)

---

## Notes

- This scaffold intentionally has no dashboard component — it is a developer tool only.
- The four reference variants were chosen to cover: healthy fast grokker, canon (healthy, diffuse
  specialization), late grokker with dense checkpoints, and failed/no-second-descent. Together
  they exercise the full range of pipeline behavior.
- Once REQ_087 (ActivationBundle) passes regression, the scaffold remains in place for REQ_088/089.
  It should not be retired until the 2L MLP work is complete and validated.
- If `MISCOPE_PROJECT_ROOT` redirection works cleanly, the script may be as short as 30-40 lines.
