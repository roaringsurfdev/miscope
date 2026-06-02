# REQ_134: Regression Harness → Canonical Analyzer-Set Pattern

**Status:** Implemented — *all CoS met on `feature/REQ_134_regression_harness_canonical_pattern` (2026-06-01). The restored net immediately surfaced four analyzers whose reference artifacts had drifted from current code; all were refreshed and the checker is green across the three pinned variants. See "Validation findings" below. Awaiting merge approval (held per 2026-06-01 decision).*
**Priority:** Low-Medium — the v1.0.0 byte-regression safety net is currently inoperable.
**Branch:** TBD
**Sequencing:** Lands **before REQ_134's sibling REQ_133** (decision 2026-06-01). This work binds to
the stable `plan_analysis` interface REQ_133 preserves and restores the byte-regression net that
becomes REQ_133's oracle for an ordering-only refactor. **Scope this requirement to incremental /
existing-artifact regression only** — a clean-from-scratch regression mode trips REQ_133's unordered
cross-epoch→cross-epoch edges (see REQ_133 Spike finding 2) and therefore depends on REQ_133.
**Dependencies:**
- REQ_132 (collapsed the register triad; this harness now calls `register(...)`).

**Attribution:** Engineering Claude (stubbed 2026-05-30 from a REQ_132 validation run).

---

## Problem

`scripts/run_regression_check.py` hand-curates the analyzer set it registers on
the pipeline. That curated list has fallen behind the analyzer dependency graph:
it omits upstreams that current Specs declare as dependencies, so a real run
fails midway.

Observed 2026-05-30 (variant `p109_seed485_dseed598`):
- `fourier_frequency_quality` depends on `neuron_grouping` (not registered) → skipped.
- `neuron_dynamics` requires `activation_basis_projection` (not registered) →
  cross-epoch blocked-by `RuntimeError`, aborting the run.

This is **pre-existing** (predates REQ_132 — the omitted dependencies merged with
REQ_126/127 and REQ_130) and reproduces identically on `develop`. REQ_132 only
made the symptom visible by prompting a harness run.

## Direction

The dashboard and `scripts/run_analysis.py` already use the canonical pattern and
*cannot* fall behind the dependency graph:

```python
specs = AnalyzerRegistry.list_for_family(variant.family)
plan  = plan_analysis(variant, specs, force=force)
pipeline.run(plan=plan)
```

`run_regression_check.py` should adopt the same pattern rather than enumerating
analyzers by hand.

### Findings (scoping pass, 2026-06-01)

Reading the three regression scripts surfaced more drift than the original symptom.
The harness is broken in **three independent ways**, and the analyzer-set drift is
only one of them:

1. **Path drift (immediate failure).** [generate_regression_checksums.py:88](../../../scripts/generate_regression_checksums.py#L88)
   writes to `tests/regression/reference_checksums.json`; [run_regression_check.py:146](../../../scripts/run_regression_check.py#L146)
   defaults to `regression/reference_checksums.json` — **which does not exist.** A
   default invocation of the checker exits immediately ("Checksums file not found")
   before any analyzer runs.

2. **Selection drift (the named symptom).** [run_analysis_regression.py:81-90](../../../scripts/run_analysis_regression.py#L81-L90)
   *already* uses the canonical `list_for_family → plan_analysis → run(plan=)` pattern
   (since REQ_120). Only [run_regression_check.py:66-85](../../../scripts/run_regression_check.py#L66-L85)
   hand-curates a 14-analyzer `register(...)` list, and it omits declared upstreams
   (`activation_basis_projection`, `neuron_grouping`) → the run aborts mid-way on a
   `blocked_by` `RuntimeError`.

3. **Exclude-set drift.** All **three** scripts carry *different* exclude sets:
   - generator: `DEPRECATED(8) | {landscape_flatness, fourier_nucleation}`
   - regen: `DEPRECATED(6) | {landscape_flatness, fourier_nucleation}` (omits the two
     REQ_102 names)
   - checker: `{landscape_flatness, coarseness, gradient_site, fourier_nucleation,
     dominant_frequencies, neuron_freq_norm}` (a fourth, disjoint set)

   So the set that is *checksummed*, the set that is *recomputed*, and the set the
   *EXTRA-artifact filter* ignores are all different. This is the root cause — three
   copies of a list that must agree, and don't.

**Clean-from-scratch gotcha (the REQ_133 boundary).** [run_regression_check.py:210-214](../../../scripts/run_regression_check.py#L210-L214)
symlinks checkpoints/config into a fresh `results_regression/` tree but **not**
`artifacts/` — so it analyzes into an *empty* artifacts dir. That is precisely the
clean-from-scratch scenario that trips REQ_133's unordered cross-epoch→cross-epoch
edges (Spike finding 2): `intragroup_manifold` / `transient_frequency` /
`activation_dmd` would come out `blocked_by` their cross-epoch upstreams. Today the
run aborts on finding 2 (missing per-epoch upstream) *before* reaching those edges, so
the cross→cross block has never been observed — but adopting the canonical set (which
registers the missing upstreams) **would expose it.** Hence the scope below.

## Conditions of Satisfaction

1. **Single source of truth for the analyzer set.** `run_regression_check.py` selects
   its analyzer set via `AnalyzerRegistry.list_for_family(variant.family)` filtered by a
   shared exclude constant, and runs through `plan_analysis(...)` + `run(plan=plan)` — no
   hand-enumerated `register(...)` list. (Finding 2.)
2. **One shared exclude set.** The three scripts' divergent exclude sets are reconciled
   into a single shared constant (imported, not copy-pasted). The set that is
   checksummed, the set that is recomputed, and the EXTRA-artifact filter all reference
   it. A test fails if a script defines its own. (Finding 3.)
3. **One shared checksums path.** Generator and checker resolve the same
   `reference_checksums.json` path by default. (Finding 1.)
4. **No missing-upstream abort.** Because the set comes from `list_for_family`, every
   declared `ArtifactInput` upstream is present; the run no longer aborts on `blocked_by`
   for `activation_basis_projection` / `neuron_grouping`.
5. **Incremental / existing-artifact scope.** The check runs against variants whose
   declared upstream artifacts are present on disk before recompute (so cross-epoch→
   cross-epoch dependencies resolve via on-disk state, not single-pass ordering). The
   empty-`artifacts/` clean-from-scratch path is **explicitly out of scope** and deferred
   to REQ_133. A clean-from-scratch regression mode is not introduced here.
6. **Checksums regenerated.** `reference_checksums.json` is regenerated to cover exactly
   the new set; stale REQ_102 entries (`dominant_frequencies`, `neuron_freq_norm`) and
   any other no-longer-covered analyzers are removed. The committed file matches a fresh
   regen on `develop`.
7. **Green end-to-end.** `run_regression_check.py` exits 0 across the reference variants
   (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598) with no MISSING / MISMATCH /
   EXTRA — restoring the v1.0.0 byte-regression safety net.

## Validation findings (2026-06-01)

Implemented CoS 1–6 behind one shared module
[scripts/regression_common.py](../../../scripts/regression_common.py); reconciled all three
scripts + 9 new tests in
[tests/integration/test_regression_harness_shared.py](../../../tests/integration/test_regression_harness_shared.py).
Cheap checks all pass: a real p113 plan over the canonical 21-analyzer set produces every
cross-epoch→cross-epoch item (`activation_dmd` / `intragroup_manifold` / `transient_frequency`)
with **zero `blocked_by`** (CoS 4), and a fresh checksum regen is byte-identical to the committed
`reference_checksums.json` (CoS 6).

**Net works; four stale references surfaced and refreshed.** A forced recompute reproduced the
overwhelming majority of analyzers byte-identically and flagged exactly the analyzers whose stored
artifacts had drifted from current `develop` code — **not** nondeterminism or a harness bug:

- `repr_geometry` and `fourier_frequency_quality` — output changed in their REQs. (REQ_126 PR 3
  `a8480a3` "defusion" moved the `*_fourier_alignment` keys out of `repr_geometry` into the separate
  `centroid_fourier_alignment` analyzer, which matches byte-for-byte; overlapping keys agree at
  `rtol=1e-3`. The stale references predated that rewrite — dated 2026-05-06 vs the 2026-05-27 code.)
- `neuron_dynamics` and `intragroup_manifold` — also drifted (single-file cross-epoch outputs).

All four were recreated with current code for the three pinned reference variants and the checksums
regenerated. Verified self-consistent: `run_regression_check.py --no-recompute` is **green across all
three** (p113 2775 + p109 2775 + p101 3886 = 9436 artifacts; no MISSING/EXTRA/MISMATCH). Cross-checked
on a sample (`p113 neuron_dynamics`) that `data/` disk == regenerated checksums == a fresh recompute
all agree, with only the pre-refresh committed baseline as the outlier.

**CoS 7 met via the full-cycle refresh** (user decision 2026-06-01). Two follow-on notes:

- A clean *from-scratch* recompute of all 21 (empty `artifacts/`) is **not** the validated path here —
  it trips REQ_133's unordered cross→cross edges and is out of scope (see CoS 5). The validated path
  is the `--no-recompute` integrity check plus the per-analyzer recompute the refresh exercised. A full
  forced recompute via the canonical overlay remains available (and optional) for a determinism re-proof.
- `neuron_dynamics` / `intragroup_manifold` are single-file cross-epoch outputs — the classic
  candidates for float-reduction nondeterminism. If a *recompute* check ever flags either **without** a
  code change behind it, exclude it like `landscape_flatness` rather than treating it as a harness bug.

## Notes

- The byte-regression safety net for v1.0.0 is restored and green via `--no-recompute`. The harness is
  pinned to the three reference variants (`generate_regression_checksums.py` `REFERENCE_VARIANTS`);
  broader coverage waits on the planned retrain-with-denser-checkpointing + full re-analysis pass.
- `--no-recompute` compares `data_root` artifacts directly against the checksums (fast integrity
  check); it cannot catch a code-introduced regression — only a true recompute (the default) does.
- Reference variants and their roles live in
  [generate_regression_checksums.py:23-29](../../../scripts/generate_regression_checksums.py#L23-L29)
  (p59/s485/ds999 is currently commented out).
- The `_req112` checksums snapshot was retired 2026-06-01 (well past REQ_112; no code
  referenced it — only historical mentions in staging REQ_113/114/116 docs remain).
- Folding `run_analysis_regression.py` and the checker onto a shared selection helper is
  the cleanest expression of CoS 1-2, but the two scripts play different roles (regen vs
  check) and need not merge into one file.
