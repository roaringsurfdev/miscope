# REQ_102: Analyzer Deprecation (Retire Stale Paths)

**Status:** Completed — all nine retirement candidates removed; merged to `develop` 2026-05-29
**Priority:** Medium — close-out track for the Atlas-driven phase 1 consolidations. Bounded by prerequisite REQs completing.
**Branch:** TBD
**Dependencies:**
- REQ_106 (defines the layering principles whose violation is one of the deprecation criteria).
- REQ_109 (primitive layer — *staging*; the new analyzers consume it).
- REQ_111 (Universal Core pure renames — *staging*; gates the `effective_dimensionality` retirement via parity validation. The `parameter_trajectory_pca` retirement is degenerate — the rename was already aligned at the registered-name level, so REQ_111's mechanical port closed the file/class gap with no separate analyzer left to retire).
- REQ_126 (Family basis projection consolidation — gates the `coarseness` retirement via blob-vs-plaid preservation verification; gates the five Fourier-analyzer retirements via absorption parity).
- REQ_117 (DMD reorganization — *staging*; absorbed `centroid_dmd`'s modal paths).

**Atlas reference:** [docs/analysis_atlas.md](../../analysis_atlas.md) — the consolidation map is the canonical source → target mapping for every retirement.

**Attribution:** Engineering Claude.

---

## Scope evolution

The original REQ_102 retired a short list (`coarseness`, `fourier_nucleation`, `centroid_dmd`) plus the migrate-track tail of REQ_111. The Analysis Atlas (a)(b)(c) audit pass (2026-05-27) updated this scope:

- **`fourier_nucleation` is no longer retired.** The Atlas marks it `retain` — the iterative refinement is the value; only the one-shot projection step is absorbed (into REQ_126).
- **`centroid_dmd`** is substantially complete via REQ_117 (staging): modal paths absorbed by `activation_dmd` + `parameter_dmd`; trajectory portion deferred to a future `representation_trajectory` reorganization. The wrapper class is retired here once its remaining consumers migrate.
- **`coarseness` retirement is gated on REQ_126** verifying that `activation_basis_projection` preserves the blob-vs-plaid signal.
- **Migrate-track retirements redistribute.** Five Fourier-analyzer retirements (`dominant_frequencies`, `attention_fourier`, `neuron_fourier`, `attention_freq`, `neuron_freq_clusters`) gate on REQ_126. The pure-rename retirement of `effective_dimensionality` gates on REQ_111 (narrowed). The originally-paired `parameter_trajectory_pca` retirement is degenerate — that rename was already aligned at the registered-name level, so REQ_111's mechanical port closed the file/class gap and left no separate analyzer to retire.

This REQ becomes the close-out track for the Atlas-driven phase 1 consolidations.

---

## Problem Statement

Several analyzers have been superseded or carry layering violations that warrant retirement. Carrying them forward into the publishable library raises maintenance burden and confuses external readers about which paths are canonical.

Retirement candidates and their gates:

| Retiring analyzer | Replaced by | Gating REQ |
|---|---|---|
| `coarseness` | `activation_basis_projection` | REQ_126 (blob-vs-plaid preservation check) |
| `dominant_frequencies` | `weight_basis_projection` | REQ_126 (absorption parity) |
| `attention_fourier` | `weight_basis_projection` | REQ_126 (absorption parity) |
| `neuron_fourier` | `weight_basis_projection` | REQ_126 (absorption parity) |
| `attention_freq` | `activation_basis_projection` | REQ_126 (absorption parity) |
| `neuron_freq_clusters` | `activation_basis_projection` | REQ_126 (absorption parity) |
| `effective_dimensionality` | `weight_spectra` | REQ_111 (parity validation — bit-exact, recorded) |
| `parameter_trajectory_pca` | `parameter_trajectory` | Already complete via REQ_111 mechanical port — no separate analyzer to retire |
| `centroid_dmd` (wrapper) | `activation_dmd` + `parameter_dmd` | REQ_117 (already shipped — pending consumer migration) |

REQ_106 introduces a layering-audit deprecation criterion: an analyzer that re-implements an upstream derivation, mixes data-plane access into measure code, or cannot conform to declared-dependencies discipline is a deprecation candidate if migration would amount to a rewrite. Audit before declaring; some violations are migrations under their owning REQ, not retirements here.

**Retirement gating principle:** No analyzer is retired here without a recorded validation outcome from its owning REQ.

---

## Conditions of Satisfaction

### REQ_117-gated retirement (already shipped — pending consumer migration)

- [x] `centroid_dmd` (wrapper class): remaining consumers migrated to `activation_dmd` / `parameter_dmd` artifacts. Wrapper removed from `analysis/analyzers/`, `registry.py`, `__init__.py`. Old artifact directories left in place (read-only legacy); loader continues to read them on request.

### REQ_126-gated retirements

- [x] `coarseness` analyzer: removed after REQ_126 records blob-vs-plaid preservation verification on the canon reference set (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598).
- [x] `dominant_frequencies` analyzer: removed after REQ_126 records absorption parity for the W_E-site weight basis projection. Last analyzer consumer (`fourier_frequency_quality`) re-pointed to `neuron_grouping` under REQ_130 (2026-05-29).
- [x] `attention_fourier` analyzer: removed after REQ_126 records absorption parity for attention-site weight basis projections.
- [x] `neuron_fourier` analyzer: removed after REQ_126 records absorption parity for MLP-input weight basis projections.
- [x] `attention_freq` analyzer: removed after REQ_126 records absorption parity for attention-site activation basis projections.
- [x] `neuron_freq_clusters` analyzer: removed after REQ_126 records absorption parity for MLP activation basis projections. Three surviving analyzer consumers (`neuron_dynamics`, `freq_group_weight_geometry`, `neuron_group_pca`) re-pointed to `reconstruct_neuron_freq_norm` under REQ_131 (2026-05-29).

### REQ_111-gated retirements

- [x] `effective_dimensionality` analyzer: removed after REQ_111 records parity validation outcome (*matches* or *old-has-bug-fixed-in-new*) for `weight_spectra`. REQ_111 recorded **bit-exact parity on the canon reference set (2026-05-27)**.
- [x] `parameter_trajectory_pca` — degenerate retirement, complete via REQ_111 mechanical port (2026-05-27). The registered analyzer name was already `parameter_trajectory`; only the file (`parameter_trajectory_pca.py` → `parameter_trajectory.py`) and class (`ParameterTrajectoryPCA` → `ParameterTrajectory`) carried the old name. No separate analyzer is registered to remove.

### Layering audit (REQ_106 criterion)

- [ ] For each surviving analyzer in the registry post-consolidation, run REQ_106's grep tests against its source: does its `analyze()` call `np.linalg.svd` directly? Does it inline filesystem path construction? Does it re-derive a field that exists upstream?
- [ ] For each violation, classify: **migrate** (rewrite under owning REQ) or **retire** (value doesn't justify rewrite).
- [ ] Audit results recorded in this REQ's Notes section.

### Cleanup (applies to every retirement)

- [x] Family configurations updated: each `family.json` removes references to retired analyzers from its `analyzers` list. (Verified clean 2026-05-29 — no retired names in `data/modulo_addition_1layer/family.json`.)
- [x] `analysis/analyzers/__init__.py` no longer imports / exports retired analyzer classes.
- [x] `analysis/analyzers/registry.py` no longer registers retired classes. (Registry auto-discovers by module iteration; deleting the analyzer files de-registers them.)
- [x] Renderers / views that referenced retired analyzers either deleted or migrated. Dashboard churn hidden behind the visualization layer per user direction (separate cleanup track; not blocking). Renderer modules retained where still consumed by surviving views (analyzer-layer-only scope).
- [x] CHANGELOG entry on the release describing each retirement and pointing to its replacement.

### Documentation

- [x] `analysis/README.md` updated to reflect the canonical analyzer set (2026-05-29 — diagram, usage example, file listing, and artifact table re-pointed to `weight_basis_projection` / `activation_basis_projection`; stale `band_concentration.py` docstring fixed).
- [x] Each retired analyzer file (in git history) carries a final commit with deprecation notice + pointer to replacement before deletion. (Successor mapping recorded in CHANGELOG and the Per-retirement evidence pointers above; full source→target map in the Atlas consolidation map.)

---

## Constraints

**Must:**
- No retirement without a recorded validation outcome under the owning REQ.
- Existing artifacts on disk remain readable. Researchers with old `results/` trees continue to access historical data — `ArtifactLoader` reads retired-analyzer artifacts if asked.
- Family configurations stay valid throughout. No release with a config referencing a removed analyzer.

**Must avoid:**
- **Silent retirement.** Each retirement gets a CHANGELOG entry with a clear pointer to its successor.
- **Retiring analyzers whose outputs are still being read** by views or notebooks without prior migration. Audit consumer surface before deleting.
- **Bundling retirement with refactor.** Retirements happen here; refactors and consolidations happen under their owning REQ.

**Flexible:**
- Order of retirement within a gating bucket. Default: simplest first (REQ_117-gated; then REQ_111-gated as parity validation lands; then REQ_126-gated as REQ_126 completes per analyzer).
- Whether to keep retired-analyzer code in a `deprecated/` subdirectory for one release before deletion. Default: no — git history is sufficient.

---

## Architecture Notes

**Audit before deleting.** For each candidate:

1. Grep the codebase for the analyzer name (analyzer string, class name).
2. Confirm no view, no `load_data`, no notebook consumes its artifacts (or, if some do, that they've been migrated).
3. Confirm family configurations don't list it.
4. Run REQ_106 layering grep tests on surviving analyzers.
5. Then delete (or migrate under the owning REQ).

**Subtractive REQ.** This is mostly subtraction. The risk is consumer surface — views, notebooks, downstream analyzers that still read old artifacts. Audit is the safeguard.

**Dashboard cleanup is a separate track.** Per user direction, dashboard pages may continue to load via legacy paths during the parallel period; the visualization layer absorbs the churn. A future REQ may pick up dashboard migration explicitly. Not blocking here.

---

## Notes

### Consumer audit (2026-05-28, on `feature/req-102-analyzer-deprecation`)

First audit pass over the retirement candidates. Unlock confirmed: REQ_127
(downstream-visualization migration) merged to staging — that was the deferral
gate per the Atlas — and REQ_111/117/126 recorded their validation outcomes.

Finding: **this is consumer-migration, not pure deletion.** Surface by layer:

- **Family run configs (`data/*/family.json`): clean.** No retiring analyzer
  appears in any `analyzers` / `secondary_analyzers` / `cross_epoch_analyzers`
  list. Their on-disk artifacts are historical; nothing schedules them.
- **`views/universal.py` (ViewDefinition path): migrated, retirement-safe.**
  REQ_127 re-pointed it to the new analyzers via legacy adapter functions
  (`_adapt_attention_fourier_legacy`, the `dominant_frequencies` coefficient
  reshaper, `effective_dimensionality → weight_spectra`). It reads the *new*
  analyzers; the old-analyzer names survive only in adapter/renderer naming.
- **`views/dataview_universal.py` (DataView catalog): aspirational, not a hard
  blocker (resolved 2026-05-28).** It directly `load_epoch("dominant_frequencies")`
  / `("attention_fourier")`, but the DataView tier was stubbed and subsequently
  bypassed — per user, "we're not there yet" (a future path exists for heavier
  DataView use in notebook research + analysis/visualization consistency). Usage
  audit confirms the only consumers are `demos/demo_dataview_catalog.ipynb` and
  the test suite (`test_dataview_catalog.py`, `test_view_availability.py`,
  `test_freq_specialization_sequencing.py`) — no dashboard, no research notebooks.
  The dataviews load artifacts by string name (no analyzer-class import), so
  retiring the analyzers does not break imports; the affected dataview
  definitions just won't regenerate for new variants. Action: lightly re-point or
  prune those dataview definitions + refresh the demo/tests as part of cleanup —
  not a load-bearing migration.
- **`visualization/renderers/dmd.py`: obsolete dead UI code (confirmed
  2026-05-28).** Its three renderers (`render_dmd_eigenvalues`,
  `render_dmd_residual`, `render_dmd_reconstruction`) have **no live callers** —
  only `test_centroid_dmd.py` exercises them; no view, page, or notebook uses
  them. Superseded by the separate windowed DMD analyzers (`activation_dmd` =
  Activation-space, `parameter_dmd` = Parameter-space) and their renderers. Remove
  `renderers/dmd.py` (and its export from `visualization/__init__.py`) with the
  `centroid_dmd` retirement. **Salvage:** see Downstream handoff below.
- **`apps/dashboard/` pages (dimensionality, viability_certificate,
  activation_heatmaps, visualization): not blocking** per the existing
  "Dashboard cleanup is a separate track" direction above — legacy reads
  tolerated during the parallel period.
- **`apps/research/sketches/*`, `scripts/*`: low priority.** Exploratory; migrate
  opportunistically or leave (artifacts remain readable).

Proposed retirement order (simplest-consumer-surface first):

1. `effective_dimensionality` (REQ_111) — universal path already re-pointed to
   `weight_spectra`; verify no library-level direct loads remain, then retire.
2. `centroid_dmd` wrapper (REQ_117) — re-point the DMD view/dataview feed to
   `activation_dmd` / `parameter_dmd`, then retire.
3. The six REQ_126-gated Fourier/coarseness analyzers — **gated on migrating the
   DataView path** (blocker above) **and on the p109 refresh** (below).

### p109 validation data dependency (flagged 2026-05-28)

The REQ_126-gated retirements validate blob-vs-plaid / absorption preservation on
the canon 3-variant set (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598). As
of 2026-05-28, p113 and p101 have been re-analyzed with the new analyzer set, but
**p109 has not been refreshed.** The user will re-run p109 when we reach that
gate (it forces a WSL restart to free memory — see REQ_128 field evidence).
**Pause before closing any REQ_126-gated retirement until p109 is refreshed.**

### Analyzer-dependency audit (2026-05-28, second pass) — 2 of 6 REQ_126-gated retirements deferred

A producer/consumer audit of the analyzer dependency graph (the first audit
checked views/families/scripts but not analyzer-to-analyzer `ArtifactInput` /
`requires`) found that **two of the six REQ_126-gated analyzers are still
load-bearing producers for *surviving* analyzers.** REQ_127 migrated *views*,
not these analyzer deps. Gotcha: an artifact's name can differ from its
file/class name (`neuron_freq_clusters` registers as `neuron_freq_norm`).

| Retiring analyzer | Artifact name | Surviving analyzer consumers |
|---|---|---|
| `neuron_freq_clusters` | `neuron_freq_norm` | `neuron_dynamics`, `freq_group_weight_geometry`, `neuron_group_pca` (all `ArtifactInput`/`requires`) |
| `dominant_frequencies` | `dominant_frequencies` | `fourier_frequency_quality` (`depends_on` + `ArtifactInput`) |

Those consumers are themselves future-consolidation targets (Atlas: the
neuron-group analyzers fold into a planned consolidation; `fourier_frequency_quality`
is retained). Re-pointing them to the basis-projection replacements is a real
refactor with parity implications — out of REQ_102's "don't bundle retirement
with refactor" scope.

**Decision: defer these two** (confirmed with user 2026-05-28). Retire the four
with no surviving analyzer consumers — `coarseness`, `attention_fourier`,
`neuron_fourier`, `attention_freq` (done). The two deferred:

- **`neuron_freq_clusters` / `neuron_freq_norm` — leave untouched.** Per user,
  this is a *backbone of critical analysis*; do not change until there is clarity
  on what should change. It was slated for "replacement" mainly because the
  original is specialized rather than generic (reflected in the `freq_cluster`
  naming), not because it is unwanted. Not a retirement candidate for now.
- **`dominant_frequencies` — blocked on a `fourier_frequency_quality` refactor.**
  The unblock path is re-pointing `fourier_frequency_quality` from
  `dominant_frequencies` to `neuron_grouping`; if feasible it's worth doing to
  free `dominant_frequencies`. **This is its own requirement** (candidate stub).
  Research context: `fourier_frequency_quality` currently is *not* yielding
  valuable information — itself a finding. It sits at the hard, unresolved
  question of whether model performance depends on *which* frequencies are
  chosen and *when* that choice happens; the literature's "models learn
  frequencies from the prime / group math" frame is not supported by the data
  here (user's assessment). Because that analyzer will need close re-examination,
  **bit-wise parity is likely unnecessary** for any eventual re-point.

**Regression checksums:** `run_regression_check.py` excludes `coarseness`
already, but `attention_freq` / `attention_fourier` / `neuron_fourier` are in
`regression/reference_checksums.json` — retiring them requires a checksum regen
(a pipeline run on the user's side, like p109). `run_analysis_regression.py` is
registry-driven and auto-adapts; the hardcoded `run_regression_check.py` set is
updated by hand as part of each retirement.

### Downstream handoff: Centroid Trajectory view (2026-05-28)

Retiring `centroid_dmd` removes `renderers/dmd.py`, whose
`render_dmd_reconstruction` ("actual vs DMD-reconstructed centroid trajectories")
is the **only** place the raw per-class **Centroid Trajectory** is plotted — it
was enmeshed in that reconstruction renderer and lives nowhere else. The modal
DMD value is fully superseded by `activation_dmd` / `parameter_dmd`; the
trajectory plot is the lone salvage.

The analyzer-side home is `representation_trajectory` (planned consolidation
absorbing `global_centroid_pca` + `centroid_dmd`'s trajectory portion — see
REQ_117, Atlas consolidation map). That analyzer is **not built yet** (only
`global_centroid_pca.py` exists). So the Centroid Trajectory **view** is genuine
downstream work: **recreate it against `representation_trajectory` when that
analyzer lands.** Per user, it is fine to rebuild the view from scratch rather
than pivot the old reconstruction renderer. Capture this in the
`representation_trajectory` reorganization REQ when it is formalized.

### Layering audit results

*(empty until first audit pass on surviving analyzers)*

### Per-retirement evidence pointers

Format: `{analyzer} retired {date}, gated by {REQ}: {outcome pointer}`.

Removed in prior REQ_102 commits merged to `develop` (CoS gates satisfied at removal time):

- `effective_dimensionality` retired (REQ_111): bit-exact SV/PR parity vs `weight_spectra` on canon (2026-05-27).
- `centroid_dmd` retired (REQ_117): modal paths absorbed by `activation_dmd` + `parameter_dmd`; dead `renderers/dmd.py` removed (no live callers).
- `coarseness`, `attention_freq` retired (REQ_126): absorption parity into `activation_basis_projection`.
- `attention_fourier`, `neuron_fourier` retired (REQ_126): absorption parity into `weight_basis_projection`.

Removed this session (2026-05-29) — the two formerly-deferred analyzers, unblocked by REQ_130/REQ_131:

- `dominant_frequencies` retired 2026-05-29 (REQ_126): superseded by `weight_basis_projection`. Unblocked by REQ_130 re-pointing `fourier_frequency_quality` to `neuron_grouping`. Removed `analyzers/dominant_frequencies.py`, its `__init__` import/export, `scripts/run_regression_check.py` registration, `tests/test_dominant_frequencies_analyzer.py`; added to `EXCLUDED_ANALYZERS`. The four REQ_126 absorption-parity tests in `test_weight_basis_projection.py` were removed (their legacy on-disk artifacts were cleaned to match `family.json`; parity already recorded by REQ_126).
- `neuron_freq_clusters` (artifact `neuron_freq_norm`) retired 2026-05-29 (REQ_126): superseded by `activation_basis_projection`. Unblocked by REQ_131 re-pointing `neuron_dynamics` / `freq_group_weight_geometry` / `neuron_group_pca` to `reconstruct_neuron_freq_norm`. Removed `analyzers/neuron_freq_clusters.py`, its `__init__` import/export, `scripts/run_regression_check.py` registration, `tests/test_neuron_freq_specialization.py`; added to `EXCLUDED_ANALYZERS`. Renderer `renderers/neuron_freq_clusters.py` retained (still consumed by surviving views — analyzer-layer-only scope per user).

### Close-out (2026-05-29)

The 2026-05-28 deferral of `dominant_frequencies` and `neuron_freq_clusters` is
**reversed and resolved.** REQ_130 (fourier-quality re-point) and REQ_131
(neuron_freq_norm consumer migration) both merged to `develop` and severed the
analyzer-to-analyzer dependency edges that forced the deferral. With those cut,
both analyzers were removed under REQ_102's analyzer-layer-only scope (renderers
and dashboard pages remain a separate visualization track per the existing
direction above).

Validation: full `packages/miscope` suite green (**1473 passed, 29 skipped**),
ruff clean on touched files. The user removed stale legacy artifact directories
to match `family.json` and regenerated `tests/regression/reference_checksums_req102.json`
as the post-retirement baseline. `family.json` confirmed free of all retired
analyzer names.

All nine retirement candidates are now removed from the tree. The surviving
analyzer inventory is 24 modules, matching the Atlas target set.

### Pairings

- This REQ pairs with REQ_103 (PyPI Publication Hardening) — the publishable library should not include retired analyzers. CHANGELOG entries from retirements feed REQ_103's release notes.
- The full source → target map lives in the Atlas's consolidation map ([docs/analysis_atlas.md](../../analysis_atlas.md)). This REQ does not duplicate it; it cites it.

### Closure

This REQ closes when the last gated retirement is recorded. After that, the surviving analyzer set matches the Atlas's target inventory (~16 analyzers + new analyzers landing under future REQs).
