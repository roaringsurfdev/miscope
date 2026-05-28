# REQ_102: Analyzer Deprecation (Retire Stale Paths)

**Status:** Draft (rescoped after Analysis Atlas (a)(b)(c) audit pass)
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
| ~~`parameter_trajectory_pca`~~ | `parameter_trajectory` | Already complete via REQ_111 mechanical port — no separate analyzer to retire |
| `centroid_dmd` (wrapper) | `activation_dmd` + `parameter_dmd` | REQ_117 (already shipped — pending consumer migration) |

REQ_106 introduces a layering-audit deprecation criterion: an analyzer that re-implements an upstream derivation, mixes data-plane access into measure code, or cannot conform to declared-dependencies discipline is a deprecation candidate if migration would amount to a rewrite. Audit before declaring; some violations are migrations under their owning REQ, not retirements here.

**Retirement gating principle:** No analyzer is retired here without a recorded validation outcome from its owning REQ.

---

## Conditions of Satisfaction

### REQ_117-gated retirement (already shipped — pending consumer migration)

- [ ] `centroid_dmd` (wrapper class): remaining consumers migrated to `activation_dmd` / `parameter_dmd` artifacts. Wrapper removed from `analysis/analyzers/`, `registry.py`, `__init__.py`. Old artifact directories left in place (read-only legacy); loader continues to read them on request.

### REQ_126-gated retirements

- [ ] `coarseness` analyzer: removed after REQ_126 records blob-vs-plaid preservation verification on the canon reference set (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598).
- [ ] `dominant_frequencies` analyzer: removed after REQ_126 records absorption parity for the W_E-site weight basis projection.
- [ ] `attention_fourier` analyzer: removed after REQ_126 records absorption parity for attention-site weight basis projections.
- [ ] `neuron_fourier` analyzer: removed after REQ_126 records absorption parity for MLP-input weight basis projections.
- [ ] `attention_freq` analyzer: removed after REQ_126 records absorption parity for attention-site activation basis projections.
- [ ] `neuron_freq_clusters` analyzer: removed after REQ_126 records absorption parity for MLP activation basis projections.

### REQ_111-gated retirements

- [ ] `effective_dimensionality` analyzer: removed after REQ_111 records parity validation outcome (*matches* or *old-has-bug-fixed-in-new*) for `weight_spectra`. REQ_111 recorded **bit-exact parity on the canon reference set (2026-05-27)** — retirement is unblocked; pending consumer migration (dashboard pages, family.json) before removal.
- [x] ~~`parameter_trajectory_pca`~~ — degenerate retirement, complete via REQ_111 mechanical port (2026-05-27). The registered analyzer name was already `parameter_trajectory`; only the file (`parameter_trajectory_pca.py` → `parameter_trajectory.py`) and class (`ParameterTrajectoryPCA` → `ParameterTrajectory`) carried the old name. No separate analyzer is registered to remove.

### Layering audit (REQ_106 criterion)

- [ ] For each surviving analyzer in the registry post-consolidation, run REQ_106's grep tests against its source: does its `analyze()` call `np.linalg.svd` directly? Does it inline filesystem path construction? Does it re-derive a field that exists upstream?
- [ ] For each violation, classify: **migrate** (rewrite under owning REQ) or **retire** (value doesn't justify rewrite).
- [ ] Audit results recorded in this REQ's Notes section.

### Cleanup (applies to every retirement)

- [ ] Family configurations updated: each `family.json` removes references to retired analyzers from its `analyzers` list.
- [ ] `analysis/analyzers/__init__.py` no longer imports / exports retired analyzer classes.
- [ ] `analysis/analyzers/registry.py` no longer registers retired classes.
- [ ] Renderers / views that referenced retired analyzers either deleted or migrated. Dashboard churn may be hidden behind the visualization layer per user direction (acknowledged as separate cleanup track; not blocking).
- [ ] CHANGELOG entry on the release describing each retirement and pointing to its replacement.

### Documentation

- [ ] `analysis/README.md` updated to reflect the canonical analyzer set.
- [ ] Each retired analyzer file (in git history) carries a final commit with deprecation notice + pointer to replacement before deletion.

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
- **`views/dataview_universal.py` (DataView catalog, published library): NOT
  migrated — blocker.** Directly `load_epoch("dominant_frequencies")` and
  `load_epoch("attention_fourier")`. The DataView catalog is wired in
  (`views/__init__.py` exports it; `variant.py` / `catalog.py` consume it), so
  these are live library consumers reading old artifacts directly. Must migrate
  (or confirm DataView is being deprecated) before retiring those two Fourier
  analyzers. **Open question for the user:** REQ_127 migrated the ViewDefinition
  path but appears to have left the DataView path — in scope for REQ_102, or its
  own track?
- **`visualization/renderers/dmd.py`: fed from `centroid_dmd`.** Its data
  contract is `load_cross_epoch("centroid_dmd")`. The actual load lives in the
  view/dataview that feeds it; re-pointing that feed to `activation_dmd` /
  `parameter_dmd` is the REQ_117-gated consumer migration this REQ owns.
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

### Layering audit results

*(empty until first audit pass on surviving analyzers)*

### Per-retirement evidence pointers

Format: `{analyzer} retired {date}, gated by {REQ}: {outcome pointer}`.

- *(empty until first retirement)*

### Pairings

- This REQ pairs with REQ_103 (PyPI Publication Hardening) — the publishable library should not include retired analyzers. CHANGELOG entries from retirements feed REQ_103's release notes.
- The full source → target map lives in the Atlas's consolidation map ([docs/analysis_atlas.md](../../analysis_atlas.md)). This REQ does not duplicate it; it cites it.

### Closure

This REQ closes when the last gated retirement is recorded. After that, the surviving analyzer set matches the Atlas's target inventory (~16 analyzers + new analyzers landing under future REQs).
