# Milestone v0.8.1 — Prediction Trace and Neuron Groups

**Released:** 2026-03-26
**Branch:** develop → main
**Version bump:** 0.8.0 → 0.8.1

---

## What shipped

### REQ_075: Per-Input Prediction Trace

**Problem solved:** All prior analysis operated on aggregate statistics. This fills the gap with a per-pair view: which specific training pairs has the model learned, in what order, and do residue classes graduate together?

**Key files:**
- `src/miscope/analysis/analyzers/input_trace.py` — InputTraceAnalyzer (per-epoch) + summary keys
- `src/miscope/analysis/analyzers/input_trace_graduation.py` — InputTraceGraduationAnalyzer (cross-epoch)
- `src/miscope/visualization/renderers/input_trace.py` — three renderers
- `src/miscope/views/universal.py` — view registrations (lines ~826–888)
- `tests/test_input_trace.py` — full test suite

**Key decisions:**
- Full p² grid (not just training pairs) with a `split` boolean mask — enables showing train/test structure in the accuracy grid without separate artifacts
- `graduation_epochs` uses a stability window (3 checkpoints) to avoid counting transient correct predictions
- Per-checkpoint storage (not a single stacked matrix) to keep artifact size tractable at 300+ checkpoints

**Findings so far (from phase diagram notebook):**
- Early-epoch frequency commitments predict final grokking structure — the "early prediction hypothesis" validated (see REQ_081 notes and phase_diagram.ipynb)
- Residue class accuracy timeline shows the expected staircase pattern during second descent

---

### Neuron Group PCA (informal, no REQ number)

**Problem solved:** Characterizes within-frequency-group coordination in W_in weight space across training.

**Key files:**
- `src/miscope/analysis/analyzers/neuron_group_pca.py` — NeuronGroupPCAAnalyzer
- `dashboard/pages/neuron_groups.py` — Neuron Groups page
- Scatter, trajectory, phase, graduation views registered in view catalog

**Key decisions:**
- Top-3 PC variance explained tracked per frequency group (not just top-1) to capture group spread
- Off-by-one bug fixed: neuron frequency labels now match the system convention used elsewhere

---

### Phase Diagram Notebook

**File:** `notebooks/phase_diagram.ipynb`

Explores variant classification in a frequency × timing feature space. The early prediction cell validates that frequency rank order at epoch ~500 predicts the final committed set — motivating structural diagnostic work (REQ_081).

---

### Variant Context Bar (dashboard)

Active variant metadata displayed inline on dashboard pages, reducing the need to navigate back to the selector to recall which variant is loaded.

---

## Infrastructure requirements added (not yet implemented)

These requirements were written this milestone and are active for the next cycle:

- **REQ_079** — Dashboard plot export (server-side PNG with canonical filenames, batch export)
- **REQ_080** — Artifact freshness and incremental analysis (staleness detection, gap reporting)
- **REQ_081** — Structural training diagnostics (geometry-based stopping criteria beyond loss thresholds)

---

## Source of truth post-milestone

Code and tests are authoritative. CHANGELOG.md provides quick context. Archived requirements preserve decision rationale and CoS for archaeology.
