# REQ_052: Fourier Frequency Quality Scoring

**Status:** Active
**Priority:** High
**Related:** REQ_043 (Fourier Profile Expansion, deferred), REQ_053 (Frequency Quality and Error Analysis, depends on this)
**Last Updated:** 2026-03-03

## Problem Statement

The platform currently captures *which* frequencies the model selects (via the `dominant_frequencies` analyzer) but has no way to evaluate *whether* those frequencies are sufficient to represent the ground-truth signal for the task.

For modulo addition, the correct solution has a known Fourier structure: the perfect logit tensor can be fully decomposed in the Fourier basis, and only a small subset of frequencies carry meaningful energy. A model that initializes with (or converges to) high-quality frequencies — those that account for most of the ideal signal's energy — may learn differently than one that selects a poor subset.

This creates a gap between what is observed and what it means: we can watch the model select frequencies, but we cannot score that selection or track how close it is to optimal over training.

## Conditions of Satisfaction

### Analyzer

- [ ] A new analyzer `fourier_frequency_quality` is implemented under `src/miscope/analysis/`
- [ ] The analyzer accepts the prime `p` from family context (not hardcoded)
- [ ] Per-epoch artifact contains:
  - `quality_score` — scalar in [0, 1]: R² (energy fraction) of the ideal signal explained by the model's dominant frequency set, i.e. `||T_F||² / ||T||²`
  - `dominant_frequencies` — int array of the frequency indices used (sourced from existing `dominant_frequencies` artifacts)
  - `k` — number of dominant frequencies evaluated
  - `reconstruction_error` — scalar residual energy not captured (`1 - quality_score`)
- [ ] Artifacts follow the standard `artifacts/fourier_frequency_quality/epoch_{NNNNN}.npz` pattern
- [ ] The ideal logit tensor is computed analytically from task structure (not derived from model weights)

### Views

- [ ] A cross-epoch view `fourier_quality_trajectory` is registered in the view catalog
- [ ] The view shows quality score over training epochs as a line plot
- [ ] The view is accessible via `variant.at(epoch).view("fourier_quality_trajectory")`
- [ ] The selected epoch (from `EpochContext`) is marked as a vertical cursor on the trajectory

### Validation

- [ ] A notebook cell demonstrates the quality score computed for at least two variants with different convergence profiles
- [ ] A variant that converges cleanly (e.g., p59/485) and one that struggles (e.g., p101/999) are compared to confirm the score reflects observable training differences

## Constraints

**Must:**
- Reuse existing Fourier library functions (`project_onto_fourier_basis` or equivalent) — no reimplementation
- The quality score is a property of the *frequency subset*, not of the weights themselves; the ideal tensor is computed from task parameters alone
- The analyzer reads dominant frequencies from existing `dominant_frequencies` artifacts rather than re-deriving them

**Must not:**
- Change the `dominant_frequencies` analyzer or its artifact schema
- Require REQ_043 (Fourier Profile Expansion) — that is a separate deferred requirement

**Decision authority:**
- **Resolved:** Quality score is energy fraction / R² (`||T_F||² / ||T||²`). Cosine similarity (`||T_F|| / ||T||`) is the square root of this and is monotonically equivalent, but R² has cleaner scientific precedent and avoids compressing the scale toward 1.0.
- **Claude decides:** Specific Fourier projection implementation and artifact field names within the schema above

## Context & Assumptions

The ideal logit tensor for mod-p addition is a p×p×p tensor where T[a, b, c] = 1 if (a+b) % p == c, else 0. Its Fourier decomposition is computed over the (a, b) input dimensions using the standard cosine/sine basis for integers mod p. The energy at frequency k is the combined norm of the sine and cosine components at that frequency across all output positions c.

The model's dominant frequencies (from the existing analyzer) are the k highest-norm frequencies from the embedding `W_E`. The quality score asks: if we restrict the ideal tensor's Fourier decomposition to only those k frequencies, how much of the total energy is retained?

This framing treats the model's frequency selection as a *filter* on the ideal signal. A quality score of 0.9 means the model's chosen frequencies can reconstruct 90% of the correct answer's energy.

## Notes

- The score is computed at each epoch, so "does quality improve over training?" is directly answerable
- Comparing early-epoch quality scores across variants may reveal whether good initialization predicts fast convergence — this is the primary research motivation
- REQ_053 builds directly on this artifact to correlate quality with per-class accuracy failures
