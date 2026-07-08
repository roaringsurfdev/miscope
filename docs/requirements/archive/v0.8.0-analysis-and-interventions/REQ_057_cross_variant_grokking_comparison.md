# REQ_057: Cross-Variant Grokking Health Comparison

**Status:** Active
**Priority:** Medium
**Related:** REQ_052, REQ_053, REQ_055, REQ_056

## Problem Statement

Comparison across variants is currently visual and anecdotal. Each variant is analyzed individually; conclusions about what separates healthy from anomalous models rely on side-by-side inspection of plots and qualitative judgment.

This has produced a working typology of failure modes (undershooter, overshooter, consensus-then-diversification) and a set of candidate metrics that appear to separate them. But the typology is not yet quantitative, and the candidate metrics have not been systematically compared across all variants in a single view. As a result:

- It is not possible to rank variants by grokking health
- It is not possible to confirm that the failure mode categories are discrete rather than a continuum
- Any new variant trained must be manually inspected and individually compared — there is no automated health assessment

## Conditions of Satisfaction

### Cross-Variant Metrics Table

- [ ] For each trained variant, compute and display in a single table:
  - Grokking onset epoch (first epoch where test loss < threshold)
  - Final test loss
  - Frequency band count (how many distinct frequency bands developed specialized neurons)
  - Fisher discriminant at final epoch (from representational geometry)
  - Circularity plateau value (from centroid geometry time-series)
  - Competition window duration (epoch range where neuron thrashing is active)
  - Convergence status at training end (have key metrics plateaued?)

- [ ] Table is sortable by any column
- [ ] Variants are colored or labeled by failure mode category (healthy / undershooter / overshooter / consensus / unknown)

### Cross-Variant Trajectory Overlays

- [ ] Loss curves for all variants on a single chart — aligned by absolute epoch and optionally by grokking onset
- [ ] Fourier quality trajectory for all variants on a single chart (from REQ_052)
- [ ] Neuron specialization trajectory for all variants on a single chart, at adjustable threshold (from REQ_056)

### Failure Mode Classification

- [ ] A rule-based or threshold-based classification assigns each variant to a failure mode category using the metrics above
- [ ] Classification thresholds are explicit and adjustable — no hidden criteria
- [ ] Classification output is auditable: for any variant, show which metrics drove its category assignment

### Data Access

- [ ] Per-variant summary metrics exposed as a dataview — consumers can load all variants' summary metrics in a single call without iterating per-variant artifacts

## Constraints

**Must:**
- Use only metrics derivable from existing stored artifacts — no new analysis passes required for the initial implementation
- Classification must be rule-based and transparent, not model-based

**Must not:**
- Hardcode variant lists — must discover available variants from the family registry

**Explicitly deferred:**
- Causal analysis or statistical testing of failure mode predictors
- Automated anomaly detection beyond the rule-based classification

## Context & Assumptions

The existing variants in modulo_addition_1layer span a range of grokking behaviors:
- Fast, clean grokkers (p109/485, p113/485)
- Normal grokkers (most variants)
- Anomalous: undershooter (p101/999 — premature frequency commitment, degenerate cos/sin)
- Anomalous: overshooter (p59/485 — gradient-poor path, missing frequency band, late settlement)
- Anomalous: consensus-then-diversification (p107/999 — initial head consensus, unstable post-grokking diversification)

The metrics developed across prior requirements (Fisher discriminant, circularity, Fourier quality, neuron specialization) have all been validated against individual variants. This requirement assembles them into a unified cross-variant view.

The forcing question for this requirement: does the typology hold quantitatively, or does it dissolve when looked at systematically? If the failure modes are real, the metrics should cluster. If they are anecdotal, the metrics will show a continuum with no clear separations.

The per-variant summary dataview is the key infrastructure piece — it enables future requirements to consume cross-variant data without re-implementing per-variant iteration.

## Notes

**Metrics derivation sources:**
- Grokking onset: derivable from loss curve (metadata) at inference time
- Frequency band count: derivable from neuron_dynamics cross-epoch at threshold t%
- Fisher discriminant: from representational_geometry cross-epoch artifact
- Circularity plateau: from representational_geometry cross-epoch artifact (max or late-epoch value)
- Competition window: from neuron_dynamics — epoch range where switch_count > 0 (or > threshold)
- Convergence status: heuristic on whether key metrics have stabilized in last N epochs

**Display priority:**
- The metrics table and loss curve overlay are highest value — implement these first
- Failure mode classification can follow once the table reveals whether the metrics actually cluster
