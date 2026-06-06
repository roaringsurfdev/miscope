# REQ_144: Recast the Variant Summary Engine as Derived Tables

**Status:** Draft — queued behind REQ_141 (sequencing decided in a 2026-06-06 session).
**Priority:** Medium — architectural; pays down a large imperative aggregator and
removes a warehouse-bypass, not a defect.
**Branch:** TBD (`feature/REQ_144_variant_summary_derived_tables`); lands **after
REQ_141 merges to `develop`**.
**Parent:** REQ_141 (Derived Tables). This is the "next aggregator" REQ_141's
success criterion forecasts — the proof that converting one is a mechanical
application of the litmus test, not a redesign.
**Dependencies:** REQ_141 (`DerivedTable` primitive + registry modeling), 110-A
(`warehouse` columnar tables), 110-C (`miscope.query`), REQ_107 (output-schema
registry), REQ_133 (freshness DAG). Inherits REQ_140's materializer isolation.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

`miscope.analysis.variant_analysis_summary.VariantAnalysisSummary` is an 814-line
imperative cross-analyzer rollup that produces `variant_summary.json` — the
per-variant outcome snapshot (loss extrema, grokking/second-descent timing,
window ranges + per-window metrics, first-mover detection, learned/committed
frequencies, handshake dynamics, transient/homeless counts, failure mode). It is
**REQ_141's exact pattern, one level up:** a cross-axis aggregator that is mostly
*bucket 2* (reductions over the epoch axis) but is written as Python loops over
in-memory lists, reaching around the warehouse into raw artifacts.

Three problems make it a refactor candidate now that the lakehouse surface exists:

1. **It bypasses the warehouse.** It reads raw artifacts directly —
   `load_summary("weight_spectra")`, `load_summary("repr_geometry")`,
   `load_cross_epoch("transient_frequency")`, and `variant.metadata["train_losses"]`
   — rather than querying conformed columnar tables. This is the precise
   artifact-stacking reach REQ_141 exists to retire.

2. **The plumbing does a JSON↔Parquet↔JSON round-trip.** The engine writes
   `variant_summary.json`; `warehouse/outcomes.py` flattens scalar fields to
   `variant_outcomes` columns *and* carries the whole blob as a `summary_json`
   string; `build_variant_registry` reads it back and `json.loads` the carrier.
   The rich window dicts are trapped in the carrier, unqueryable. Under a
   derived-table model, `variant_outcomes` *is* a derived table over warehouse
   tables and the round-trip dissolves; the window metrics become proper
   long-format columnar tables.

3. **REQ_141 breaks it regardless.** `_load_transient_metrics` reads
   `transient_frequency` via `load_cross_epoch`; REQ_141 converts that analyzer
   into a derived table with a new output shape and no back-compat shim (v1.0.0
   cutoff). REQ_141 must patch this consumer minimally to stay green; REQ_144 is
   where that patch graduates into the engine's full conversion.

The litmus test from REQ_141 applies here too. Most summary fields are **bucket 2**
(loss min/argmin epochs, threshold-crossing epochs, dimensionality crossover,
window ranges and per-window start/end metrics, first-mover detection) — SQL over
conformed tables. A few touch **bucket 3** outputs (geometry/dimensionality
participation ratios from joint fits), which stay imperative but should be
*queried as columnar facts*, not re-loaded from artifact summaries. The failure-mode
classification ([`_get_variant_preformance_classification`],
[`_load_failure_mode`]) is a pure function of already-computed scalar outcomes — a
natural derived-table column or a small classifier over the outcomes row.

---

## Conditions of Satisfaction (draft — refine against REQ_141's final shape)

- [ ] **`variant_outcomes` becomes a derived table**, not a JSON-flatten of a
  separately-written `variant_summary.json`. It declares its output schema +
  version + provenance like any REQ_141 derived table; `registry.field(...)`
  reports it as producer.
- [ ] **Window metrics get a queryable home.** The per-window start/end metric
  dicts currently trapped in `summary_json` become a long-format columnar /
  derived table keyed by `(variant, window, …)`, reachable through
  `miscope.query` — no consumer composes a path (constraint 3).
- [ ] **The engine stops reaching around the warehouse.** Inputs are read as
  conformed warehouse facts (losses, weight-spectra participation ratios,
  repr-geometry circularity/fisher, neuron-frequency attribution, the REQ_141
  `transient_frequency` derived table), not via `load_summary` / `load_cross_epoch`
  / `variant.metadata`.
- [ ] **`build_variant_registry` is unaffected at its interface** (or the registry
  is itself reframed as a derived view) — consumers (dashboard `variant_table`,
  `viability_certificate`, `initialization_sweep`, `analysis_run`; the freshness
  DAG; scripts) read identical values.
- [ ] **Freshness threads through the DAG** — the outcomes/window derived tables
  are downstream nodes; recomputing an input restages them (REQ_133, no second
  mechanism).
- [ ] **The god-object is decomposed.** The 814-line engine is broken along the
  bucket boundaries; bucket-2 reductions move into derived-table definitions,
  leaving a thin imperative tail for genuine bucket-3 reads.

## Validation

- [ ] **Byte-parity on the three baselines** (p113/s999/ds598, p109/s485/ds598,
  p101/s999/ds598; never `find | head`). `variant_summary.json` /
  `variant_registry.json` / `variant_outcomes` are value-identical before vs.
  after (`rtol=1e-3` per REQ_126 only where a float recompute is unavoidable;
  integer counts/epochs exact). A shape-of-behavior change is a finding, not noise.
- [ ] **Memory/cost demonstrated, not asserted** (REQ_141's frictionless test):
  expensive derived computation is materialized; interactive registry/outcomes
  queries read bytes.
- [ ] **Consumer regression net stays green** — `test_variant_summary.py`,
  `test_warehouse.py`, `test_freshness.py`, `test_families.py`, dashboard
  `test_variant_table.py`.

## Constraints

- **Must avoid** a second staleness mechanism or a registry-bypass back door
  (REQ_141 invariants carry over).
- **Universal-instrument invariant:** family context (prime, bands) enters as a
  column/parameter, never as table ownership.
- **Storage-encapsulation invariant:** derived tables reached through
  `miscope.query` / the warehouse reader; definitions in code, locations in config.

## Decision Authority
- [x] Propose options for review — open forks: (a) whether `variant_registry.json`
  survives as a file or becomes a pure derived view; (b) how much of the failure-mode
  classifier becomes a declarative derived column vs. a thin Python classifier over
  the outcomes row; (c) the keying of the window-metrics long table.

## Notes
- Sequencing rationale (2026-06-06 session): kept out of REQ_141 to preserve its
  "one vertical slice" mandate and because the summary engine's blast radius
  (dashboard pages, freshness, scripts, dedicated tests) warrants its own parity
  gate. REQ_141 patches the `transient_frequency` consumer minimally; REQ_144
  graduates that into the full conversion.
- This engine is the canonical *variant-level* bucket-2 aggregator, mirroring the
  *cross-epoch* one (`transient_frequency`) REQ_141 proves on. Confirming the
  litmus split holds cleanly here is the generalization evidence REQ_141 asks for.
