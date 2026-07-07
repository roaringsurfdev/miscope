# v0.9.0 — lakehouse-platform

**Released:** 2026-07-07
**Role:** Last release before the v1.0.0 clean-room rebaseline.

This milestone consolidates a large body of work (REQ_055–REQ_156, 58 requirements)
that accumulated on `develop` since v0.8.4. It marks the platform's maturation from a
collection of individual analyzers into a lakehouse-backed, registry-driven,
monorepo-packaged analysis platform, capped by Layer 4 circuit analysis.

It is the **substrate the v1.0.0 clean-room design was lifted from.** The design
docs in `docs/design/` (`PLATFORM.md`, `data_model.md`, `architecture.md`) re-state
this platform's intended design under the Line (infrastructure vs analysis) and
drive the v1.0.0 reconciliation.

## Highlights by arc

- **Lakehouse data platform** (REQ_108, 110/110A/110B/110C) — columnar warehouse,
  address-only tensor catalog, DuckDB query surface, publication bundles.
- **Analysis-platform foundations** — hooked models (REQ_105, 112, 113),
  discoverability registry (REQ_107), measurement primitives (REQ_109, 126),
  analyzer spec/registry & I/O unification (REQ_119, 120, 121), fluid dependency
  scheduler (REQ_133), attention-head coordinate (REQ_136).
- **Monorepo & config** (REQ_115, 123, 124, 125) — uv workspace layout, unified data
  root, per-app config.
- **Derived tables & incremental refresh** (REQ_140, 141, 144, 145, 149) —
  aggregation layer, signature-based freshness predicate replacing `force`.
- **Layer 4 circuits** (REQ_152, 154, 155, 156) — full OV circuit, circuit spectra
  siblings/views, conformed circuit tables.
- **Dynamics & geometry** (REQ_055, 073, 088, 089, 090, 092, 096, 117, 118) — DMD
  reorganization, 2-layer MLP family, frequency-group weight geometry, intragroup
  manifold geometry, neuron grouping.
- **Consolidation & migration** (REQ_097, 098, 102, 104, 114, 116, 122, 127–132,
  135) — analyzer cleanup, deprecations, consumer migrations, dead-code cull,
  manifest retirement.
- **Regression scaffolding** (REQ_086, 087, 134) — regression snapshot scaffold,
  activation bundle abstraction, canonical regression harness pattern.

## Key decisions preserved here

- Lint + typecheck became **blocking CI gates** with a per-requirement staging
  checkpoint (drift prevention), after an 89-error pyright drift cleanup.
- **Storage layout is internal to the API** (third architectural invariant): no path
  literals or direct `ArtifactLoader` use outside the storage primitives.
- Warehouse `variant_outcomes` (REQ_144) became the **source of truth** for
  per-variant metrics; `variant_summary.json` retired, `variant_registry.json`
  demoted to a view.

## Carry-forward (not in this release)

Work still `active/` at cut time carries into the v1.0.0 line and will be
**re-authored against the clean-room design** rather than continued as-is — notably
REQ_158 (Architecture object) and REQ_159 (TaskType/Task context provider), whose
design was materially simplified by the rebaseline (Task collapse; see
`docs/design/PLATFORM.md`).
