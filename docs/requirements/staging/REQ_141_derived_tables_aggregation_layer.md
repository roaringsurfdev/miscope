# REQ_141: Derived Tables — a Declarative Cross-Epoch Aggregation Layer

**Status:** Completed — merged to `develop` 2026-06-06 (merge of
`feature/REQ_141_derived_tables`, 8 commits). Value-identical parity on the three
baselines; validated on the dashboard across force + no-force paths.
**Priority:** Medium — architectural; unlocks memory + provenance wins, not a defect.
**Branch:** `feature/REQ_141_derived_tables` (merged to `develop`).
**Parent:** REQ_110 (Lakehouse Surface) — builds the next layer on the columnar warehouse + query surface.
**Dependencies:** 110-A (`warehouse` columnar tables), 110-C (`miscope.query`), REQ_107 (output-schema registry), REQ_133 (freshness DAG). Interacts with REQ_140 (materializer scope/isolation).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

The analysis layer treats every analyzer as an independent island: each one
loads what it needs and recomputes its own view of the data. Now that the
lakehouse surface exists (columnar warehouse + DuckDB query layer + a registry
that declares every field's coords and kind), a sharper data model is possible —
one where **per-epoch analyzers expose conformed columnar facts, and cross-epoch
results that are mere aggregations of those facts are *derived* from them rather
than recomputed from raw artifacts.**

A litmus test separates three kinds of cross-epoch output:

1. **Per-epoch fact trapped in a cross-epoch analyzer.** A field that is a pure
   function of a single epoch's data, emitted by a cross-epoch analyzer only
   because that's where the stacking loop happens to live.
2. **Aggregation over the epoch axis.** A reduction (cumulative / windowed /
   argmax-over-time / group-by) that needs the time axis but not the stacked
   *tensors* — expressible as a query over a conformed columnar table.
3. **Joint fit over all epochs.** PCA / DMD / geometry that genuinely needs the
   whole stacked tensor at once. Irreducible; out of scope here.

The motivating case is the platform's highest-traffic lens. `neuron_dynamics`
(`output_scope="cross_epoch"`) emits `dominant_freq` and `max_frac`, both keyed
`(variant, epoch, neuron)` — **per-epoch argmaxes** (kind 1). To produce them it
stacks `(n_epochs, n_freq, d_mlp)` norm matrices in memory and argmaxes in bulk,
though nothing about the argmax needs the time axis. Those two columnar fields
are already materialized as the `neuron_frequency_attribution` warehouse table
(`warehouse/mapping_semantic.py`), which `analysis/neuron_frequency.py` reads as
*the* conformed `(epoch, neuron) → frequency` dimension (REQ_110D). So the
per-epoch columnar surface already exists — **it is just authored by a
cross-epoch analyzer that stacks to produce it.**

Downstream, `transient_frequency` (`output_scope="cross_epoch"`) is a pure
aggregation (kind 2) of that same dimension: it `load_cross_epoch`s the full 2-D
`dominant_freq`/`max_frac` arrays and computes committed-neuron counts per
`(epoch, frequency)` gated by `max_frac >= 0.70`, peaks (argmax-over-epoch), and
membership. Its committed-count core is literally
`SELECT epoch, frequency, COUNT(*) FROM neuron_frequency_attribution
WHERE frac_explained >= 0.70 GROUP BY epoch, frequency`.

This requirement establishes the missing layer and proves it on this lens:
a **derived table** — a registered, versioned, provenance-bearing query over
warehouse tables — as a first-class output producer alongside analyzers. The
codified-analysis invariant must hold: a derived table carries the same declared
output schema and audit trail an analyzer does. The goal is not to let
aggregations bypass the registry; it is to give them a *declarative* home in it.

**The purpose of the layer is frictionless researcher-facing analysis.** A slow or
memory-intensive computation is converted into on-disk storage: the cost is paid
**once, at materialization time**, and never re-paid in an interactive query. This
is the load-bearing reason to persist, and it implies a property each derived table
must declare — *materialized* vs. *view*. A cheap aggregation can be a query-time
view (computed on read, always live, no storage); an **expensive one must be a
persisted Parquet table** so the researcher reads bytes, not a recomputation. If the
cross-epoch SQL turns out to be time-intensive, it is saved to disk by definition —
"frictionless at the point of inquiry" is the acceptance test for where the
materialize/view line falls.

---

## Conditions of Satisfaction

- [ ] **A `DerivedTable` primitive exists and is first-class in the registry.**
  A derived table declares: a name, an output schema (REQ_107 `OutputField`s with
  kind + coords), a version, its input warehouse tables, and the query that
  produces it. `registry.field(name)` reports a derived field's producer (the
  derived table) and its keying coords exactly as it does for an analyzer field;
  `registry.search(...)` finds derived tables. A derived table without a declared
  output schema fails `registry.load()`, same as an analyzer.
- [ ] **Derived tables materialize into the warehouse and are queryable.** A
  derived table emits a Parquet table reachable through `miscope.query.open(...)`
  as a registered view, indistinguishable to a consumer from a semantic table.
  No consumer composes a file path to reach it (constraint 3).
- [ ] **Materialized vs. view is a declared, cost-driven property.** Each derived
  table declares whether it persists to disk (materialized) or computes on read
  (view). An expensive aggregation is persisted so a researcher's interactive query
  reads bytes, not a recomputation; the chosen line is justified by a measured
  query cost, not by guess. The neuron-frequency derived table is materialized
  (it feeds the dashboard and registry build).
- [ ] **Freshness threads through the DAG.** A derived table is a node downstream
  of its input tables; when an input is recomputed, the derived table is stale and
  re-materializes. This extends the REQ_133 ordering/freshness machinery to a new
  node kind — it does not fork a parallel staleness mechanism.
- [ ] **Bucket-1 keystone: per-epoch attribution moves to a per-epoch analyzer.**
  `dominant_freq` / `frac_explained` keyed `(epoch, neuron)` are produced by a
  **per-epoch** analyzer (depending on `activation_basis_projection`), so the
  `neuron_frequency_attribution` table is sourced from a natively per-epoch
  producer rather than back-filled by a cross-epoch stack. `neuron_dynamics`
  shrinks to its genuine cross-epoch tail (`switch_counts`, `commitment_epochs`),
  computed by streaming the per-epoch attribution — the
  `(n_epochs, n_freq, d_mlp)` stack is gone. This keystone is the first instance of
  the **output-completeness rule** below: the cross-epoch stack existed *only*
  because the per-epoch producer's output contract was too narrow to emit the fact
  it already had in hand.
- [ ] **Bucket-2 proof: `transient_frequency` becomes a derived table.** Its
  columnar outputs (committed counts, peaks, is-final, homeless) are produced by a
  registered derived table over `neuron_frequency_attribution`, not by an
  npz-stacking analyzer. The ragged peak-membership arrays (the one non-columnar
  part) are handled explicitly — see Decision Authority.
- [ ] **`analysis/neuron_frequency.py` is unaffected at its interface.** The
  conformed-dimension API (`NeuronFrequencyAttribution`, `load`) returns identical
  values; only its upstream producer changes. Its self-heal path still works.

## Validation

- [ ] **Byte-parity on the three baselines** (p113/s999/ds598, p109/s485/ds598,
  p101/s999/ds598 — the only variants validated against until REQ_137; never
  `find | head`). The `neuron_frequency_attribution` table, the
  `transient_frequency` outputs, and `neuron_dynamics`'s remaining fields are
  value-identical before vs. after (allow `rtol=1e-3` per REQ_126 if any float
  recompute is unavoidable; integer counts/indices must be exact). A shape-of-
  behavior change is a finding to surface, not precision noise.
- [ ] **Memory reduction is demonstrated, not asserted.** Using the existing
  `apps/research/sketches/profile_analysis_memory.py` harness, the peak resident
  set for producing the neuron-frequency lens on a baseline drops measurably (the
  `(n_epochs, n_freq, d_mlp)` stack and the full-array `load_cross_epoch` no longer
  appear). Record before/after numbers in the notes.
- [ ] **Registry round-trip test.** `registry.field("committed_counts")` (or the
  chosen derived field) reports the derived table as producer with the right
  coords; `registry.load()` rejects a derived table missing its schema.
- [ ] **Freshness test.** Recomputing `activation_basis_projection` for one epoch
  marks the per-epoch attribution stale → marks `transient_frequency`'s derived
  table stale → both re-materialize on the next pass; an untouched derived table is
  not rebuilt.

---

## Constraints

**Must have:**
- **Analyzer output-completeness (user expectation, 2026-06-06).** An analyzer
  declares and emits, at per-epoch granularity, every columnar fact it can cheaply
  produce from data already in hand. A too-narrow output contract — one headline
  output, everything else recomputed later — is the antipattern, because it forces
  a secondary pass that re-opens npz to recover a fact that was free during the
  analyzer's pass (the bucket-1 keystone is the canonical example). **Emission locus
  is settled: widen the analyzer's declared `AnalyzerSpec.outputs` (REQ_107) so the
  fact lands in the npz; the existing decoupled `materialize_variant_columnar`
  reshapes it to Parquet (cheap, no recompute). Do *not* co-emit Parquet inside the
  analyzer pass** — that would couple the analysis engine to warehouse format and
  forfeit "rebuild the warehouse from artifacts without re-running analysis." Goal
  state: npz opened at most once (by materialize); downstream reads Parquet, never
  npz. Within this REQ the rule is applied to the neuron-frequency slice only;
  widening other analyzers (`repr_geometry` circularity/fisher, `weight_spectra`
  participation ratios — both reached today via `load_summary`) is follow-on and
  feeds REQ_144.
- **Frictionless at the point of inquiry.** No researcher-facing query pays a
  slow/memory-intensive cost interactively. Expensive derived computation is
  materialized to disk and read back; the cost is paid once, off the interactive
  path. This is the layer's reason to exist, not a nice-to-have.
- **One provenance model.** A derived table carries a declared output schema +
  version + audit trail. Aggregations do not get a registry-bypass back door
  (this is the whole point of choosing the declarative form over loose SQL).
- **Storage-encapsulation invariant** (constraint 3): derived tables are reached
  through `miscope.query` / the warehouse reader, never a path literal; their
  definitions are code, their deployment locations are config.
- **Universal-instrument invariant** (constraint 1): a derived table is a view of
  data, not owned by a family. Family context (e.g. prime) enters as a column or a
  parameter, never as table ownership.

**Must avoid:**
- Converting all 11 cross-epoch analyzers in one pass. This requirement establishes
  the primitive and proves it on **one vertical slice** (neuron-frequency).
- Touching the bucket-3 analyzers' fits (`parameter_trajectory`,
  `global_centroid_pca`, `neuron_group_pca`, `intragroup_manifold`,
  `freq_group_weight_geometry`, `parameter_dmd`, `activation_dmd`, `gradient_site`).
  Their *columnar tails* are noted for follow-on, but the joint fits stay.
- A second staleness mechanism. Derived tables join the REQ_133 DAG; they don't get
  a bespoke freshness checker.

**Flexible:**
- Where derived-table definitions physically live (a `warehouse/derived/` module,
  co-located with the semantic mapping, or alongside the analyzer they replace) —
  decide for legibility.
- Whether the query is authored as SQL text or a small builder over the query
  surface — pick whichever keeps the audit trail honest and the definition readable.

---

## Context & Assumptions

- A derived table is conceptually adjacent to a REQ_110-A **semantic table**: both
  produce a queryable Parquet table from declared schema. The difference is the
  *source* — a semantic table maps an analyzer's on-disk artifact; a derived table
  runs a query over already-materialized tables. Reuse the semantic-table
  machinery where it fits rather than inventing a parallel writer.
- The three buckets and the litmus test are the durable design output of this
  session; the neuron-frequency slice is the first instance, not the whole job.
  Validate that the bucket-1/2 split holds for `input_trace_graduation`
  (already a model streamed aggregator — likely a clean bucket-2 derived table)
  before generalizing.
- Assume nothing is in production (v1.0.0 cutoff pending): no back-compat shim for
  the old `neuron_dynamics` output shape is needed — update consumers in place.

## Decision Authority
- [x] Propose options for review — **both forks resolved 2026-06-06 (see Resolved
  design below).** Registry modeling → sibling `DerivedTableSpec` + narrow
  `SchemaProducer` protocol (1C). Ragged peak-membership → columnar
  `(variant, frequency, member_neuron)` long table (2B).
- [ ] Make reasonable decisions and flag for review
- [ ] Full autonomy to proceed

### Resolved design (2026-06-06)
- **Registry modeling — 1C (sibling type + shared protocol).** A concrete
  `DerivedTableSpec` dataclass (`name`, `version`, `outputs: tuple[OutputField]`,
  `input_tables: tuple[str]`, `query`, `materialized: bool`) stays a sibling of
  `AnalyzerSpec` — *not* a unified producer record. `AnalyzerSpec` and
  `RegistryIndex.analyzers` are untouched. The registry's enumeration/lookup code
  (`field()`, `search()`, the output-schema half of `validate()`) iterates a narrow
  structural `SchemaProducer` protocol — anything with `name`, `version`,
  `outputs` — so analyzers and derived tables are treated uniformly without a lossy
  union. `RegistryIndex` gains `derived: tuple[DerivedTableSpec]`; `build_index`
  collects them; `FieldInfo` reports derived producers (kept distinct from analyzer
  producers so `field()` names the derived table as producer). Rejected: 1B
  (unified `producer` discriminator) — would refactor the stable REQ_107
  `.analyzers` surface for no gain.
- **Ragged peak-membership — 2B (columnar long table).** Peak membership becomes a
  derived `(variant, frequency, member_neuron)` long table over
  `neuron_frequency_attribution` (filter at peak epoch, gate by frac threshold).
  Peak membership *is* a per-neuron fact, so long-format is its natural home and
  the whole slice stays columnar. Value-parity (not byte) per the CoS clause. Only
  consumer of the old flat+offsets tensor is one research sketch
  (`apps/research/sketches/sketch_per_group_kinks.py`) — migrate it. Rejected: 2A
  (companion tensor) — preserves a tensor exception in an otherwise columnar slice.

## Success Validation
"Done" looks like: the neuron-frequency lens is produced by a per-epoch analyzer
feeding a registered derived table; the warehouse and query surface expose the
same tables with the same values; the registry reports the derived producer; the
memory profile drops; and the pattern is documented well enough that converting
the next aggregator is a mechanical application of the litmus test, not a redesign.

---

## Implementation status (2026-06-06) — COMPLETE on `feature/REQ_141_derived_tables`

Four commits, each green (full suite + ruff + pyright):
1. **DerivedTable primitive + registry wiring** (69438e2) — CoS #1. `DerivedTableSpec`
   + `SchemaProducer` protocol (1C); `RegistryIndex.derived`; `field()`/`search()`/
   `validate()`/`derived()` cover derived tables.
2. **Materialization + query exposure** (22cc3c0) — CoS #2/#3. `warehouse/derived.py`
   executor (`materialize_variant_derived`), `query.open_variant`, materialized→Parquet
   +catalog (discovered by `query.open`), view-mode live registration, REQ_140
   isolation, topological derived→derived ordering.
3. **Neuron-frequency slice — bucket-1 + bucket-2** (026c921) — one atomic change
   (the `transient_frequency`↔`neuron_dynamics` npz coupling). New per-epoch
   `neuron_frequency_attribution` analyzer; `neuron_dynamics` shrunk to its
   cross-epoch tail (cube gone); `transient_frequency` → `committed_counts` /
   `transient_frequencies` / `transient_peak_members` derived tables; analyzer
   deleted; consumers migrated (`variant_summary`, a `transient_frequency_dim`
   accessor for the renderers, the per-group-kinks sketch); `learned_frequencies`
   (unconsumed) retired; `family.json` updated.
4. **Freshness test** (this commit).

**Validation results:**
- **Parity — value-identical on all three baselines** (`apps/research/sketches/validate_req141_parity.py`,
  read-only): `dominant_freq` exact, `max_frac` rtol 1e-3, `switch_counts` /
  `commitment_epochs` exact (incl. NaN), `threshold` exact; derived
  `ever_qualified` / `is_final` / `peak_epoch` / `peak_count` / `homeless_count` /
  peak-members all exact. p101 exercises the transient path (1 transient freq).
- **Memory — the stacked allocation is eliminated, not asserted.** Old
  `neuron_dynamics` built an `(n_epochs, n_freq, d_mlp)` float64 cube; the new path
  never stacks the `n_freq` axis. Per baseline: p113 57.6MB→2.1MB (28×), p109
  55.5MB→2.1MB (27×), p101 72.1MB→2.9MB (25×); the per-epoch analyzer holds only the
  ~205–229KB `(n_freq, d_mlp)` matrix transiently. `transient_frequency` no longer
  `load_cross_epoch`s the full 2-D arrays (DuckDB streams the aggregation).
- **Registry round-trip / freshness** — covered by `test_derived_tables.py`,
  `test_derived_materialize.py` (incl. derived re-materialize on input recompute).

**Freshness approach (no second mechanism, per constraints):** derived tables join
the materialize DAG downstream of their inputs (columnar→derived ordering +
topological derived→derived sort). The warehouse full-rebuilds from artifacts
(REQ_140 model), so when an input analyzer is recomputed and the warehouse is
re-materialized, derived tables rebuild from the current tables — they are never
independently stale. No derived-table-specific staleness checker was added (the
constraint forbids it); the analyzer-level freshness DAG (REQ_080/133) already
decides what gets recomputed upstream.

## Notes

### The three buckets, mapped (session output, 2026-06-06)

| Bucket | Analyzers | Disposition |
|---|---|---|
| **1 — per-epoch fact trapped in cross-epoch** | `neuron_dynamics.dominant_freq` / `.max_frac` | Move to a per-epoch analyzer (this REQ). |
| **2 — aggregation over the epoch axis** | `transient_frequency`←neuron_dynamics; `input_trace_graduation`←input_trace; the `switch_counts`/`commitment_epochs` tail of `neuron_dynamics` | Derived tables / streamed reductions. `transient_frequency` proven here; others follow. |
| **3 — joint fit over all epochs (the exception)** | `parameter_trajectory`, `global_centroid_pca`, `neuron_group_pca`, `intragroup_manifold`, `freq_group_weight_geometry`, `parameter_dmd`, `activation_dmd`, `gradient_site` | Fits stay imperative. Only their columnar *tails* are future bucket-2 candidates. |

### Follow-on (explicitly out of scope here)
- **Recast the variant summary engine as derived tables (REQ_144, drafted).**
  `VariantAnalysisSummary` is the variant-level twin of this pattern — a mostly
  bucket-2 aggregator that reaches around the warehouse and round-trips
  JSON↔Parquet↔JSON. Sequenced after this REQ to keep the one-slice mandate intact.
  **Coupling to honor here:** `variant_analysis_summary._load_transient_metrics`
  consumes `transient_frequency` via `load_cross_epoch`; when this REQ converts
  that analyzer to a derived table (new shape, no back-compat per v1.0.0 cutoff),
  patch this consumer in place so the summary stays green — REQ_144 graduates the
  patch into the full conversion.
- Convert `input_trace_graduation` to a derived table (clean bucket-2; already
  streams).
- Audit bucket-3 columnar tails for surfaceable per-epoch/per-group metrics.
- Revisit whether `neuron_dynamics` survives as an analyzer at all, or dissolves
  fully into (per-epoch attribution analyzer) + (derived tables) once its tail is
  also expressed declaratively.

### Interaction with REQ_140
REQ_140 hardens `materialize_variant_columnar` (scope to `family.json`, per-analyzer
isolation). Derived-table materialization should inherit the same isolation
discipline — a failing derived table is skipped-and-reported, never fatal — and the
same scope source. Sequence REQ_141 after REQ_140 lands so it builds on the hardened
materializer rather than racing it on the same file.

### Why declarative (the fork the session resolved)
The session weighed keeping aggregators as streamed Python analyzers vs. making them
declarative derived tables. The declarative form was chosen: the SQL is shorter,
streams natively (near-zero Python-side memory), and — critically — *forces* the
provenance question to be answered once, in the registry, rather than re-answered per
analyzer. The risk it introduces (a class of outputs that bypass the audit trail) is
exactly what the first CoS forecloses.
