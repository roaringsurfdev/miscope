# REQ_152: FullOVCircuit — the end-to-end OV path as a first-class object

**Status:** Draft — not started.
**Priority:** High — the data model's designated "build first" Layer 4 object
(`data_model_master.md`, Part V build-readiness scan). First requirement built
*against* the canonical data model.
**Branch:** `feature/REQ_152_full_ov_circuit`.
**Attribution:** Engineering Claude (under user direction).
**Data model anchor:** `docs/data_model_master.md` Layer 4 — `FullOVCircuit`
(`STABLE · empty · PLANNED`); companion `docs/canonical_object_harvest.md` Source 1.

---

## Problem Statement

The data model harvests a precise, community-accepted **theorem object** from the
Elhage *Mathematical Framework*: the full OV circuit `W_U W_O W_V W_E` — the
square `[vocab, vocab]` end-to-end map from a source token to its output-logit
contribution, per attention head. It is **given-real by mathematical derivation**
(Gate 1, `STABLE`): a closed-form function of given weight primitives, an object
*whether or not anything has measured it*. For modular arithmetic it exposes the
task's additive structure directly, and its eigenspectrum yields a **copying
score** (the head's tendency to copy vs. transform) — a standard, interpretable
mechanistic diagnostic we do not currently compute anywhere.

Today this object is **empty**: the composed weight is never materialized, so none
of its gauge-invariant attributes exist in the store and the registry has nothing
to search. The codebase already has every ingredient — per-head weight access
(`library/weights.py`), the einsum composition precedent
(`modulo_addition_1layer.py` `_compose_attn_qk` / `_compose_attn_v`), the
universal Fourier instrument over composition sites (`weight_basis_projection`),
the tensor catalog write path (`tensor_catalog.py`), and `Coord.HEAD` (REQ_136).
What is missing is the composition itself and the one genuinely new measurement
(the eigenspectrum-derived copying score).

This requirement materializes `FullOVCircuit` and points instruments at it, so its
attributes become first-class, registry-searchable, warehouse-queryable fields.

## Approach (decided)

A **dedicated `full_ov_circuit` per-epoch analyzer** (the encapsulation-first
option; weight_spectra stays untouched), paired with a **`full_ov` composition
site** so the universal Fourier instrument supplies `dominant_frequency` for free.
The asymmetry this resolves: `weight_basis_projection` is already site-driven (runs
on composed operands), but `weight_spectra` is matrix-name-driven (iterates
`WEIGHT_MATRIX_NAMES`, never sees a composed matrix) and emits no
`effective_rank`/`operator_norm` columns. Rather than refactor a STABLE,
widely-used analyzer, the new analyzer owns the composition, the eigenspectrum, and
the spectral-invariant columns; the universal Fourier lens contributes the
frequency column via the existing site machinery.

Attribute homes (per `FullOVCircuit` in the data model):

| Attribute | Source | Mechanism |
| --- | --- | --- |
| `copying_score` | **NEW** | eigenspectrum of the composed matrix: Σ Re(λ)₊ / Σ\|λ\| |
| `effective_rank` | NEW (this analyzer) | participation ratio of the composed matrix's singular values |
| `operator_norm` | NEW (this analyzer) | largest singular value of the composed matrix |
| `dominant_frequency` (task-conditional) | CHEAP | `weight_basis_projection` over the new `full_ov` site |
| composed `[p,p]` matrix + eigenvalues | NEW | tensor refs (blob plane) |

## Conditions of Satisfaction

- [ ] **`full_ov_circuit` analyzer** (`per_epoch`, `needs_weights=True`). Composes,
  per head, the end-to-end OV path `W_U W_O W_V W_E` restricted to the task's token
  range `[:p]` → a `(n_heads, p, p)` matrix (the `attn_qk` shape family). Follows
  the einsum composition precedent; no inline ad-hoc weight reshaping outside the
  library helpers.
- [ ] **`copying_score` (the new measurement).** Per head, the eigenspectrum-based
  copying score `Σ Re(λ)₊ / Σ |λ|` over the eigenvalues of the head's composed
  `[p,p]` matrix. Declared on `AnalyzerSpec.outputs` (REQ_107) as `kind=columnar`,
  `dtype=float64`, `coords=(variant, epoch, site, head)`. **Sanity-bounded `[0, 1]`.**
- [ ] **`effective_rank`, `operator_norm` columnar fields** on the same analyzer,
  same coords/keying — gauge-invariant scalars of the composed operand alone (Part I
  step 5 promotion criterion). Computed from the composed matrix's SVD.
- [ ] **Tensor refs for the composed matrix + eigenvalues.** The dense `(n_heads,
  p, p)` circuit and its per-head eigenvalues declared as `kind=tensor`,
  `coords=(variant, epoch, site)`, landing in the tensor catalog via the existing
  write path (header-read shape, bytes-authoritative dtype) — never shoved into the
  columnar plane.
- [ ] **`full_ov` BasisProjectionSite** added to the family's
  `weight_basis_projection_sites` (period axes `(1, 2)`, 2D, mirroring `attn_qk`),
  so `dominant_frequency` for the OV circuit comes from the universal Fourier
  instrument — not re-implemented in the new analyzer.
- [ ] **Registry declaration & discoverability (REQ_107).** Analyzer registered via
  `@register_analyzer(SPEC)` with full `outputs` schema; added to the family's
  `analyzers` list; `registry.field("copying_score")` resolves to it.
- [ ] **Warehouse landing.** Columnar fields reach `miscope.query` /
  `variant.warehouse` and tensor fields resolve through the catalog, via the
  standard materialize pass — no path literals, no direct `ArtifactLoader` outside
  the API (invariant 3). A **conformed `FullOVCircuit` semantic claim** joining the
  analyzer's scalars with `weight_basis_projection`'s `full_ov` `dominant_frequency`
  into one object-keyed table is the intended end-state; an MVP may rely on the
  generic analyzer-named fallback table first and add the claim as the dial-tilt.
- [ ] **Validation against baselines.** On the canonical baselines
  (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598): `operator_norm` and
  `effective_rank` match a direct independent SVD of the composed matrix
  value-identically; `copying_score ∈ [0, 1]`; the composed matrix equals an
  independent reference composition (e.g. TransformerLens factored-matrix or a
  hand-rolled `W_U W_O W_V W_E`) within float tolerance. No non-baseline variant is
  used as an oracle (stale-artifact risk).

## Constraints

- **Universal instrument (invariant 1).** Spectral decomposition and the Fourier
  projection are universal lenses — they run on *any* operand. The new analyzer adds
  a new **operand** (a composed circuit), not a family-specific view. The Fourier
  `dominant_frequency` must come from the existing universal instrument over the
  `full_ov` site, not be re-derived inside the analyzer.
- **Task provides context, not views (invariant 2).** The token range `[:p]` and the
  irrep/Fourier basis are **Task context** (the TaskType supplies the basis, realized
  per Task). The composition slice to `[:p]` is family/task-supplied context, never a
  hardcode of `113`/`109` in the universal analyzer. `dominant_frequency` is an
  **instrument-conditional** invariant and carries its instrument in provenance — it
  may not pose as unconditional alongside `effective_rank`/`operator_norm`.
- **Storage internal to the API (invariant 3).** Emit through the declared schema;
  the warehouse consumes the declaration unchanged (REQ_136 pattern). Tensors are
  fetched by reference, never filtered as columns.
- **Theorem object → attributes are the research, identity is given.** The *object*
  is STABLE (the `W_U W_O W_V W_E` definition is borrowed and exact). What its
  attributes *mean* is interpretive and is carried as attributes — never as object
  identity. Do **not** promote any thresholded reading (e.g. "is a copying head") to
  a given-real column; that is an Event (Layer 7), out of scope here.
- **Atomicity / encapsulation-first.** The analyzer composes + measures the OV
  circuit only. Do not refactor `weight_spectra`, and do not pull the
  `weight_basis_projection` dependency into the new analyzer — the frequency column
  arrives via the site mechanism and a downstream join/claim.

## Notes

- **Eigenvalues are generally complex.** `W_U W_O W_V W_E` is not symmetric, so use
  `np.linalg.eig` (not `eigvalsh`); the copying score uses real parts:
  `Σ max(Re(λ), 0) / Σ |λ|`. Document this so the metric is not misread as a
  symmetric-matrix quantity. `effective_rank`/`operator_norm` use **singular**
  values (always real, ≥0) of the same composed matrix.
- **Shape/convention care.** Codebase stores `W_E` as `(d_vocab, d_model)`, `W_U`
  as `(d_model, d_vocab)`, attention `W_V` as `(n_heads, d_model, d_head)`, `W_O` as
  `(n_heads, d_head, d_model)`. The math convention `W_U W_O W_V W_E` (column-vector
  left-multiply) becomes a transpose-aware matmul/einsum on these stored layouts;
  the `(p, p)` result's `[out_token, src_token]` orientation must be fixed and
  documented (validate against the reference composition).
- **Keying vs. the data model.** The data model keys `FullOVCircuit` by
  `(variant, epoch, layer_index, head_index)`. This build is single-block (one
  layer, like `attention_patterns`), so `site="full_ov"` + `head` carry the identity;
  multi-layer `layer_index` generality is out of scope (fold a `layer`/`site`
  coordinate in later as REQ_136 did for `head`).
- **Sibling objects deferred.** `OVCircuit` (`W_O W_V`), `QKCircuit` (`W_Q^T W_K`),
  `FullQKCircuit`, and `DirectPath` (`W_U W_E`) are the same composition pattern and
  follow as fast siblings once this proves the path — they are explicitly *not* in
  REQ_152 (the data model says FullOVCircuit first). The `FullOVCircuit` FK → OVCircuit
  edge in the schema can be left logical until OVCircuit is built.
- **Spectral-instrument unification (logged, not scoped here).** The principled
  end-state is a single site-driven spectral instrument (generalize `weight_spectra`
  to consume composition sites, the "Make weight_spectra site-driven" option) so
  `effective_rank`/`operator_norm` stop being re-implemented per composed object.
  Deferred deliberately: codify it once several Layer 4 siblings exist and the
  composed-operand pattern has stabilized (the "adaptability lives in the boundaries"
  path). Park as a follow-up; do not pre-build.
- **Depends on the Track 0 discriminator fix** (`fix/discriminator-group-type-stamp`,
  data model Part VI #1) being merged to `develop` first, so the L2 weight-side vs.
  activation-side discriminator is trustworthy when the conformed claim lands.
