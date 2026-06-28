# REQ_001: Mechanistic Interpretability Domain Object Schema

## Problem Statement

The current data model has no first-class representation of mechanistic interpretability
objects. Analysis results are stored as blobs produced by Analyzer objects, which means
the interpretability objects themselves — attention heads, circuits, residual stream sites,
MLP layers — are implicit in the output rather than explicit in the schema.

This creates a disorganized accumulation of analysis data with no coherent structure to
design against, version against, or query against. The objects that *should* be central
to the data model are instead downstream artifacts of the analysis process.

The goal is to define a schema of first-class domain objects grounded in the mechanistic
interpretability literature, specifically the compositional vocabulary established in
"A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021). These objects
are not optimization metrics or performance indicators. They are the natural units of
mechanistic analysis — the things the model *is* and *does*, not how well it *performs*.

## Conditions of Satisfaction

- [ ] A `Family` object exists that defines architecture, task, and the valid set of
      interpretability object types for Variants belonging to it. Family-specific logic
      (e.g. Fourier mode structure for modular arithmetic) lives here and is not forced
      onto other Family types.
- [ ] A `Variant` object exists with a primary key of `(prime, seed, data_seed)` for
      modular addition families. Variant identity is treated as irreducible — Variants
      with different seeds are different models, not noisy samples of the same model.
- [ ] The schema distinguishes three temporal object types:
      - **Snapshot objects** — keyed by `(Variant, checkpoint)`, carrying point-in-time
        measurements
      - **Trajectory objects** — keyed by `(Variant)`, carrying properties of the path
        through weight space across checkpoints
      - **Event objects** — keyed by `(Variant)`, carrying scalars that locate transitions
        or phase changes in training time
- [ ] The following interpretability objects are defined as first-class schema entities,
      with their temporal type noted:
      - `ResidualStreamSite` — snapshot; one per layer boundary per checkpoint
      - `AttentionHead` — snapshot; one per head per checkpoint
      - `QKCircuit` — snapshot; sub-object of AttentionHead
      - `OVCircuit` — snapshot; sub-object of AttentionHead
      - `MLPLayer` — snapshot; one per layer per checkpoint
      - `EmbeddingMatrix` and `UnembeddingMatrix` — snapshot; boundary objects
      - `HeadTrajectory` — trajectory; one per head per Variant
      - `ParameterGroupTrajectory` — trajectory; one per parameter group per Variant
      - `SpecializationEvent` — event; one or more per Variant, marking phase transitions
- [ ] Derived objects (e.g. composed virtual weights W_Q^T W_K, W_O W_V) are
      representable in the schema. These are computed from base parameters rather than
      read directly from checkpoints, and the schema accommodates that distinction.
- [ ] The valid attribute space for interpretability objects is family-scoped. Attributes
      specific to modular arithmetic (e.g. Fourier spectrum per head) are defined at the
      `ModuloAddition1LayerFamily` level and are not present on objects belonging to
      other family types.
- [ ] Cross-Variant aggregation is not baked into the data model. Comparison across
      Variants is a query-time operation, not a schema-level assumption.

## Constraints

**Must have:**
- Object definitions grounded in the Mathematical Frameworks paper vocabulary. QK and OV
  circuits are the canonical example: they are not arbitrary matrix decompositions but
  argued first-class units of transformer computation.
- Clean separation between snapshot, trajectory, and event object types. These have
  different keys and different semantics and must not be collapsed into a single temporal
  model.
- Family inheritance respected. `ModuloAddition1LayerFamily` extends `BaseFamily`.
  Family-specific object types and attributes must not leak into the base.

**Must avoid:**
- Conflating interpretability objects with performance or optimization objects. Nothing
  in this schema is about loss, accuracy, or training health. Those live elsewhere.
- Making Variant identity compositional in ways that imply comparability. A Variant is
  a unique model instance. The schema should not encourage averaging or aggregating
  across seeds as if they were replicates.
- Defining a universal attribute space that tries to accommodate all possible analysis
  types globally. Attribute space is family-scoped.

**Flexible:**
- Exact representation of sub-object relationships (e.g. whether QKCircuit is a nested
  object or a related entity with a foreign key to AttentionHead)
- Whether derived objects (composed virtual weights) are materialized or computed
  on demand
- ORM, document store, or other persistence strategy — Engineering Claude owns this

## Context & Assumptions

The Mathematical Frameworks paper (Elhage et al., 2021) is the primary domain reference.
Key concepts:

- The **residual stream** is a shared communication channel. Every component reads from
  and writes to it. It is not owned by any single component.
- The **QK circuit** (W_Q^T W_K) governs attention patterns — which tokens attend to
  which. Its eigenstructure reveals what features are being compared.
- The **OV circuit** (W_O W_V) governs what information is moved and how it is
  transformed. Its eigenstructure reveals what the attended tokens write into the
  residual stream.
- **Composed virtual weights** (products of matrices across heads and layers) are the
  actual computational primitives for multi-layer circuits. They are derived objects but
  mechanistically important.
- **Attention heads** are composite objects — each carrying both a QK and OV circuit —
  but the head as a unit also has functional identity (e.g. induction head, previous-
  token head) that is a property of the head, not of either sub-circuit alone.

For the modular addition task specifically, the model (p=101, 1-layer, 4-head
transformer) learns to represent modular arithmetic via Fourier modes. A small number
of frequencies are selected during training in a phase transition that is visible in
the Fourier spectra of the QK and V matrices. This Fourier structure is the primary
interpretability signal for this family and motivates the family-scoped attribute space.

The existing codebase has:
- `BaseFamily` / `ModuloAddition1LayerFamily` inheritance already in place
- `Family.generate_dataset()` and `Family.context_data` (including Fourier mode
  generation) already implemented
- Variant identity as `(prime, seed, data_seed)` already established
- Analyzer Registry with versioning (v2, proof-of-concept stage)

Engineering Claude should review the existing schema and Analyzer Registry before
proposing the new object hierarchy, to ensure the design accounts for what already
exists.

## Decision Authority

- [ ] Propose options for review
- [x] Make reasonable decisions and flag for review
- [ ] Full autonomy to proceed

Engineering Claude owns the object hierarchy design and persistence strategy. Decisions
about sub-object representation, materialization of derived objects, and schema
implementation should be made and flagged for review rather than blocked on approval.
Architectural decisions that would significantly constrain REQ_002, REQ_003, or REQ_004
should be surfaced before implementation.

## Success Validation

- A new Variant can be instantiated and its interpretability object slots are
  well-defined and queryable, even before any analyzers have populated them.
- The schema can represent the difference between "this attribute has not been computed
  yet" and "this object type does not support this attribute in this family."
- Given a checkpoint, it is possible to retrieve all snapshot objects for that Variant
  at that checkpoint without querying analyzer output blobs.
- The Fourier spectrum attributes on QKCircuit and OVCircuit are present and typed
  for ModuloAddition1LayerFamily Variants, and absent for Variants of other families.

---
## Notes

**Repo-grounded taxonomy (2026-06-19):** `docs/object_taxonomy.md` is the
in-repo form of this requirement, built from the live registry
(`miscope.registry.index()`). Key reconciliations the outside-Claude draft
couldn't see:

- The objects **already exist as coord-signatures** in the registry; the schema
  is ~60% built. Naming them is the deliverable, not constructing a new store
  (objects = reader projections over the warehouse, never a second persistence
  path — invariant #3).
- **`FrequencyMode` is transverse**, not an attribute of QK/OV. Its identity is
  `frequency` but its attributes live on neuron-/group-/site-keyed rows. It is
  the family's true central object and the join target across components.
  `FrequencyMode` and `FrequencyGroup` are two views of one mechanism, currently
  unrelated (`group_freqs` is the missing edge).
- **Events must key `(variant, object-ref)`, not `(variant)`** — the p109 late
  reorganization is silent in aggregate diagnostics but visible per-head/per-freq;
  a Variant-keyed event *is* the aggregate that hides it.
- The most canonical community object (**AttentionHead**) is the least
  first-class thing we have (2 attributes; head axis buried in a `patterns`
  blob). **QKCircuit/OVCircuit are empty.** These are the forward agenda.
- New objects the escape work forced open: **DecisionQuery** (REQ_151 already
  started it).
- **DECISION (2026-06-19):** the overloaded `site` coord splits into
  **`ActivationSite`** and **`WeightMatrix`** (field-native names, not a coined
  symmetric `-Site` pair). Earned on four coinciding seams (attribute kind,
  SNR/energy guard, DMD variant, *and* it resolves the
  `circularity`/`fisher`/`snr` double-declaration collision). Subsumes the
  earlier `WeightGroup` proposal.

[Engineering Claude adds implementation notes, alternatives considered, things to revisit]
