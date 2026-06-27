# REQ_002: Analyzer Inversion — Analyzers as Visitors

## Problem Statement

The current architecture treats Analyzers as the primary data-producing objects.
Each Analyzer runs and emits a blob of results. The mechanistic interpretability
objects (attention heads, circuits, residual stream sites, etc.) are implicit in
these blobs rather than being explicit entities that accumulate attributes over time.

This inversion needs to be corrected. Analyzers should become *visitors* — stateless
or nearly-stateless components that know how to compute and write specific attributes
onto domain objects. The domain objects defined in REQ_001 are the accumulation
targets. An Analyzer's job is to populate properties, not to own results.

This is a structural refactor, not a change in what gets computed. The analysis logic
itself is largely preserved; what changes is where the results live and how they are
organized.

**Blocks:** REQ_001 must be complete before this work begins.

## Conditions of Satisfaction

- [ ] Analyzers are refactored to accept a domain object (or collection of domain
      objects) as input and write computed attributes onto those objects, rather than
      returning standalone result blobs.
- [ ] The Analyzer Registry is updated to register analyzers against object types
      rather than against analysis categories. An Analyzer declares which object types
      it populates and which attributes it writes.
- [ ] Family-aware dispatch is implemented: when an Analyzer is run against a Variant,
      the Registry resolves which analyzers are valid for that Variant's Family and
      dispatches accordingly. Fourier-specific analyzers do not run against non-modular-
      arithmetic Variants.
- [ ] All attributes currently captured in analyzer blobs are either migrated to
      domain object attributes or explicitly designated as intermediate/scratch data
      that does not need to be persisted on the domain object.
- [ ] Existing analysis coverage is preserved. No currently-computed metric is silently
      dropped during the refactor.
- [ ] An Analyzer can be run incrementally — populating a subset of attributes on an
      existing domain object without requiring a full recomputation of all attributes.

## Constraints

**Must have:**
- Analyzers must be family-aware. The set of analyzers that run for a given Variant
  is determined by its Family, consistent with the family-scoped attribute space
  defined in REQ_001.
- The Registry must be the authoritative source for analyzer-to-object-type mappings.
  Ad hoc analyzer invocation outside the Registry should be discouraged or prevented.
- Existing analysis logic should be preserved where possible. This is a structural
  refactor, not a rewrite of computation.

**Must avoid:**
- Analyzers that write to global or cross-Variant state. Each analyzer invocation
  is scoped to a specific Variant (and checkpoint, for snapshot analyzers).
- Silent data loss during migration of existing blob results to domain object
  attributes.
- Tight coupling between Analyzer implementation and persistence layer. Analyzers
  should write to domain objects; persistence is the domain object's concern.

**Flexible:**
- Whether Analyzers are implemented as classes, functions, or another pattern
- How the Registry handles analyzer dependencies (e.g. if AnalyzerB requires
  attributes written by AnalyzerA, the ordering strategy is Engineering Claude's call)
- Migration strategy for existing blob data — full recomputation vs. in-place
  migration are both acceptable if the end state is correct

## Context & Assumptions

The Analyzer Registry is currently in its second evolution with versioning added as
a proof-of-concept. The versioning model will be replaced by REQ_003, so the refactor
here should treat versioning as a seam to be connected later rather than something
to solve within this requirement.

The key conceptual shift: an Analyzer is not a data producer. It is a computation
that knows how to populate certain properties on certain object types. The analogy
is a visitor pattern — the domain objects define the structure, the analyzers define
the computations that fill it in.

Motivating example of the before/after:

**Before:**
```
FourierAnalyzer.run(variant, checkpoint) → {
    'qk_spectral_entropy': [...],
    'v_peak_frequency': [...],
    ...
}
```

**After:**
```
FourierAnalyzer.visit(head: AttentionHead) →
    head.qk_circuit.spectral_entropy = ...
    head.qk_circuit.peak_frequency = ...
    head.ov_circuit.spectral_entropy = ...
```

The result lives on the object. The Analyzer is the mechanism, not the container.

## Decision Authority

- [ ] Propose options for review
- [x] Make reasonable decisions and flag for review
- [ ] Full autonomy to proceed

Engineering Claude owns the visitor pattern implementation, Registry refactor, and
migration strategy. Surface decisions that would significantly affect the versioning
seam (REQ_003) or that require dropping existing analysis coverage.

## Success Validation

- Running an Analyzer against a Variant populates attributes on the corresponding
  domain objects, retrievable via the domain object interface rather than via a
  separate result store.
- The Registry can enumerate, for any Family, the complete set of analyzers and the
  object type attributes each one populates.
- A domain object can report which of its attributes have been populated and which
  have not, without querying the Analyzer Registry.
- Adding a new Analyzer for an existing object type requires no changes to the domain
  object schema — it is purely additive to the Registry.

---
## Notes
[Engineering Claude adds implementation notes, alternatives considered, things to revisit]
