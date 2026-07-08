# REQ_003: Object-Scoped Versioning Model

## Problem Statement

The current versioning system is analyzer-scoped: it tracks which version of an
Analyzer produced a given result blob. After the structural inversion in REQ_001 and
REQ_002, this versioning model no longer fits — there are no result blobs, only
domain objects with attributes.

The new versioning model needs to be object-scoped. The unit of versioning is the
domain object type and its attribute set, not the analyzer that computes values.
This enables a precise distinction between two fundamentally different kinds of change:

- **Additive changes** — a new attribute is added to an object type. Existing instances
  simply lack the new attribute and can be backfilled by running the relevant analyzer.
  No existing consumers break. No existing values are invalidated.

- **Breaking changes** — an existing attribute is removed, renamed, or its computation
  changes in a way that makes old and new values *incomparable*. Existing instances
  carry stale or misleading data. This is a breaking change even if the attribute name
  and type are unchanged — semantic incomparability is a breaking change.

This project is pre-v1.0.0 with no back-compat constraints, which means the versioning
model can be designed correctly rather than compatibly. The changelog starts now.

**Blocks:** REQ_001 must be complete. Can proceed in parallel with REQ_002.

## Conditions of Satisfaction

- [ ] Each object type carries a `schema_version` that increments when the set of
      defined attributes changes.
- [ ] Each attribute definition carries a `semantic_version` that increments when the
      computation producing that attribute changes in a way that makes old and new
      values incomparable.
- [ ] Each object instance carries enough version metadata to determine:
      - Whether the instance is missing attributes defined in the current schema
        (schema staleness)
      - Whether any populated attribute was computed under a semantic version that
        differs from the current one (semantic staleness)
- [ ] Additive schema changes (new attribute added) do not increment `schema_version`
      in a way that marks existing instances as broken — only as incomplete.
- [ ] Breaking semantic changes (computation changed) mark affected attribute values
      on existing instances as stale, flagging them as not comparable to values
      computed under the current semantic version.
- [ ] A changelog exists per object type recording what changed, when, and whether
      each change was additive or breaking. This changelog is the authoritative record
      and starts at v1.0.0.
- [ ] The versioning model is defined against object types and attributes, not against
      Analyzers. Analyzer versions may exist internally but are not the public-facing
      versioning surface.

## Constraints

**Must have:**
- The additive vs. breaking distinction must be first-class in the model. A schema
  diff alone is insufficient — semantic incomparability must be explicitly declared
  by the developer introducing the change, not inferred automatically.
- Schema staleness and semantic staleness must be distinguishable. They have different
  remediation paths: schema staleness is fixed by backfilling; semantic staleness may
  require recomputation or explicit acceptance of a version boundary.

**Must avoid:**
- Versioning the Analyzer as a proxy for versioning the attribute. After REQ_002,
  the Analyzer is a visitor with no persistent identity in the result store.
- Implicit semantic versioning — i.e. inferring comparability from version numbers
  alone without developer declaration. A change that looks additive (same attribute
  name and type) can be semantically breaking.
- Version metadata that is expensive to check at query time. Staleness should be
  determinable without recomputing the attribute.

**Flexible:**
- Version numbering scheme (semantic versioning, integer counters, or another
  approach — Engineering Claude's call)
- Whether changelogs are stored as code artifacts, database records, or another form
- Granularity of semantic versioning (per-attribute vs. per-analyzer-pass that
  produces a group of related attributes)

## Context & Assumptions

The current versioning system is a proof-of-concept added to the second-generation
Analyzer Registry. It can be refactored or replaced entirely — there are no back-compat
constraints at this stage.

The key insight motivating this design: **breaking changes are about comparability
across Variants, not just about schema shape.** If the computation for
`spectral_entropy` changes between when Variant A was analyzed and when Variant B was
analyzed, those two values are not comparable — even if both are stored as floats
under the same attribute name. A pure schema diff would not catch this. The versioning
model must make comparability explicit.

Three version concepts to keep distinct:

- `schema_version` on the object type — tracks what attributes are defined
- `semantic_version` on an attribute definition — tracks how the value is computed
- `analysis_version` on an instance attribute value — records which semantic version
  was current when this value was written

Staleness check: `instance.attribute.analysis_version == current semantic_version`

## Decision Authority

- [ ] Propose options for review
- [x] Make reasonable decisions and flag for review
- [ ] Full autonomy to proceed

Engineering Claude owns the versioning scheme design, changelog format, and storage
strategy. The distinction between additive and breaking changes is a product decision
— surface any cases where the boundary is ambiguous.

## Success Validation

- Given an object instance, it is possible to determine in O(1) whether any attribute
  is schema-stale (missing) or semantically-stale (computed under an old version).
- Introducing a new attribute to an object type does not require recomputation of
  existing instances to remain valid — they are incomplete, not broken.
- Changing the computation of an existing attribute and declaring it a semantic break
  causes existing instance values for that attribute to be flagged as stale.
- The changelog for each object type is human-readable and unambiguous about whether
  each entry is additive or breaking.

---
## Notes

**Overlaps REQ_145 (2026-06-19).** Object-scoped staleness substantially
re-derives REQ_145's signature-based freshness predicate (already active). The
one genuinely new idea worth lifting into REQ_145 rather than building a parallel
versioning surface: a provenance-signature change **cannot distinguish**
"recomputed, values still comparable" from "recomputed, values now incomparable."
REQ_003's *developer-declared semantic break* fills exactly that gap. Recommend
folding this concept into REQ_145 and retiring REQ_003/004 as standalone. See
`docs/object_taxonomy.md`.

[Engineering Claude adds implementation notes, alternatives considered, things to revisit]
