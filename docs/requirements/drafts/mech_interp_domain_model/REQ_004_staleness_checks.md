# REQ_004: Staleness Detection Against the New Versioning Model

## Problem Statement

The current staleness checking logic is built against the analyzer-scoped versioning
model — it checks whether a result blob was produced by a recent enough Analyzer
version. After REQ_002 and REQ_003, there are no result blobs and the versioning
model is object-scoped. Staleness checking needs to be reimplemented against the
new model.

Two distinct staleness conditions must be detectable and distinguishable:

- **Schema staleness** — an object instance is missing one or more attributes that
  are currently defined for its object type. This is recoverable by running the
  relevant Analyzer(s) to populate the missing attributes.

- **Semantic staleness** — an object instance has a populated attribute that was
  computed under an older semantic version, making it not comparable to values
  computed under the current version. This may require recomputation or explicit
  acknowledgment of a version boundary.

These two conditions have different causes, different severities, and different
remediation paths. The staleness system must not conflate them.

**Blocks:** REQ_003 must be complete before this work begins.

## Conditions of Satisfaction

- [ ] A staleness check can be run against any domain object instance and returns
      a structured report distinguishing schema-stale attributes from semantically-
      stale attributes.
- [ ] Schema staleness is reported as: "attribute X is defined for this object type
      but has not been computed for this instance."
- [ ] Semantic staleness is reported as: "attribute X was computed under semantic
      version N but the current version is M — values may not be comparable to
      instances analyzed under version M."
- [ ] The staleness check does not recompute any attribute values. It is a metadata
      inspection only.
- [ ] A bulk staleness check can be run across all instances of an object type for
      a given Variant (or Family), returning a summary of which attributes are stale
      and for how many instances.
- [ ] The system surfaces a recommended remediation for each staleness condition:
      schema staleness → run Analyzer(s) X, Y; semantic staleness → recompute or
      accept version boundary.
- [ ] Staleness checking integrates with the Analyzer Registry from REQ_002 so that
      the recommended Analyzer(s) for backfilling a schema-stale attribute are
      automatically identified.

## Constraints

**Must have:**
- The additive/breaking distinction from REQ_003 must be respected. Schema staleness
  (additive change) and semantic staleness (breaking change) must produce different
  staleness report entries and different recommended remediations.
- Staleness checks must be fast enough to run routinely — before analysis queries,
  before cross-Variant comparisons, or as part of a data health dashboard. They
  should not be expensive operations.

**Must avoid:**
- Conflating schema staleness with semantic staleness in the report output or in
  the remediation recommendations.
- Staleness checks that require recomputing attribute values to determine staleness.
  All staleness information must be derivable from version metadata alone.
- A single global "stale / not stale" boolean. The report must be attribute-level.

**Flexible:**
- Whether staleness checking is exposed as a method on domain objects, a standalone
  utility, or both
- Report format (structured dict, dataclass, printed summary — Engineering Claude's
  call based on how it integrates with the rest of the platform)
- Whether a "soft stale" / "hard stale" distinction is useful for semantic staleness
  based on how significant the version delta is

## Context & Assumptions

Staleness detection is the operational surface of the versioning model. It is how
a researcher knows whether the data they are looking at is current, incomplete, or
potentially misleading.

The two staleness types have different practical implications:

Schema staleness is benign and common. As new analyzers are added (REQ_002 makes
this purely additive), existing Variants will naturally be schema-stale for new
attributes. This is expected and the remediation is straightforward: run the new
Analyzer against existing checkpoints.

Semantic staleness is more serious. It means that a value the researcher might
use in a cross-Variant comparison was computed differently than the current method.
This is easy to miss if staleness checking only looks at attribute presence rather
than semantic version. The version boundary must be visible.

Motivating scenario: the computation for `spectral_entropy` on `QKCircuit` is
updated to use a different normalization. Existing instances have the old value.
New instances have the new value. A researcher comparing spectral entropy across
Variants trained at different times would unknowingly be comparing incomparable
values unless semantic staleness is surfaced.

## Decision Authority

- [ ] Propose options for review
- [x] Make reasonable decisions and flag for review
- [ ] Full autonomy to proceed

Engineering Claude owns the staleness report design, API surface, and integration
with the Analyzer Registry. Surface any cases where the boundary between schema
and semantic staleness is ambiguous in practice.

## Success Validation

- Running a staleness check against a Variant that has never been analyzed returns
  a report showing all defined attributes as schema-stale, with recommended
  Analyzers for each.
- Running a staleness check against a fully-analyzed Variant returns a clean report
  with no stale attributes.
- After a semantic version bump on an attribute, running a staleness check against
  an existing instance flags that attribute as semantically stale, not schema-stale.
- A bulk staleness check across all Variants in a Family correctly identifies which
  Variants were analyzed before and after a semantic version boundary.

---
## Notes
[Engineering Claude adds implementation notes, alternatives considered, things to revisit]
