# REQ_003_004: Artifact Loader

## Problem Statement
Visualization components need to load analysis artifacts independently from the pipeline. The `ArtifactLoader` provides a standalone interface for loading artifacts without requiring the full pipeline or model specification.

This enables the separation of analysis computation from visualization rendering.

## Conditions of Satisfaction
- [x] `ArtifactLoader` class takes artifacts directory path
- [x] Can load any analyzer's artifact by name
- [x] Returns dict with epochs array and data arrays
- [x] Can list available analyzers from manifest
- [x] Can get metadata (shapes, dtypes) for an analyzer
- [x] Works independently of AnalysisPipeline
- [x] Raises appropriate errors for missing artifacts

## Constraints
**Must have:**
- Standalone (no dependency on AnalysisPipeline or model_spec)
- Reads manifest.json for metadata
- Returns numpy arrays

**Must avoid:**
- Requiring pipeline instantiation
- Loading all artifacts into memory at once

**Flexible:**
- Caching strategy
- Whether to validate manifest against actual files

## Context & Assumptions
- Artifacts stored as `.npz` files in artifacts directory
- Manifest.json contains metadata for each analyzer
- Each artifact has 'epochs' array and data arrays

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Can instantiate with just artifacts directory path
- Can load any artifact without pipeline
- Returns correct structure and shapes
- Errors are clear and helpful

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code location:**
- `analysis/artifact_loader.py` (103 lines)

**Methods implemented:**
- `load(analyzer_name)` → dict with epochs and data arrays
- `get_available_analyzers()` → list of analyzer names
- `get_epochs(analyzer_name)` → sorted list of epochs
- `get_metadata(analyzer_name)` → dict with shapes, dtypes, updated_at
- `get_model_config()` → dict with prime, seed, etc.

**Design decisions:**
- Manifest is cached after first access (lazy loading)
- Clear error messages include expected paths and available options
- Works with manually created artifacts (no pipeline required)

**Tests:** 14 tests in 5 classes
- Instantiation tests
- Empty directory handling
- Pipeline integration
- Independence verification
- Manifest caching
