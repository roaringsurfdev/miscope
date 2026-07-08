# Milestone Summary: v0.1.2-quality

**Release Date:** 2026-02-01
**Theme:** Code Quality and Community Accessibility

## Overview

This release focuses on removing barriers to adoption and enforcing code quality standards. The workbench is now accessible to researchers using Python 3.11+, has zero git dependencies, and maintains enforced type safety and linting.

## Completed Requirements

### REQ_011: Enforce Ruff/Pyright in CI
- Removed `continue-on-error: true` from lint and typecheck CI jobs
- Fixed all 27 ruff violations across the codebase
- Fixed all 48 pyright type errors
- Added targeted pyright ignores for dynamic TransformerLens patterns
- **Impact:** Code quality is now enforced on every PR

### REQ_012: Remove neel-plotly Dependency
- Created `visualization/line_plot.py` with native Plotly implementation
- Supports multi-line plots, axis labels, log scales, and toggle buttons
- Updated `ModuloAdditionRefactored.py` to use new visualization utilities
- Exported `line()` function from `visualization` package
- **Impact:** Zero git dependencies, simpler installation

### REQ_013: Python 3.11+ Compatibility
- Updated `requires-python` from `>=3.13` to `>=3.11`
- Relaxed numpy constraint to `>=1.26` for TransformerLens compatibility
- Added CI matrix testing Python 3.11, 3.12, and 3.13
- Updated ruff and pyright target versions to Python 3.11
- **Impact:** Accessible to wider ML/research community

## Key Files Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | Python version, numpy constraint, tool targets |
| `.github/workflows/ci.yml` | Version matrix, removed advisory flags |
| `visualization/line_plot.py` | New file replacing neel-plotly |
| `visualization/__init__.py` | Exported `line` function |
| `ModuloAdditionRefactored.py` | Updated imports, pyright ignores |
| `FourierEvaluation.py` | Fixed type annotations |
| `dashboard/app.py` | Fixed return types, imports |

## Test Coverage

- All 156 tests pass on Python 3.11, 3.12, and 3.13
- Ruff check passes with zero violations
- Pyright passes with zero errors (with targeted ignores)

## Breaking Changes

None. This release is fully backward compatible.

## Dependencies

```toml
numpy>=1.26  # Relaxed from >=2.4.1 for TransformerLens
# neel-plotly removed entirely
```

## Next Steps

With code quality infrastructure in place and broader Python support, the workbench is ready for:
- Additional analysis capabilities (future REQs)
- Community feedback and contributions
- Extended visualization features
