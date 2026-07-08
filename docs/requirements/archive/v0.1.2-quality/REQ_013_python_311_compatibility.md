# REQ_013: Python 3.11+ Compatibility

## Problem Statement
The project currently requires Python 3.13+ (`requires-python = ">=3.13"`), which is restrictive for the mechanistic interpretability community. Many researchers use Python 3.11 or 3.12, especially those with older CUDA setups or institutional constraints.

Broadening compatibility to Python 3.11+ would make the workbench accessible to more users without sacrificing functionality.

## Conditions of Satisfaction
- [ ] Update `requires-python` to `">=3.11"` in pyproject.toml
- [ ] CI tests against Python 3.11, 3.12, and 3.13 (matrix)
- [ ] All tests pass on all supported Python versions
- [ ] Dependencies compatible with Python 3.11+
- [ ] Any Python 3.12+ syntax replaced with 3.11-compatible alternatives
- [ ] Documentation updated with supported Python versions

## Constraints
**Must have:**
- Full functionality on Python 3.11, 3.12, and 3.13
- No degraded experience on older versions

**Must avoid:**
- Dropping features that only work on newer Python
- Significant code complexity to support old versions

**Flexible:**
- Whether to support Python 3.10 (likely not worth the effort)
- Handling of version-specific optimizations

## Context & Assumptions
- Python 3.11 is widely used in ML/research environments
- PyTorch and TransformerLens support Python 3.11+
- Assumption: Few if any Python 3.13-specific features are used
- Assumption: Main issues will be dependency version constraints

## Decision Authority
- [ ] Make reasonable decisions and flag for review

## Success Validation
- CI matrix shows green for 3.11, 3.12, 3.13
- `uv sync` works on Python 3.11 environment
- Dashboard launches and functions on Python 3.11
- Training and analysis complete successfully on Python 3.11

---
## Notes

**Potential issues to investigate:**
- numpy>=2.4.1 may require Python 3.12+ (need to check)
- Type hint syntax (e.g., `list[str]` vs `List[str]`)
- Match statements (Python 3.10+, should be fine)

[Implementation notes will be added here]
