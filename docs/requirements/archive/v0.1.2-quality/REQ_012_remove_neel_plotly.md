# REQ_012: Remove neel-plotly Dependency

## Problem Statement
The project currently depends on `neel-plotly`, a git-based dependency from Neel Nanda's repository. This dependency was inherited from the original notebook exploration and creates several issues:

1. **Installation complexity**: Git dependencies require special handling in some environments
2. **Stability risk**: External repo changes could break our builds
3. **Unnecessary dependency**: The workbench has its own visualization renderers that may not need neel-plotly

Removing this dependency simplifies installation and reduces external risk.

## Conditions of Satisfaction
- [ ] Identify all usages of neel-plotly in the codebase
- [ ] Replace or remove each usage with native plotly or custom code
- [ ] Remove neel-plotly from pyproject.toml dependencies
- [ ] Remove neel-plotly from uv.sources
- [ ] All tests pass without neel-plotly
- [ ] Visualizations maintain equivalent functionality

## Constraints
**Must have:**
- All existing visualization functionality preserved, especially within ModuloAdditionRefactored.py. This notebook serves as a guide for future visualizations at this time.
- No regression in visual output quality

**Must avoid:**
- Breaking changes to visualization APIs
- Significant increase in code complexity

**Flexible:**
- Whether to inline needed utilities or rewrite from scratch
- Exact visual styling (minor differences acceptable)

## Context & Assumptions
- neel-plotly provides convenience functions for plotly visualizations
- The workbench already has custom renderers in `visualization/renderers/`
- Assumption: Usage is limited to a few utility functions
- Assumption: Native plotly can achieve equivalent results

## Decision Authority
- [ ] Make reasonable decisions and flag for review

## Success Validation
- `neel-plotly` does not appear in pyproject.toml
- `uv sync` completes without fetching neel-plotly
- All visualization tests pass
- Dashboard visualizations render correctly

---
## Notes

[Implementation notes will be added here]
