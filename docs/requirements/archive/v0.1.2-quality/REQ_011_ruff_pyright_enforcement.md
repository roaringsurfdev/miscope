# REQ_011: Ruff and Pyright Enforcement

## Problem Statement
The CI workflow currently runs ruff (linting/formatting) and pyright (type checking) as advisory checks that don't block merges. This was intentional to allow the MVP to ship without blocking on code style issues.

However, to maintain code quality as the project grows, these checks should eventually become blocking. This requires cleaning up existing violations first, then enabling enforcement.

## Conditions of Satisfaction
- [ ] All existing ruff violations fixed or explicitly ignored with rationale
- [ ] All existing pyright errors resolved or explicitly typed as ignored
- [ ] CI workflow updated to remove `continue-on-error: true` from lint/typecheck jobs
- [ ] PRs with lint or type errors are blocked from merging
- [ ] Documentation updated to reflect coding standards

## Constraints
**Must have:**
- Zero ruff errors on enforcement
- Zero pyright errors on enforcement
- Clear rationale for any `# noqa` or `# type: ignore` comments

**Must avoid:**
- Blanket ignores that hide real issues
- Overly strict rules that create friction without value

**Flexible:**
- Which ruff rules to enable beyond current set
- Pyright strictness level (basic vs strict)
- Whether to add pre-commit hooks for local enforcement

## Context & Assumptions
- Current ruff config: `select = ["E", "F", "I", "UP"]` with `ignore = ["E501"]`
- Current pyright config: `typeCheckingMode = "basic"`
- Assumption: Most violations are import ordering and minor style issues
- Assumption: Type errors may require adding type hints to existing code

## Decision Authority
- [ ] Make reasonable decisions and flag for review

## Success Validation
- CI lint job passes without `continue-on-error`
- CI typecheck job passes without `continue-on-error`
- New PRs with violations are blocked
- Existing codebase is clean

---
## Notes

[Implementation notes will be added here]
