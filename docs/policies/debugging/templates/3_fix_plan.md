# Fix Plan: [BUG_NNN]

## Validated Hypothesis

**Hypothesis [N]:** [Name and brief description]

**Evidence summary:**
[Brief summary of what confirmed this hypothesis]

---

## Proposed Fix

### The Change

**Description:**
[Explain what needs to be changed and why this fixes the validated issue]

**Approach:**
[Describe the minimal change strategy]

### Files to Modify

1. **File:** `[path/to/file]`
   - **Change:** [What will be modified]
   - **Lines affected:** [Approximate line numbers or range]

2. **File:** `[path/to/file]`
   - **Change:** [What will be modified]
   - **Lines affected:** [Approximate line numbers or range]

[Add more files as needed]

---

## Impact Analysis

### Scope

- **Lines changed:** [Estimated number]
- **Files modified:** [Number of files]
- **Components affected:** [List components]
- **Tests impacted:** [Which tests need to pass / will need updates]

### Threshold Check

**Does this fix exceed bug/design-flaw thresholds?**

- [ ] More than 10-15 lines changed
- [ ] More than 2-3 files modified
- [ ] Crosses component boundaries
- [ ] Breaks existing tests

**If any boxes checked:** This may be a design flaw requiring refactoring. Consult before proceeding.

---

## Risk Assessment

**Risk level:** [Low / Medium / High]

**Potential side effects:**
- [List any potential unintended consequences]
- [Consider: performance, backwards compatibility, edge cases]

**Mitigation:**
- [How will you minimize risk?]
- [What tests will verify correct behavior?]
- [Rollback plan if this doesn't work]

---

## Testing Strategy

### Tests to Run

1. **Test:** [Name/description]
   - **Purpose:** [What this verifies]
   - **Expected result:** [What should happen]

2. **Test:** [Name/description]
   - **Purpose:** [What this verifies]
   - **Expected result:** [What should happen]

[Add more tests as needed]

### Verification

- [ ] Original bug symptoms resolved
- [ ] No new bugs introduced
- [ ] All existing tests pass
- [ ] Performance acceptable
- [ ] Edge cases handled

---

## Implementation Notes

[Any specific considerations for implementing this fix]

---

## Alternative Approaches Considered

**Alternative 1:** [Brief description]
- **Why not chosen:** [Reason]

**Alternative 2:** [Brief description]
- **Why not chosen:** [Reason]

[Document any other approaches you considered and why this one is best]

---

## Approval

**Ready to implement:** [Yes / No / Needs consultation]

**Rationale:**
[Why you're confident this is the right fix, or why you need to consult]
