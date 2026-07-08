# Requirements Documentation

## Purpose

This directory contains structured requirements that define what needs to be built. Requirements follow a problem-first approach that empowers Claude to find the best solutions while maintaining clear boundaries and success criteria.

## Key Principles

**Requirements should be:**
1. **Problem-focused, not solution-prescriptive** - State what needs to be solved, not how to solve it
2. **Testable** - Include clear conditions of satisfaction that define "done"
3. **Bounded** - Explicit constraints prevent scope creep and misalignment
4. **Clear on decision authority** - Claude knows when to ask vs. when to proceed

## File Naming

```
REQ_001_descriptive_name.md
REQ_002_another_feature.md
REQ_003_bug_fix_context.md
```

Use sequential numbering and descriptive names that make requirements easy to reference in conversation.

## Working with Requirements

### Creating Requirements
1. Copy `TEMPLATE.md` to create a new requirement
2. Fill in all sections, being as specific as possible in:
   - Problem statement (the "why")
   - Conditions of satisfaction (what "done" looks like)
   - Constraints (what must/must not happen)
   - Decision authority (when to ask vs. proceed)

### Assigning Work
**Explicit direction works best:**
- "Work on REQ_003"
- "Review REQ_001 and REQ_002, then start on whichever makes sense first"
- "REQ_005 blocks REQ_006, so prioritize REQ_005"

**Optional: Add status tracking if useful:**
```markdown
**Status:** READY | BLOCKED | DRAFT | IN_PROGRESS | COMPLETE
**Blocks:** REQ_005, REQ_007
**Blocked by:** REQ_001
```

### Claude's Notes
As Claude works on requirements, observations and implementation notes are added to the `Notes` section at the bottom of each requirement file. This keeps context together without requiring separate documentation.

## Template Structure

See `TEMPLATE.md` for the standard format. Key sections:

- **Problem Statement** - What and why
- **Conditions of Satisfaction** - Testable outcomes
- **Constraints** - Must have / Must avoid / Flexible
- **Context & Assumptions** - Background needed
- **Decision Authority** - When to ask vs. proceed
- **Success Validation** - How to verify it works
- **Notes** - Implementation observations (added by Claude)

## Related Files

- `/notes/thoughts.md` - Claude's unstructured parking lot for ideas
- `/policies/debugging/` - Structured debugging process for bug fixes
- `Claude.md` - Overall collaboration framework

## Interrupt vs. Log Boundary

**Claude interrupts (stops to discuss) when:**
- Requirement conflicts with another requirement's constraints
- Architectural decision would paint us into a corner
- Clarification needed to proceed

**Claude logs for later review when:**
- Alternative approaches might be worth considering
- Potential improvements or refactoring ideas emerge
- Non-blocking observations arise

This maintains flow while preserving collaborative intelligence.
