# Debugging Policy

## Purpose

This policy provides a structured approach to debugging that emphasizes:
- Systematic hypothesis generation and testing
- Minimal code changes to prevent destabilization
- Clear documentation for traceability and handoff
- Explicit validation before implementation

This is an **interim strategy** to create scaffolding for efficient and effective debugging with minimum risk to the codebase.

## When to Use This Policy

Use this structured debugging process for all bug fixes. This does NOT apply to:
- Feature development
- Refactoring (see guidelines below for distinguishing bugs from design flaws)
- Exploratory code changes

## The Debugging Process

### Overview

1. **Bug Report** - Document symptoms and reproduction steps
2. **Hypothesis Generation** - List possible causes (1-5 theories)
3. **Evidence Search** - Test hypotheses by examining code
4. **Fix Planning** - Design minimal change (only after validation)
5. **Implementation** - Execute fix and document results

### Key Principles

- **One hypothesis at a time** - Test theories individually to avoid confusion
- **Evidence before fixes** - Never change code without validated hypothesis
- **Append-only hypotheses** - Add new theories to the end of the list, don't resort
- **Test efficiency** - Choose which hypothesis to test based on speed/likelihood, not list order
- **Log discovered issues** - Document unrelated problems found during investigation as separate bugs

### Directory Structure

For each bug, create:

```
/debugging/
  BUG_NNN_name/
    0_bug_report.md        # Symptoms, reproduction, expected vs actual
    1_hypotheses.md        # Competing theories (append-only list)
    2_evidence_search.md   # What we looked for, what we found
    3_fix_plan.md          # Proposed minimal change
    4_implementation.md    # Execution record, commit ref, results
    discovered_issues.md   # Evidence X findings (optional)
```

Use templates from `/policies/debugging/templates/` for each file.

## Step-by-Step Workflow

### Step 0: Bug Report
- Create `0_bug_report.md` using template
- Document symptoms, how discovered, reproduction steps
- Clarify expected vs actual behavior

### Step 1: Generate Hypotheses
- Create `1_hypotheses.md` using template
- List 1-5 possible causes
- For each hypothesis, predict what evidence would support it
- Add new hypotheses to end of list as they arise (append-only)

### Step 2: Search for Evidence
- Create `2_evidence_search.md` using template
- Choose which hypothesis to test based on efficiency (quick validation, low complexity)
- Document what code/files you examine
- Record what you actually find
- Mark hypothesis as: Validated / Invalidated / Inconclusive

**If you find Evidence X (unexpected finding):**
- Ask: Does this explain the bug symptoms?
  - If YES: This might be the actual cause (new hypothesis)
  - If NO: This is likely a different bug - log it in `discovered_issues.md` and continue

**When to stop:**
- ✅ Evidence validates a hypothesis → Proceed to Step 3
- ❌ All hypotheses invalidated → Stop and consult
- ⚠️ Inconclusive results → Generate new hypotheses or consult

### Step 3: Plan the Fix
- Create `3_fix_plan.md` using template
- Design the **minimal change** that fixes the validated issue
- Estimate impact (lines, files, components, tests)

**Bug vs Design Flaw Thresholds:**
If your fix exceeds these limits, flag for consultation:
- More than 10-15 lines changed
- More than 2-3 files modified
- Crosses component boundaries
- Breaks any existing tests

These thresholds suggest a design flaw requiring refactoring, not a bug fix.

### Step 4: Implement
- Create `4_implementation.md` using template
- Execute the planned fix
- Document which hypothesis/plan was used
- Record commit hash/reference
- Document test results
- If fix doesn't work: Return to Step 2 or consult

## Important Disciplines

### Never Change Code Without Validated Evidence
If you can't find evidence supporting your hypothesis, **stop**. Don't proceed to making changes. The hypothesis is probably wrong.

### Minimal Changes Only
Every character changed in a codebase can have non-local, unpredictable effects. Be conservative. Use:
- Proper test coverage
- Ability to rollback changes
- Small, targeted modifications

### Log, Don't Fix, Unrelated Issues
If you discover a different problem (Evidence X) while investigating:
1. Document it in `discovered_issues.md`
2. Decide if it's the actual cause of current bug
3. If not, treat it as a separate debugging exercise
4. Don't "fix while you're in there" - this causes scope creep

### Hypothesis Testing Order
Test hypotheses in order of efficiency, not list order:
- Quick to validate/invalidate
- Minimal expected impact if wrong
- Most likely cause (balanced with above factors)

## When to Consult

Stop and consult (hand off for fresh perspective) when:
- All hypotheses have been invalidated
- Fix plan exceeds bug/design-flaw thresholds
- You're stuck or uncertain about next steps
- You discover the issue is "tip of the iceberg" (bigger problem)

## Success Criteria

A successful debugging session produces:
- Complete documentation trail from symptom → fix
- Validated hypothesis with supporting evidence
- Minimal code changes that fix the issue
- Passing tests
- Clear commit message referencing bug number

## Templates

All templates are located in `/policies/debugging/templates/`:
- `0_bug_report.md`
- `1_hypotheses.md`
- `2_evidence_search.md`
- `3_fix_plan.md`
- `4_implementation.md`
