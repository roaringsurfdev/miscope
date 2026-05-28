# Requirements

This directory contains requirements for the Training Dynamics Workbench.

## Directory Structure

```
requirements/
├── README.md           # This file
├── active/             # Requirements currently being worked on
├── staging/            # Requirements complete, awaiting next release
├── drafts/             # Requirement ideas and research notes
├── future/             # Deferred/future requirements (not yet scheduled)
└── archive/            # Released requirements organized by milestone
    ├── v0.1.0-mvp/
    ├── v0.1.1-cuda/
    ├── v0.1.2-quality/
    ├── v0.2.0-foundations/
    ├── v0.2.1-coarseness/
    └── v0.4.0-notebook-api/
```

## Lifecycle

A requirement moves through three stages:

1. **`active/`** — In flight. Status: `Draft` or `In Progress`.
2. **`staging/`** — Implementation complete; merged to `develop` but not yet
   released. Status: `Completed`. Looking at `staging/` answers the question
   "what's about to ship in the next release?"
3. **`archive/vX.Y.Z-name/`** — Released. Moved here as part of the milestone
   release process.

`drafts/` and `future/` sit outside this flow — they're parking lots for ideas
that haven't been scheduled yet.

## Finding current status

This project moves quickly, so status is **not** tracked by a hand-maintained
snapshot here (those rot). Read it from the self-documenting sources instead:

- **In flight** — the files in [`active/`](active/).
- **About to ship** — the files in [`staging/`](staging/).
- **Released** — the milestone directories under [`archive/`](archive/) and the
  top-level `CHANGELOG.md`.
- **Parking lots** — [`drafts/`](drafts/) (ideas, research notes) and
  [`future/`](future/) (documented but unscheduled).
- **Recent activity / who changed what** — `git log`.

Each requirement's own header carries its status, priority, and effort.

## Working with Requirements

### Adding New Requirements

1. Create a new requirement file using the naming convention:
   ```
   REQ_XXX_short_description.md
   ```

2. Place in the appropriate directory:
   - `active/` - Requirements scheduled for near-term implementation
   - `future/` - Documented requirements not yet scheduled (deferred, exploratory)

3. Use the template structure:
   - Problem Statement
   - Conditions of Satisfaction
   - Constraints
   - Context & Assumptions

4. Reference by number: "Work on REQ_014"

### Completing a Requirement

When a requirement's implementation is merged to `develop`:

1. Update the requirement file's `Status:` field to `Completed`.
2. Move the file from `active/` to `staging/`.

The requirement remains in `staging/` until its release milestone moves it to
`archive/`.

### Completing a Milestone

When a set of requirements is ready to release:

1. Create archive directory: `archive/vX.Y.Z-name/`
2. Move requirements from `staging/` (and any not-yet-staged completions) to
   the new archive directory.
3. Create `MILESTONE_SUMMARY.md` in archive
4. Update `CHANGELOG.md` in project root
5. Bump version in `apps/dashboard/src/dashboard/version.py`

### Referencing Archived Requirements

For historical context on any requirement:
```
requirements/archive/v0.1.0-mvp/REQ_001_configurable_checkpoint_epochs.md
```

The milestone summary provides quick reference:
```
requirements/archive/v0.1.0-mvp/MILESTONE_SUMMARY.md
```

## Quick Reference

For current project capabilities, see:
- [CHANGELOG.md](../CHANGELOG.md) - Version history and features
- [README.md](../README.md) - Getting started guide
- [PROJECT.md](../PROJECT.md) - Project scope and vision
