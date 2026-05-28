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

## Current Status

**Latest Version:** 0.4.0 (Notebook Research API)

### Active Requirements

None — all current requirements are complete.

### Drafts

| Document | Description |
|----------|-------------|
| [analysis](drafts/analysis.md) | Large cross-epoch summaries |
| [general](drafts/general.md) | Parallelization |

### Future Requirements

Requirements that have been documented but are not yet scheduled for implementation.

| Requirement | Description | Priority | Effort |
|-------------|-------------|----------|--------|
| [REQ_014](future/REQ_014_checkpoint_click_navigation.md) | Click-to-navigate checkpoint markers | Low | Medium |
| [REQ_015](future/REQ_015_checkpoint_editor.md) | Checkpoint editor using Train/Test loss curve | Medium | High |
| [REQ_017](future/REQ_017_multi_model_support.md) | Support for multiple toy models | High | High |
| [REQ_019](future/REQ_019_multiscale_activation_visualization.md) | Multi-scale activation visualization (downsampled) | Medium | Medium |
| [REQ_034](future/REQ_034_turn_detection.md) | Grokking turn detection | High | Medium |
| [REQ_035](future/REQ_035_dashboard_interaction.md) | Dashboard interaction + Dash migration | High | High |
| [REQ_128](future/REQ_128_analyzer_input_provisioning.md) | Analyzer input provisioning — lazy accessor vs eager materialization | Medium | Medium |

### Completed Milestones

| Version | Name | Date | Requirements |
|---------|------|------|--------------|
| [0.4.0](archive/v0.4.0-notebook-api/) | Notebook Research API | 2026-02-13 | REQ_036, REQ_037 |
| [0.2.1](archive/v0.2.1-coarseness/) | Coarseness Analysis | 2026-02-08 | REQ_022, REQ_023, REQ_024 |
| [0.2.0](archive/v0.2.0-foundations/) | First Foundational Release | 2026-02-06 | REQ_020, REQ_021 (a–f) |
| [0.1.2](archive/v0.1.2-quality/) | Quality | — | REQ_011, REQ_012, REQ_013 |
| [0.1.1](archive/v0.1.1-cuda/) | CUDA | — | REQ_016, REQ_018 |
| [0.1.0](archive/v0.1.0-mvp/MILESTONE_SUMMARY.md) | MVP | 2026-02-01 | REQ_001 through REQ_010 |

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
