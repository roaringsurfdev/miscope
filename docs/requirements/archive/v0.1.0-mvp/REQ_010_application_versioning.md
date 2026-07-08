# REQ_010: Application Versioning

## Problem Statement
As the Training Dynamics Workbench evolves, we need a way to track which version is running. This facilitates bug reporting, ensures users know what features are available, and provides clear markers for releases.

A version number displayed in the dashboard helps users communicate issues precisely and helps developers correlate bugs with specific code states.

## Conditions of Satisfaction
- [x] Version number follows MAJOR.MINOR.BUILD format
- [x] Version displayed prominently in the dashboard UI
- [ ] BUILD incremented with each commit/change
- [ ] MINOR and MAJOR increments made via explicit planning decisions
- [x] Version accessible programmatically (e.g., `from dashboard import __version__`)
- [x] Starting version: 0.1.0

## Constraints
**Must have:**
- Semantic versioning format: MAJOR.MINOR.BUILD
- Visible in dashboard header or footer
- Single source of truth for version (one file to update)

**Must avoid:**
- Manual version tracking that could get out of sync
- Version hidden in hard-to-find location

**Flexible:**
- Exact placement in UI (header, footer, about section)
- Whether to include git commit hash for development builds
- Whether BUILD auto-increments or is manually updated

## Context & Assumptions
- MAJOR: Breaking changes, significant architectural shifts
- MINOR: New features, notable enhancements
- BUILD: Bug fixes, small improvements, incremental changes
- Current state represents version 0.1.0 (initial MVP)
- Assumption: Manual BUILD increment is acceptable for MVP (auto-increment deferred)

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Dashboard shows version number clearly
- Can import version from Python: `from dashboard import __version__`
- Version updates are reflected in UI after restart
- Bug reports can reference specific version numbers

---
## Notes

**Versioning scheme:**
- 0.x.x: Pre-release / MVP phase
- 1.0.0: First stable release (all MVP requirements complete and tested)

**Post-MVP enhancements:**
- Auto-increment BUILD on commit via pre-commit hook
- Include git commit short hash in development builds (e.g., 0.1.5+abc123)
- Version history / changelog accessible from dashboard

## Implementation Notes

**Completed: 2026-02-01**

### Files Created/Modified
- `dashboard/version.py` - Single source of truth for version (new)
- `dashboard/__init__.py` - Exports `__version__` for package-level access
- `dashboard/app.py` - Displays version in dashboard header
- `tests/test_dashboard.py` - Added `TestVersioning` class with 5 tests

### Architecture
```
dashboard/version.py          # __version__ = "0.1.0" (single source of truth)
       ↓
dashboard/__init__.py         # from dashboard.version import __version__
       ↓
dashboard/app.py              # Displays "Training Dynamics Workbench v0.1.0"
```

### Usage
```python
# Package-level import (recommended)
from dashboard import __version__

# Direct module import
from dashboard.version import __version__
```

### Version Display
The version appears in the dashboard header as:
```
# Training Dynamics Workbench v0.1.0
```

### Deferred to Post-MVP
- Auto-increment BUILD on commit via pre-commit hook
- Git commit hash in development builds (e.g., 0.1.5+abc123)
- Version history / changelog in dashboard
