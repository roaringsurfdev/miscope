# REQ_036: Application Configuration

**Status:** Draft
**Priority:** High (dependency for notebook API and future multi-environment support)
**Dependencies:** None
**Last Updated:** 2026-02-13

## Problem Statement

Configuration values like the results directory path and model families directory are currently passed as arguments or hardcoded at the point of use. This creates two problems:

1. **Notebook friction**: A researcher starting a notebook must know and specify filesystem paths to reach model families and results. These paths are implementation details that shouldn't leak into research workflows.

2. **Scattered defaults**: Different entry points (dashboard, CLI, tests) each resolve paths independently. There is no single source of truth for "where does this application store its data?"

## Design

### Configuration Module

New file: `config.py` (project root)

A lightweight settings module that provides default paths and allows overrides. No framework, no YAML/TOML files — just a Python module with sensible defaults relative to the project root.

```python
from config import get_config

cfg = get_config()
cfg.results_dir        # Path to results/
cfg.model_families_dir # Path to model_families/
cfg.project_root       # Resolved project root
```

### Path Resolution

Default behavior: resolve paths relative to the project root directory (the directory containing `config.py`). This works for the common case — running from the project directory or from a notebook in a `notebooks/` subdirectory.

### Override Mechanism

Environment variables provide overrides without touching code:

```bash
export TDW_RESULTS_DIR=/path/to/results
export TDW_MODEL_FAMILIES_DIR=/path/to/model_families
```

Environment variables take precedence over defaults. This supports:
- Non-standard directory layouts
- Future multi-environment deployment
- CI/test configurations

### What This Does NOT Include

- No config file format (YAML, TOML, JSON) — premature for a single-user research tool
- No per-user settings — environment variables suffice
- No runtime config mutation — config is read-once at import time
- No UI for configuration — this is infrastructure, not a feature

## Scope

**This requirement covers:**
1. `config.py` module with project path resolution
2. Environment variable override support
3. Integration: FamilyRegistry, dashboard, and notebook API use config for default paths
4. Tests

**This requirement does not cover:**
- Config file formats (YAML, TOML)
- UI-accessible settings
- Per-user configuration profiles
- Runtime configuration changes

## Conditions of Satisfaction

### Configuration
- [ ] `get_config()` returns an object with `results_dir`, `model_families_dir`, and `project_root`
- [ ] Default paths resolve correctly when running from project root
- [ ] Environment variables (`TDW_RESULTS_DIR`, `TDW_MODEL_FAMILIES_DIR`) override defaults
- [ ] Config is importable from notebooks without path manipulation

### Integration
- [ ] FamilyRegistry can be instantiated with no arguments (uses config defaults)
- [ ] Dashboard uses config for default paths instead of hardcoded values

### Tests
- [ ] Default path resolution produces valid paths
- [ ] Environment variable overrides work
- [ ] Config works when imported from a subdirectory (e.g., `notebooks/`)

## Constraints

**Must have:**
- Zero-dependency (standard library only)
- Works from project root and from subdirectories (notebooks/)
- Environment variable overrides

**Must avoid:**
- External config file formats that add dependencies
- Breaking existing code that passes paths explicitly (explicit paths still work)
- Over-engineering for hypothetical multi-user or cloud scenarios

**Flexible:**
- Exact resolution strategy for project root (file-relative vs cwd vs git root)
- Config object type (dataclass, NamedTuple, plain class)
- Whether config is a singleton or created per-call
- Additional config keys beyond the initial two paths
