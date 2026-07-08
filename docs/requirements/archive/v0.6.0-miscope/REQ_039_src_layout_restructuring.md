# REQ_039: Source Layout Restructuring

**Status:** Ready for implementation
**Priority:** High — do before codebase grows further
**Branch:** `feature/src-layout-restructuring` (to be created at execution time)

## Problem Statement

The project has three core API packages (`analysis/`, `families/`, `visualization/`) sitting at the project root alongside their consumers (`dashboard/`, `dashboard_v2/`, `tests/`, `notebooks/`). There is no structural distinction between the public API and its consumers. The existing `miscope/` package serves as a notebook facade but doesn't contain the packages it wraps.

As the project grows toward contributor support, this flat layout creates:
- Ambiguity about what constitutes the installable API
- No namespace isolation (any root-level directory looks like a top-level package)
- Growing git blame cost if restructured later

## Solution

Move `analysis/`, `families/`, and `visualization/` under `src/miscope/`, establishing a standard Python src-layout. The `miscope` package becomes the single installable namespace containing the full API.

## Target Structure

```
project-root/
  src/miscope/             # Installable API package (import miscope.*)
    __init__.py            # Public facade (load_family, list_families)
    config.py              # Path resolution
    loaded_family.py       # Notebook-friendly wrapper
    analysis/              # Pipeline, analyzers, artifacts, protocols
    families/              # Family registry, variants, model families
    visualization/         # Renderers, export
  dashboard_v2/            # Consumer — Dash (active, stays at root)
  tests/                   # Test suite (stays at root)
  notebooks/               # Research notebooks (stays at root)
  model_families/          # JSON config + data (stays at root)
  results/                 # Generated artifacts (stays at root)
```

## Conditions of Satisfaction

1. All 566+ tests pass after restructuring
2. `ruff check .` reports 0 errors
3. `pyright` reports 0 errors
4. `pip install -e .` succeeds and `import miscope.analysis`, `import miscope.families`, `import miscope.visualization` all work
5. Git blame is preserved via `git mv`
6. Dashboard (Dash) launches and functions correctly
7. Notebooks using `from miscope import load_family` work (update from `tdw` import)

## Constraints

- Single atomic commit for the restructure (no valid intermediate state)
- Feature branch workflow per CLAUDE.md
- Consumer package (`dashboard_v2/`) stays at project root
- `dashboard/` (Gradio) is decommissioned and removed as part of this work
- `model_families/` and `results/` are data directories, not Python packages — stay at root

---

## Detailed Analysis

### Import Inventory

All imports in the project are **absolute** (zero relative imports). This makes the restructuring a mechanical find-replace operation.

#### Internal Cross-Imports (within the 3 packages being moved)

| Source File | Import |
|-------------|--------|
| `families/implementations/modulo_addition_1layer.py` | `from analysis.library import get_fourier_basis` |
| `families/registry.py` | `from families.json_family import JsonModelFamily` + 2 more |
| `families/__init__.py` | 6 lines: `from families.*` |
| `visualization/export.py` | `from analysis.artifact_loader import ArtifactLoader` |
| `visualization/renderers/parameter_trajectory.py` | `from analysis.library.weights import COMPONENT_GROUPS` |
| `visualization/renderers/effective_dimensionality.py` | `from analysis.library.weights import ...` |
| `visualization/__init__.py` | 12 lines: `from visualization.renderers.*` |
| `analysis/__init__.py` | 4 lines: `from analysis.*` |
| `analysis/analyzers/__init__.py` | 11 lines: `from analysis.analyzers.*` |
| `analysis/analyzers/registry.py` | `from analysis.protocols import ...` |
| `analysis/analyzers/*.py` (8 files) | `from analysis.library.*` imports |
| `analysis/library/__init__.py` | 5 lines: `from analysis.library.*` |
| `analysis/library/trajectory.py` | `from analysis.library.weights import ...` |
| `analysis/pipeline.py` | `from analysis.protocols import ...` |

**Total:** ~60 internal import lines across ~28 files.

#### miscope Facade Imports

| File | Import |
|------|--------|
| `tdw/__init__.py` line 53 | `from families.registry import FamilyRegistry` (runtime, in `load_family()`) |
| `tdw/__init__.py` line 70 | `from families.registry import FamilyRegistry` (runtime, in `list_families()`) |
| `tdw/loaded_family.py` lines 7-10 | `from families.protocols`, `from families.registry`, `from families.types`, `from families.variant` |

**Note:** `from tdw.config` and `from tdw.loaded_family` (lines 28-29 of `__init__.py`) use the `tdw.` prefix — these become `from miscope.config` and `from miscope.loaded_family`.

#### Consumer Imports — Dashboard (Gradio) — REMOVED

Gradio dashboard (`dashboard/`) is decommissioned as part of this restructuring. No import updates needed.

#### Consumer Imports — Dashboard v2 (Dash)

| File | Imports |
|------|---------|
| `dashboard_v2/state.py` | `from analysis import ArtifactLoader`, `from families import FamilyRegistry, Variant` |
| `dashboard_v2/callbacks.py` | `from analysis.library.weights import ATTENTION_MATRICES`, `from visualization.renderers.*` (multiple blocks) |
| `dashboard_v2/layout.py` | `from analysis.library.weights import WEIGHT_MATRIX_NAMES`, `from visualization.renderers.landscape_flatness import FLATNESS_METRICS` |

#### Consumer Imports — Tests (~16 files)

| File | Packages Imported |
|------|-------------------|
| `test_analysis_library.py` | `analysis` |
| `test_analysis_pipeline.py` | `analysis`, `families` |
| `test_artifact_loader.py` | `analysis` |
| `test_attention_freq_analyzer.py` | `analysis` |
| `test_attention_patterns_analyzer.py` | `analysis` |
| `test_coarseness_analyzer.py` | `analysis`, `families` |
| `test_cross_epoch_analyzers.py` | `analysis`, `families`, `visualization` |
| `test_dominant_frequencies_analyzer.py` | `analysis`, `families` |
| `test_effective_dimensionality.py` | `analysis`, `families`, `visualization` |
| `test_families.py` | `families` |
| `test_landscape_flatness.py` | `analysis`, `families`, `visualization` |
| `test_modulo_addition_family.py` | `analysis`, `families` |
| `test_neuron_freq_specialization.py` | `analysis` |
| `test_notebook_api.py` | `families` |
| `test_parameter_trajectory.py` | `analysis`, `families`, `visualization` |
| `test_visualization_export.py` | `visualization` |

**Total across all consumers:** ~90 import lines across ~25 files.

#### Root-Level Scripts (Legacy)

| File | Import |
|------|--------|
| `ModuloAdditionRefactored.py` | `from visualization import line` |
| `ModuloAdditionSpecification.py` | (no imports from core packages) |
| `FourierEvaluation.py` | (no imports from core packages) |

These are early MVP scaffolding files, candidates for future removal.

#### Notebooks

`notebooks/scratch_pad.py` uses `from tdw import load_family` — **update to `from miscope import load_family`**.

### Non-Code References

| File | Reference | Action |
|------|-----------|--------|
| `pyproject.toml` line 6 | `packages = ["tdw", "families", "analysis", "visualization", "dashboard", "dashboard_v2"]` | **CRITICAL** — change to `["src/miscope", "dashboard_v2"]` |
| `model_families/.../family.json` line 5 | `"class_type": "families.implementations..."` | Update to `miscope.families.implementations...` (metadata only — NOT dynamically loaded) |
| `CHANGELOG.md` | Architecture diagrams in ~8 sections | Update directory paths |
| `CLAUDE.md` | "Project Structure" section | Update directory layout |
| `src/miscope/__init__.py` | Docstring example `from visualization import` | Update to `from miscope.visualization import` |
| `.github/workflows/ci.yml` | Commands like `ruff check .`, `pytest` | No change needed (path-agnostic) |
| `.gitignore` | No references to moved packages | No change needed |

### Key Technical Detail: `tdw/config.py` Project Root Resolution

`_resolve_project_root()` in `tdw/config.py` (line 76):
```python
current = Path(__file__).resolve().parent.parent
```

After move to `src/miscope/config.py`, `.parent.parent` yields `src/` not project root. The walking loop on lines 77-78 handles this correctly (it checks all ancestors for `pyproject.toml`), but update to `.parent.parent.parent` for clarity and to avoid an unnecessary iteration.

### Key Technical Detail: Hatch src-layout

`packages = ["src/miscope"]` tells hatchling to find the `miscope` package inside `src/`. It automatically strips the `src/` prefix — the installed package is importable as `miscope`, not `src.miscope`.

Reference: [Hatch build configuration](https://hatch.pypa.io/1.13/config/build/)

---

## Implementation Plan

### Step 1: Create Feature Branch
```
git checkout develop && git checkout -b feature/src-layout-restructuring
```

### Step 2: Remove Gradio Dashboard
```
git rm -r dashboard/
```
The Gradio dashboard is superseded by the Dash dashboard (`dashboard_v2/`). Removing it now avoids updating its imports for the new namespace.

### Step 3: Move Files with `git mv`
```
mkdir -p src/miscope
git mv tdw/__init__.py src/miscope/__init__.py
git mv tdw/config.py src/miscope/config.py
git mv tdw/loaded_family.py src/miscope/loaded_family.py
git mv analysis src/miscope/analysis
git mv families src/miscope/families
git mv visualization src/miscope/visualization
```

### Step 4: Rewrite All Imports

Apply mechanical find-replace patterns to all `.py` files in `src/miscope/`, `dashboard_v2/`, `tests/`, and root-level scripts:

| Find | Replace |
|------|---------|
| `from analysis` | `from miscope.analysis` |
| `from families` | `from miscope.families` |
| `from visualization` | `from miscope.visualization` |
| `import analysis` | `import miscope.analysis` |
| `import families` | `import miscope.families` |
| `import visualization` | `import miscope.visualization` |
| `from tdw.config` | `from miscope.config` |
| `from tdw.loaded_family` | `from miscope.loaded_family` |
| `from tdw import` | `from miscope import` |

### Step 5: Fix `config.py` Root Walk
Update `src/miscope/config.py` line 76: `.parent.parent` → `.parent.parent.parent`

### Step 6: Update `pyproject.toml`
```toml
packages = ["src/miscope", "dashboard_v2"]
```

### Step 7: Update `family.json` Metadata
Update `class_type` string to use `miscope.families.implementations...` prefix.

### Step 8: Reinstall
```
pip install -e .
```

### Step 9: Verify
1. Python smoke test: `from miscope.analysis import ArtifactLoader` etc.
2. `ruff check .` — 0 errors
3. `pyright` — 0 errors
4. `pytest` — all tests pass

### Step 10: Update Documentation
- `CLAUDE.md` project structure section
- `CHANGELOG.md` (restructuring note for next release)
- Docstrings in `src/miscope/__init__.py` and `src/miscope/visualization/__init__.py`

### Step 11: Commit
Single atomic commit with descriptive message. Request merge approval to `develop`.

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Missed import | Low | ruff catches unused/broken imports; full test suite validates |
| pyproject.toml misconfiguration | Medium | Smoke test immediately after `pip install -e .` |
| Config path resolution | Low | Walking loop in `_resolve_project_root()` handles any depth |
| No valid intermediate state | N/A | Single atomic commit; trivial revert |
