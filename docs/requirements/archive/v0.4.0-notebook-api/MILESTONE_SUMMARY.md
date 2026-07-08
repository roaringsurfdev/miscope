# v0.4.0 — Notebook Research API

**Date:** 2026-02-13
**Requirements:** REQ_036, REQ_037

## Summary

Added a thin convenience layer (`tdw` package) enabling notebook-based ad-hoc research without requiring knowledge of file paths or internal module structure. Researchers can load families by name, look up variants by domain parameters, and access checkpoints, artifacts, metadata, and forward passes from a single entry point.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Thin entry point + variant as hub (not monolithic session) | Researcher may want different things in different notebooks; don't impose a session model |
| Convenience methods on Variant directly (not wrapper) | Keeps the API simple — one Variant class, not two |
| Config via env vars, no config files | Single-user research tool; YAML/TOML would add complexity without value yet |
| Build system (hatchling) added to pyproject.toml | Enables `pip install -e .` so notebooks in any directory can import `tdw` |
| `make_probe()` added to ModelFamily protocol | Family-specific probe formatting; modular addition appends equals token |

## Key Files

| File | Purpose |
|------|---------|
| `tdw/__init__.py` | `load_family()`, `list_families()` entry points |
| `tdw/config.py` | `AppConfig`, `get_config()` — path resolution with env var overrides |
| `tdw/loaded_family.py` | `LoadedFamily` — variant lookup by domain parameters |
| `families/variant.py` | Convenience properties: `artifacts`, `metadata`, `model_config`, losses, `run_with_cache()`, `make_probe()`, `analysis_dataset()`, `analysis_context()` |
| `families/protocols.py` | `make_probe()` added to `ModelFamily` protocol |
| `families/json_family.py` | `make_probe()` base stub (NotImplementedError) |
| `families/implementations/modulo_addition_1layer.py` | `make_probe()` concrete implementation |
| `tests/test_config.py` | 9 tests for app configuration |
| `tests/test_notebook_api.py` | 24 tests for notebook API |
