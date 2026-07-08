# REQ_115: Monorepo Layout (apps/ + packages/ + uv Workspace)

**Status:** Completed (merged to develop 2026-05-07)
**Priority:** High (gate for REQ_103)
**Branch:** `feature/monorepo-layout` (merged)
**Dependencies:** None upstream. **Precedes REQ_103** — PyPI publication hardening should target the final layout, not an interim one. After this lands, REQ_103 CoS items that reference `src/miscope/`, root-level `dashboard/`/`fieldnotes/`, and the `research/` restructure idea must be revised.
**Attribution:** Engineering Claude (drafted from collaborative discussion 2026-05-07)

---

## Problem Statement

The repo currently mixes the publishable library (`src/miscope/`), two consumer apps
(`dashboard/`, `fieldnotes/`), and a growing collection of documentation,
notebooks, and research artifacts at the root. A third surface — a public-facing
site for kinomorphic.com — is on the horizon and may live in this repo. Without a
clear convention for "what is a library vs. what is a deployable app vs. what is
documentation," the root will keep accumulating top-level directories and the
boundary between the publishable package and the workbench around it will keep
blurring.

REQ_103 (PyPI publication hardening) is the v1.0 gate. The cleanest mental model
for that work is: **`packages/miscope/` is a self-contained, publishable
project**, and everything else in the repo is either an app that consumes it or
documentation about it. Restructuring before REQ_103 means PyPI hardening targets
the final layout once, rather than rearranging twice.

The user has already completed the documentation consolidation under `docs/` and
introduced a `docs/requirements/staging/` folder for completed-but-unreleased
requirements. This REQ captures that work as historical record and defines the
remaining moves: apps relocation, package relocation with uv workspace setup, and
test reorganization.

---

## Goal Layout

```
packages/
  miscope/              # the publishable library
    pyproject.toml      # the project that goes to PyPI
    src/miscope/
    tests/              # unit/component tests for the package
apps/
  dashboard/            # local interactive surface (Dash)
    pyproject.toml      # required — dashboard maintains its own deps
    tests/
  fieldnotes/           # research notebook + Platform docs (Astro → GitHub Pages)
  research/             # exploratory frontend: notebooks + sketch scripts
    notebooks/          # *.ipynb research notebooks
    sketches/           # exploratory *.py (POCs, sketches, one-off analyses)
  site/                 # kinomorphic.com primary surface (future, stack TBD)
tests/
  integration/          # cross-cutting tests (e.g., dashboard ↔ miscope artifacts)
docs/                   # documentation, requirements, notes, policies (already moved)
results/                # gitignored — regenerable artifacts
scripts/                # operational scripts (run_analysis.py, create_animation.py, etc.)
pyproject.toml          # workspace root: [tool.uv.workspace] + dev tooling
PROJECT.md              # remains at root
CLAUDE.md               # remains at root
README.md               # remains at root
```

---

## Conditions of Satisfaction

### Completed work (historical record)

These moves are already on `develop` and are recorded here so this REQ is the
single point of reference for the reorganization wave:

- [x] `issues/`, `notes/`, `origins/`, `policies/`, `requirements/` consolidated
  under `docs/` (commit `083952a`).
- [x] `docs/requirements/staging/` folder created for completed-but-unreleased
  requirements; `housekeeping.md` moved to root of `docs/requirements/`
  (commit `fdbd4b6`).
- [x] Demos relocated to the root for discoverability (commit `709ed99`).
- [x] `pyproject.toml` updated to exclude jupyter notebooks from packaging
  (commit `800c983`).

### Package relocation + uv workspace

- [ ] `src/miscope/` moved to `packages/miscope/src/miscope/`. Import path
  remains `miscope.*` — no consumer-side import changes.
- [ ] `packages/miscope/pyproject.toml` created as the publishable project
  definition (project metadata, dependencies, package discovery scoped to
  `src/miscope/`). This is the file that becomes the PyPI artifact.
- [ ] Root `pyproject.toml` becomes the workspace root: `[tool.uv.workspace]`
  with `members = ["packages/*", "apps/*"]` (or explicit list), plus shared dev
  tooling configuration (ruff, pyright, pytest defaults).
- [ ] `uv sync` from the repo root resolves the workspace cleanly; editable
  install of `miscope` works for the dashboard.
- [ ] Each member that needs its own dependency surface gets its own
  `pyproject.toml`. Members without independent deps can stay as plain
  directories under the workspace.

### Apps relocation

- [ ] `dashboard/` moved to `apps/dashboard/`. Imports updated; entry points
  (`dashboard/app.py` or equivalent) still runnable.
- [ ] **`apps/dashboard/pyproject.toml` created** with the dashboard's own
  dependency surface (Dash, dashboard-specific Plotly extras, etc.) declared
  there rather than inherited from the package. Rationale: dashboard and
  miscope may diverge over time; declaring deps separately even when currently
  redundant prevents future cross-contamination of the package's dep set.
- [ ] `fieldnotes/` moved to `apps/fieldnotes/`. Astro build still produces the
  same site at the same URL.
- [ ] `apps/site/` placeholder is **not** created in this REQ — created when the
  kinomorphic.com stack is chosen, in its own requirement.

### Notebooks reorganization

`notebooks/` is effectively another exploratory frontend. The contents split
into two categories that warrant different destinations:

- [ ] **`*.ipynb` files** → `apps/research/notebooks/`. These are research
  notebooks — exploratory work paired with prose, the same kind of artifact as
  fieldnotes drafts but in computational form.
- [ ] **Exploratory `*.py` files** (sketches, POCs, one-off analyses) →
  `apps/research/sketches/`. Files in scope based on current `notebooks/`
  contents: `sketch_lissajous_fit.py`, `sketch_lissajous_v2_common_basis.py`,
  `sketch_per_group_kinks.py`, `early_gradient_analysis.py`,
  `mseed_gradient_comparison.py`, `neuron_fourier_poc.py`, `weight_space_dmd.py`.
  These are exploratory Python that happens to be authored as scripts rather
  than notebooks — same category, different file extension.
- [ ] **Operational scripts** → `scripts/` at repo root. Files in scope:
  `run_analysis.py`, `run_analysis_regression.py`, `create_animation.py`,
  `viability_certificate_calibration.py`. Console-script entry points on the
  package were considered and rejected for v1.0 — they couple operational
  tooling to the package's release cycle, and a flat `scripts/` directory
  cleanly separates "things you run against this repo" from "things you import
  from the library." Promotion to console scripts can happen later if/when a
  specific script earns its place in the public API.
- [ ] `animations/` and `exports/` directories under current `notebooks/`:
  audited for whether they are gitignored generated output (leave-and-regenerate)
  or genuine source/inputs (relocate alongside the notebooks that produce them).
- [ ] References to `notebooks/...` paths in code, fieldnotes posts, MEMORY.md,
  and CLAUDE.md updated to the new locations.
- [ ] **Root `notebooks/` directory no longer exists after this REQ lands.**
  Every file is either relocated, gitignored generated output, or explicitly
  removed. The root layout has no "catch-all" directory.

### Test reorganization

- [ ] Tests that exercise package internals move to `packages/miscope/tests/`.
- [ ] Tests that exercise dashboard internals move to `apps/dashboard/tests/`.
- [ ] Cross-cutting / end-to-end tests (e.g., dashboard rendering against real
  miscope artifacts) move to `tests/integration/` at the repo root.
- [ ] `pytest` discovery from the repo root finds all tests across all
  locations. CI runs the full set; per-package runs (`uv run --package miscope
  pytest`) are also supported.

### Pointer & configuration updates

- [ ] **CI workflows** (`.github/workflows/*.yml`) updated for new paths
  (test discovery, coverage roots, lint/typecheck targets).
- [ ] **Fieldnotes GitHub Pages deploy** workflow updated for the new
  `apps/fieldnotes/` path. Deployment URL unchanged.
- [ ] `MISCOPE_PROJECT_ROOT` default and any code that derives paths from
  `__file__` in `miscope.*` audited and updated as needed. Externally-visible
  env var contract unchanged.
- [ ] **`CLAUDE.md` project structure section** rewritten to reflect the new
  layout.
- [ ] **`README.md`** root-level layout diagram updated.
- [ ] **`MEMORY.md` and topic memory files** updated where they reference
  `src/miscope/`, root-level `dashboard/`, root-level `fieldnotes/`, etc. This
  is a sweep — not every memory entry needs path edits, but path-bearing ones do.
- [ ] **REQ_103** updated: CoS items referencing `src/miscope/`,
  `[tool.setuptools.packages.find] where = ["src"]`, and the speculative
  `research/` restructure are revised to match the realized layout.

### Validation

- [ ] `uv sync` from a fresh clone produces a working workspace.
- [ ] `uv run --package miscope pytest` runs the package's test suite.
- [ ] Dashboard launches against existing results (`uv run python -m
  apps.dashboard.app` or equivalent) and renders at least one page end-to-end.
- [ ] Fieldnotes site builds locally (`cd apps/fieldnotes && pnpm build` or
  equivalent) and the GitHub Pages deploy workflow succeeds on push.
- [ ] CI passes on the restructure PR.
- [ ] At least one notebook in `notebooks/` runs end-to-end against the
  relocated package (verifies the canonical access pattern still works).

---

## Constraints

**Must:**
- Preserve `git mv` history. Use `git mv` (not delete-and-recreate) so blame and
  log walk through the move.
- Keep the public import path stable (`import miscope.*` continues to work). The
  package directory layout changes; the package namespace does not.
- Land in a single PR. The moves are coupled (workspace setup needs the new
  paths, CI updates need both); splitting risks a half-restructured repo.

**Must avoid:**
- Renaming `miscope` itself. Only the file-system location changes.
- Introducing per-app virtualenvs. The workspace is one resolved environment;
  apps select dependencies via extras or workspace membership, not isolated envs.
- Restructuring `results/` in this REQ. That is an independent decision and
  out of scope.

**Flexible:**
- Exact workspace member declaration syntax (glob `packages/*` vs. explicit list).
- Whether `tests/integration/` becomes its own pytest config or shares the root
  config.
- Whether `apps/research/sketches/` and `apps/research/notebooks/` are separate
  subdirectories or merged into a flat `apps/research/`. Default: separate, so
  the notebook-vs-script distinction stays visible. Adjustable during
  implementation if the split feels artificial.

---

## Architecture Notes

**Why `packages/` + `apps/` rather than flat.** The convention is widely used
across modern monorepos (Nx, Turborepo, pnpm workspaces, many Python monorepos)
and carries clear semantics: `packages/*` is published or potentially-published
library code; `apps/*` is deployable or runnable surface code. A reader who has
seen this layout once knows where to look. Flat layouts force the reader to learn
this repo's specific conventions.

**Why uv workspaces.** uv natively supports workspaces (`[tool.uv.workspace]`),
which gives us:
- One resolved lockfile across all members (consistent transitive deps).
- Editable installs across members (dashboard imports miscope as a sibling,
  changes show up immediately).
- Per-member commands (`uv run --package miscope pytest`).
- Clean separation of "what gets published" (`packages/miscope/pyproject.toml`)
  from "what gets tested in CI" (workspace root).

**Why integration tests at the root.** Co-locating unit tests with the code they
test is right when the test exercises that code's internals. But cross-cutting
tests — "does the dashboard render correctly given a real miscope artifact" — do
not belong to either side; they are about the seam. Forcing them into one
package or the other obscures what they actually test. A small
`tests/integration/` is intentional, not a leftover.

**Test co-location for the package matters for PyPI.** A reviewer who clones
miscope off PyPI expects `packages/miscope/tests/` (or equivalent) to be
self-contained with the package. That is the standard shape of a publishable
Python project, and it is the shape REQ_103 will inherit.

**Site surface (kinomorphic.com) deferred.** The third app is named in the
target layout but not created here. When the stack decision lands, a separate
requirement creates `apps/site/` with the chosen tooling.

**Research as a frontend.** Treating `notebooks/` as `apps/research/` reframes
exploratory work as a deliberate consumer of the platform — the same shape as
the dashboard or fieldnotes, just authored in a different medium. Research
notebooks and sketch scripts then live under one roof regardless of whether
the author reached for `.ipynb` or `.py`. Operational scripts are a different
category (reusable tooling, not exploration) and live elsewhere.

---

## Notes

- This REQ is the gate for REQ_103 in the sense that it should land first; it
  does not block REQ_103 from being designed in parallel against the target
  layout.
- The user already completed the docs reorganization independently
  (commits `083952a`, `fdbd4b6`, `709ed99`, `800c983`). Those are recorded
  above as historical context for the reorganization wave; they are not
  in-flight work.
- Branch name suggestion: `feature/monorepo-layout`.
- Once this lands, the REQ_103 review pass is straightforward — most edits are
  path substitutions plus dropping the `research/` restructure CoS in favor of
  pointing at the realized layout.
