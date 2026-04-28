# REQ_103: PyPI Publication Hardening (Publishable Library + Templates + Public-Facing Surfaces)

**Status:** Draft
**Priority:** High
**Branch:** TBD
**Dependencies:** REQ_106 (architecture principles must be in place before publication exposes them externally); REQ_107 (registry is the public discoverability surface); REQ_097, REQ_098, REQ_099, REQ_101 — all should be substantially in place before publication. REQ_108 (Publication Surface) is the companion data-publication primitive; REQ_103 covers the package + repo + docs surfaces, REQ_108 covers the Parquet/DuckDB/Releases data surface. REQ_100 (HFHubSource) is deferred and not on the v1.0 critical path. This REQ is the v1.0 gate.
**Attribution:** Engineering Claude

---

## Problem Statement

The miscope library will be published to PyPI as the entry point for external
researchers. The publication model is **GitHub Pages (fieldnotes site) for
articles + queryable derived data** (REQ_108), with the PyPI package as the
tool a reviewer can install if they want to extend the analysis locally.
Raw-artifact publication via HuggingFace Hub is deferred until demand surfaces;
the v1.0 publication path does not depend on it.

Publication readiness is a distinct concern from internal cleanup. It requires:

- A clean public API surface (what gets exported as `miscope.*`).
- A package definition (`pyproject.toml`) and a release workflow so PyPI
  publication is reproducible from a tag.
- Schema versioning for artifacts (so notebook code can declare which
  artifact format it expects). Internal artifacts and published Parquet
  bundles (REQ_108) carry stronger immutability guarantees than working
  intermediates.
- **Canonical access templates** for notebooks and dashboard pages — the
  user's stated deliverable that addresses scattered access paths in
  the existing codebase.
- **A polished public-facing surface.** The repo currently mixes the platform
  with the research workbench that produced it (notebooks, notes, in-flight
  requirements). For an independent researcher without existing social
  capital, the first-impression cost of unstructured workbench bleed-through
  is real. The repo, the PyPI page, and the fieldnotes site are the three
  surfaces a reviewer encounters, and each needs deliberate framing.

Without these, a researcher landing on the published code has multiple paths
to the same data, no clear "this is the right way" guide, and risks finding
inconsistencies that erode trust.

---

## Conditions of Satisfaction

### Public API surface

- [ ] `pyproject.toml` at repo root defines:
  - Project metadata (name, version, description, authors, license).
  - `[tool.setuptools.packages.find] where = ["src"]` — package contents
    scoped to `src/miscope/` only. Nothing under `dashboard/`, `notebooks/`,
    `research/`, `fieldnotes/`, or `results/` ships in the wheel.
  - Required dependencies (numpy, torch, transformer-lens, sklearn,
    plotly, pandas, pyarrow, duckdb).
  - Optional extras: `dev` (test/lint tools), `dashboard` (dash + dashboard
    deps), `gpu` (cuML / cupy when REQ_098+ adds backend abstraction),
    `hf` (huggingface_hub — for if/when REQ_100 lands).
- [ ] `miscope/__init__.py` exports the documented public API:
  - Re-export `miscope.core.*` (vocabulary types, enums).
  - Re-export the variant / family entry points (`load_family`, etc.).
  - Re-export selected library helpers (PCA primitives, frequency primitives).
- [ ] Internal modules (`miscope.analysis.analyzers.*`, `miscope.dashboard.*`)
  are not part of the public API. Consumers that want them import explicitly.
- [ ] `from miscope import *` imports only the documented surface.

### Release workflow

- [ ] GitHub Action workflow (`.github/workflows/pypi-publish.yml`) builds
  and publishes to PyPI when a `v*` tag is pushed. Trusted publishing
  (OIDC) preferred over long-lived API tokens.
- [ ] Release dry-run target (Test PyPI) for verifying the package builds
  and installs cleanly before a real release.
- [ ] Wheel build is reproducible — the same tag produces the same artifact
  bytes. No timestamps from the build environment leak in.
- [ ] Tag → release notes pipeline: a `v1.0.0` tag produces a GitHub
  Release with body sourced from `CHANGELOG.md`. Releases for Parquet
  data bundles (REQ_108) use a different tag prefix (`data-*`) to keep
  package releases and data releases separate.

### Artifact schema versioning

- [ ] Schema version mint at v1.0. All artifacts written by the canonical
  analyzer set after the mint date carry `_format='{name}/v1.0'`.
- [ ] `ArtifactLoader` (or `ArtifactSource` + reader) validates `_format`
  on read and emits a clear error if a notebook expects v1.0 but reads v0.x.
- [ ] Versioning is per-artifact (e.g., `parameter_snapshot/v1.0`,
  `freq_group_weight_geometry/v1.0`) so individual schemas can evolve
  independently.
- [ ] Versioning policy documented: format changes require version bump;
  additive field additions are non-breaking and don't require a bump.
- [ ] **Published Parquet schemas (REQ_108) carry a stronger guarantee
  than internal artifact schemas.** Internal `.npz` schema changes are a
  re-run; published Parquet schema changes break every reviewer's notebook
  and every citation. REQ_108 owns the published-schema policy; REQ_103
  ensures the distinction is documented and that the version-bump policy
  for published bundles is more conservative than for internal artifacts.

### Canonical templates

- [ ] `templates/` directory at the repo root (or `examples/`) contains:
  - `notebook_canonical_access.py` — how to load a variant, query a
    frequency set, get a PCA result, render a plot. Heavily commented.
  - `notebook_experimentation.py` — how to extend with a new computation
    while still using stored artifacts as inputs. Shows "this is the
    boundary between using and inventing."
  - `dashboard_page_template.py` — formalize the per-page pattern for new
    dashboard pages (`_VIEW_LIST`, layout, callbacks).
- [ ] Templates are linted, type-checked, and tested as part of CI.
- [ ] Templates are the first link in the published README — they are the
  reading order for a new researcher.

### Repo presentation (first-impression surface)

- [ ] **Root README rewritten** to frame the repo for first-time visitors
  arriving from PyPI, fieldnotes, or a paper. Opening paragraph names what
  MIScope is (a dynamics analysis platform), where the installable package
  lives (`src/miscope/`), and what the surrounding directories are (the
  research workbench that produced it). Transparent, not apologetic — the
  research workbench is part of the project's value, but a visitor needs
  to know what they're looking at.
- [ ] **Research workbench files moved under `research/`.** `notebooks/`,
  `notes/`, exploratory `requirements/active/` drafts, and `articles/`
  consolidate into a single clearly-marked directory. A visitor sees the
  separation between platform (`src/miscope/`) and workbench (`research/`)
  immediately. Implementation of this restructure is handled directly,
  outside the formal REQ workflow — it is a tidying pass, not a design
  decision.
- [ ] **Root-level directory layout post-restructure** documented in the
  README so the structure is legible:
  ```
  src/miscope/        # the installable package
  dashboard/          # local interactive surface (optional install)
  fieldnotes/         # publication site (Astro → GitHub Pages)
  research/           # research workbench: notebooks, notes, drafts
  requirements/       # active design work (in-flight)
  results/            # gitignored — regenerable internal artifacts
  tests/              # test suite
  ```
- [ ] **PROJECT.md and CLAUDE.md remain at root** as the architectural and
  collaboration framing documents. Not moved.

### Public-facing surfaces (the three doors)

The three surfaces a reviewer encounters and what each is for:

| Surface | Audience | Purpose |
|---|---|---|
| PyPI page | "Can I install this?" | Polished package description, install command, link to fieldnotes Platform docs |
| Fieldnotes site (GitHub Pages) | "What did this find?" + "How do I use it?" | Research articles + Platform docs section |
| GitHub repo | "Can I read the code?" | Source, issue tracker, citation; framed by the root README |

- [ ] **Fieldnotes grows a "Platform" section** in the Astro navigation,
  alongside the existing research articles. Houses:
  - Install guide (`pip install miscope`, extras, supported Python versions)
  - API reference (generated from `miscope.core` and library docstrings,
    or hand-written for v1.0 if generation tooling is deferred)
  - Quickstart example: load a variant, query a result, render a figure
  - Pointer to the Discoverability Registry (REQ_107) as the
    "what's available" answer
  - Pointer to article-specific Parquet bundles (REQ_108) for reviewers
    who want to verify findings
- [ ] **One Astro site, two audiences.** The Platform section is the
  polished public face for the package; the existing research posts remain
  the public face for the findings. Single deployment, single domain,
  single CORS context — the same site that hosts a fieldnotes article also
  hosts the Platform docs and the article's published Parquet bundle (with
  Releases as a fallback host for larger bundles, per REQ_108).
- [ ] **PyPI long description** points at the fieldnotes Platform section
  as the canonical entry point for new users. PyPI is the install vector;
  fieldnotes is the front door.

### Documentation

- [ ] Top-level README documents:
  - Installation: `pip install miscope`, optional extras.
  - Quickstart against a published Parquet bundle (REQ_108) — the leanest
    "verify a finding" path that doesn't require local pipeline runs.
  - Pointer to templates as the canonical access patterns.
  - Pointer to PROJECT.md for architectural overview.
  - Pointer to fieldnotes Platform section for full docs.
- [ ] `miscope.core` module-level docstrings document each enum / type for
  generated API docs (or for the hand-written reference in fieldnotes if
  generation tooling is deferred to a follow-up).

### Validation

- [ ] Fresh `pip install miscope` in a clean venv → import miscope → load
  a published Parquet bundle (REQ_108) → query via DuckDB → render a
  figure end-to-end. Smoke test before each release. (HF-based smoke test
  added later if/when REQ_100 lands.)
- [ ] Templates run end-to-end against a published Parquet bundle.
- [ ] Repo root presents cleanly to a first-time visitor: README opens with
  the framing paragraph, no broken or stale top-level files, `research/`
  separation visible.
- [ ] Fieldnotes site builds and deploys with the new Platform section
  alongside existing research posts. Navigation makes the two audiences
  legible; CORS works for DuckDB-WASM queries against same-origin Parquet.
- [ ] **REQ_106 acceptance criteria pass on grep tests.** Layering principles are enforced before publication so external researchers see a self-consistent architecture.
- [ ] **REQ_107 registry is browsable from a notebook.** `miscope.registry.search("...")` returns reasonable results for top-level concepts (frequency, PCA, geometry, neuron). The "first 5 minutes" researcher experience works.
- [ ] **REQ_108 publication bundle workflow is operational.** At least one published Parquet bundle exists, attached to a `data-*` GitHub Release, queryable via DuckDB-WASM from a fieldnotes article. Without this, the v1.0 publication story is incomplete.

---

## Constraints

**Must:**
- Public API is stable post-1.0. Renames or removals require a major bump.
- Templates use only public API. If a template needs an internal helper,
  promote the helper to public API or refactor the template.
- The repo root presents the platform first, the workbench second. A
  visitor with no prior context should be able to tell what MIScope is
  within 30 seconds of landing.
- Single Astro site for fieldnotes — Platform docs and research posts share
  the deployment. Splitting them into separate sites is out of scope.

**Must avoid:**
- Importing `dash` or other dashboard dependencies from `miscope.core` or
  `miscope.analysis.library`. Researchers running notebooks should not need
  a Dash install. Move to optional extras.
- Bundling artifact data in the PyPI package. Published data lives in
  GitHub Releases (REQ_108).
- Breaking schema changes after v1.0 mint without a clear migration story.
  Published Parquet schemas are stricter than internal `.npz` schemas
  (REQ_108 owns the policy).
- Splitting the repo into multiple repos at v1.0. Multi-repo overhead is
  not justified by current demand; deferred per the discussion thread that
  produced this REQ. Triggering conditions for a future split are
  documented in the Architecture Notes.

**Flexible:**
- Whether `templates/` lives at repo root or under `examples/`. Either is fine.
- Choice of API docs tool (Sphinx vs MkDocs vs hand-written MDX in
  fieldnotes). Default for v1.0: hand-written MDX in the fieldnotes
  Platform section. Richer generated docs in a follow-up.
- Versioning scheme beyond v1.0 (semver-strict vs calver). Default: semver.
- Whether the `research/` directory restructure ships before or alongside
  the v1.0 mint. Either order works; the restructure is independent
  tidying, not a v1.0 blocker.

---

## Architecture Notes

**Public API minimalism.** The smaller the public surface, the more
freedom we have to refactor internals without breaking researchers' code.
Start narrow; grow on demand.

**Templates as guard rails.** The user emphasized that scattered access
paths to artifact data have been a recurring problem. The templates are
the durable answer: "use this pattern; deviate only when you mean to."
The "when to experiment with new computations" guidance in
`notebook_experimentation.py` is the explicit permission slip that keeps
researchers from inventing new paths to existing data.

**Schema versioning per-artifact, not global.** A change to one analyzer's
output shouldn't invalidate every other artifact. Per-analyzer versioning
keeps the blast radius small.

**First impression as a real cost for an independent researcher.** The user
named this directly during the discussion that produced this REQ:
> "As an independent researcher, I don't have a lot of social capital to
> lean on or burn. The work has to be the loudest message."

Without existing audience, the repo, the PyPI page, and the fieldnotes
site each have to carry their own credibility load. A research workbench
spilling into the repo root undermines the work it produced. The
restructure (`research/` directory + framed README + fieldnotes Platform
section) is the cheapest move that protects the impression without paying
multi-repo overhead.

**Why monorepo at v1.0, not split.** The split decision was discussed at
length. Neither PyPI publication nor Parquet publication actually forces
multi-repo: PyPI cares about package identity, not repo layout (Flask,
FastAPI, Pydantic have all shipped from monorepos); Parquet bloat has
GitHub Releases as a clean escape hatch (REQ_108). The Manim / Manim
Community Edition split was raised as a precedent, but it happened *after*
a community formed organically — engineering it before demand exists pays
ongoing drift cost without the offsetting benefit. The split decision is
deferred, not rejected; the triggering conditions are: a real community
contributing, a co-maintainer needing scoped commit access, licensing
divergence, or backwards-compat constraints meaningfully slowing iteration.

**Three doors, one site.** PyPI = install vector. Fieldnotes = front door
for both research findings and Platform docs. Repo = source of truth and
citation surface. Each surface has a single primary purpose; surfaces link
to each other but don't duplicate.

---

## Notes

- The HF artifact repo is **deferred**. Original framing assumed HF as the
  primary publication channel; the discussion thread that produced this
  REQ reframed publication around derived Parquet views (REQ_108) on
  GitHub Pages + GitHub Releases. HF is a complementary surface for raw
  artifacts if/when reviewers ask for them. REQ_100 (HFHubSource) is
  similarly deferred.
- This REQ is the milestone gate for v1.0. The other consolidation REQs
  fold into it implicitly: nothing publishes until they're complete.
- The `_format` placeholder key is already written by the discovery
  storage scaffolding; mint = flip from `'{name}/discovery'` to `'{name}/v1.0'`.
- The `research/` directory restructure is tidying, not a design decision —
  the user has stated they will handle this directly. Listed as a CoS
  item here so the validation step ("repo presents cleanly") has something
  concrete to check.
- The fieldnotes Platform section is new construction in the existing
  Astro site. It does not change the deployment topology (still GitHub
  Pages) or the writing workflow (still in this Claude Code context). It
  adds navigation and content, not infrastructure.
