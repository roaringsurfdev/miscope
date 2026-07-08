# REQ_040: Migrate Training & Analysis Run Management to Dash

**Status:** Draft
**Priority:** High (prerequisite for REQ_039 source layout restructuring)
**Dependencies:** REQ_035 (Dash migration — completed)
**Last Updated:** 2026-02-14

## Problem Statement

The Dash dashboard (`dashboard_v2/`) currently covers only the Analysis visualization experience. The Training tab and Analysis run-trigger UI still live in the Gradio dashboard (`dashboard/app.py`). With REQ_035 complete and Dash proven as the framework, these remaining operational UIs need to migrate before the source layout restructuring (REQ_039) can proceed cleanly.

Additionally, dashboard_v2 has no site-level navigation — it's a single-page Analysis view. As the dashboard grows to encompass training, analysis runs, and eventually reports, it needs a navigation structure that organizes these concerns.

### What's being migrated

**From the Gradio Train tab** (~100 lines):
- Family selection dropdown
- Domain parameter inputs (prime p, seed)
- Training parameter inputs (data seed, training fraction, total epochs, checkpoint epochs CSV)
- Status display + "Start Training" button

**From the Gradio Analysis tab top section** (~80 lines):
- Family + Variant dropdowns (already partially in dashboard_v2 sidebar)
- Refresh + "Run Analysis" buttons
- Analysis status display

### Versioning

`dashboard/version.py` currently tracks a single version (`0.6.0`). With two consumers (Gradio dashboard being retired, Dash dashboard active), version tracking should move to or be accessible from `dashboard_v2/`. Consider whether API and consumer versions should be tracked separately.

## Design

### Site-Level Navigation

Add a top-level navigation element (navbar or sidebar section) that organizes the dashboard into functional areas:

```
Jobs / Runs
  ├── Training
  └── Analysis
Analysis / Reports    (future — scaffolding only for now)
  ├── [Current visualization view]
  └── ...
```

The current visualization view (sidebar + 18 plots) becomes the default landing page under "Analysis / Reports." Training and Analysis Run pages are new additions.

**Navigation component:** A top navbar or collapsible left-nav section above the existing sidebar. The exact component is flexible — what matters is that the user can move between Training, Analysis Runs, and the visualization view without losing context.

### Training Page

Port the Gradio Train tab functionality to a dedicated Dash page/view:

- Family dropdown (populated from FamilyRegistry)
- Domain parameter inputs (adapt to selected family's parameter schema)
- Training parameter inputs (data seed, training fraction, epochs, checkpoint schedule)
- Variant preview (shows the variant name that would be created)
- Status display area
- "Start Training" button

**No new functionality** — this is a straight port. The training pipeline and family registry are untouched.

### Analysis Run Page

Port the Analysis run-trigger section to a dedicated Dash page/view:

- Family + Variant selection (may share components with the visualization sidebar)
- Available analyzers display
- "Run Analysis" button
- Status display area

**No new functionality** — this is a straight port. The analysis pipeline is untouched.

### Version Migration

Move `version.py` from `dashboard/` to `dashboard_v2/` via `git mv` to retain history. Single version for the application — no API/consumer split for now.

## Scope

**This requirement covers:**
1. Site-level navigation structure in dashboard_v2
2. Training page — port of Gradio Train tab
3. Analysis Run page — port of Gradio Analysis run-trigger UI
4. Version tracking migration to dashboard_v2
5. Navigation scaffolding for future "Analysis/Reports" section

**This requirement does not cover:**
- Extended job management (queuing, progress tracking, cancellation) — explicitly deferred
- Analysis Reports UI (future requirement)
- Removal of the Gradio dashboard (separate decision after full parity)
- Changes to the training pipeline, analysis pipeline, or family registry
- Changes to the existing visualization view (sidebar + plots)

## Conditions of Satisfaction

### Navigation
- [ ] Dashboard_v2 has a site-level navigation element
- [ ] User can navigate between Training, Analysis Run, and Visualization views
- [ ] Navigation works reliably between views

### Training Page
- [ ] User can select a family, configure domain and training parameters
- [ ] Variant preview updates as parameters change
- [ ] "Start Training" triggers training and displays status updates
- [ ] Training produces the same results as via the Gradio Train tab

### Analysis Run Page
- [ ] User can select a family and variant
- [ ] "Run Analysis" triggers analysis and displays status updates
- [ ] Analysis produces the same results as via the Gradio Analysis tab

### Versioning
- [ ] Dashboard_v2 displays its version
- [ ] `version.py` moved to `dashboard_v2/` with git history preserved

### General
- [ ] Gradio dashboard remains functional (frozen, not broken)
- [ ] All existing tests pass
- [ ] No changes to training pipeline, analysis pipeline, or renderers

## Constraints

**Must have:**
- Site-level navigation between Training, Analysis Run, and Visualization
- Functional parity with Gradio Train tab and Analysis run-trigger UI
- Version display in dashboard_v2

**Must avoid:**
- Scope creep into job management features (queuing, history, cancellation)
- Modifying training or analysis pipelines
- Breaking the Gradio dashboard
- Over-designing the navigation for features that don't exist yet

**Flexible:**
- Navigation component choice (top navbar vs side nav vs tabs)
- Exact layout of Training and Analysis Run pages
- Whether Training and Analysis Run share UI components (they likely will)
- Visual styling of Training and Analysis Run pages

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-14 | Sequence vs REQ_039? | REQ_040 first | Migrate all UI before restructuring source layout — avoid moving targets |
| 2026-02-14 | Extend to job management? | No — straight port | Job management is a separate concern; get parity first, then enhance |
| 2026-02-14 | Preserve view state across navigation? | Not required | Workflow is batch-oriented (set up runs, come back for results); state preservation adds complexity without clear value |
| 2026-02-14 | Version tracking approach? | Simple `git mv` of version.py | Single application version for now; differentiated versioning deferred |
| 2026-02-14 | Retire Gradio in this REQ? | No | Separate step after parity is confirmed |

## Notes

**Training and Analysis Run UIs will converge.** Both are "configure parameters → trigger a run → show status" workflows. This requirement ports them as-is, but the similarity suggests a shared "run job" component pattern in the future.

**Navigation scaffolding vs implementation.** The "Analysis/Reports" section in the nav is scaffolding — a placeholder that shows where future report functionality will land. It doesn't need to do anything yet beyond perhaps linking to the current visualization view.
