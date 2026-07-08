# REQ_076: Peer Variant Comparison Page

**Status:** Active
**Branch:** TBD
**Attribution:** Drafted by Engineering Claude

---

## Problem

Cross-variant observations currently live only in notebooks. The weight trajectory divergence notebook (REQ_072) surfaced something important: the normalized divergence and per-matrix breakdown plots, read alongside loss curves, contain information about *when and how* variants fork — information that is hard to extract from single-variant views. The epoch where weight-space paths diverge, the matrix that leads the fork, and where loss curves depart from each other are all visible together, but only with manual notebook runs against a specific group.

The scientific workflow this blocks: choose a variant, immediately see how its peers (same architecture, same prime, different data seed or model seed) behaved across the same training run. Without this, cross-variant comparison requires context-switching out of the dashboard and running notebook cells manually for any pair of interest.

---

## Conditions of Satisfaction

1. **Peer discovery** — Given a selected anchor variant, the page automatically determines two peer sets:
   - *Data-seed peers*: same prime + model seed, all available data seeds
   - *Model-seed peers*: same prime + data seed, all available model seeds

   A toggle on the page context-nav switches between the two axes. If only one axis has more than one variant, that axis is selected by default.

2. **Loss curve overlay** — A single plot showing train and test loss for each peer variant, overlaid on shared axes. Color = peer identity (using existing COLORS palette where applicable). Train = dashed, test = solid. Epoch range matches the shared checkpoint range across peers.

3. **Normalized weight divergence trajectory** — For each non-anchor peer, the normalized L2 weight divergence (divergence / reference norm, as a percentage) against the anchor, plotted as a trajectory over epochs. One line per peer. Computed from existing `parameter_snapshot` artifacts.

4. **Per-matrix divergence breakdown** — The normalized weight divergence decomposed by weight matrix (W_E, W_pos, W_Q, W_K, W_V, W_O, W_in, W_out, W_U), rendered as a 3×3 subplot grid. One line per peer per matrix.

5. **Epoch cursor** — A shared vertical cursor line across all plots. Dragging or clicking the cursor updates its position across all plots simultaneously, enabling visual alignment of "what was the per-matrix divergence at the epoch where loss curves first diverged?"

6. **Load-on-demand with progress** — Weight divergence computation (parameter_snapshot loading) is triggered explicitly (e.g., a "Load" button or on page activation), not on every variant selector change. A progress indicator is shown while loading. Loss curves render immediately from metadata without waiting for artifact loading.

7. Page is accessible from the dashboard nav and follows the existing page structure (`create_page_nav()`, `create_page_layout()`, `register_page_callbacks()`).

---

## Constraints

- **Weight divergence requires all peers to share model architecture.** The page should validate this silently and exclude incompatible peers.
- **Anchor = the variant currently selected in the variant-selector-store.** The page does not introduce a second variant picker. The anchor is always the globally selected variant.
- **Loss curves render from `variant.train_losses` / `variant.test_losses`** (metadata, no artifact loading). They must be visible immediately.
- **Weight divergence is computed in-page, not pre-stored as an artifact.** This is a derived quantity that depends on which pair is being compared. It does not belong in the analyzer pipeline.
- **No new analyzer or artifact type.** The page consumes existing `parameter_snapshot` artifacts and existing loss metadata only.
- **Epoch cursor is a UI affordance, not a global epoch selector.** It controls a local cursor overlay across the page's plots only; it does not update the global epoch slider.

---

## Notes

**Why this is likely a quick win:** Loss curves are metadata (no I/O cost). Weight divergence computation is a straightforward L2 over existing artifacts — the notebook already does this in <50 lines. The peer discovery query is a simple family filter. The main new platform piece is the page shell and the shared epoch cursor. No new analyzers, no new artifacts, no new view catalog entries required.

**Performance:** Each variant has ~94 parameter_snapshot epochs by default (more with extended training or dense-checkpoint schedules). With up to 3 peers, that is ~280 artifact loads per comparison — meaningful I/O that justifies the load-on-demand approach in CoS 6. If this proves too slow for interactive use, caching the computed divergence as a lightweight `.npz` alongside the variant's artifacts is the escape hatch. Do not design the cache in upfront; add it only if the load time is actually a problem in practice.

**Relationship to REQ_072/073:** REQ_072 defines the divergence views as part of the view catalog pipeline. This requirement is a dashboard *surface* for the same ideas, not a reimplementation. If/when REQ_072 is implemented, the weight divergence computation from this page could be factored into shared helpers. Until then, they can coexist independently.

**Epoch cursor vs. click-to-navigate:** This page's epoch cursor is a local analytical affordance (visual alignment across synchronized plots). It is distinct from the click-to-navigate pattern elsewhere in the dashboard that updates the global epoch slider. This distinction should be preserved to avoid confusion.

**Future extensibility:** Once the peer-set derivation and page shell exist, adding more views (gradient energy profiles from REQ_072, DMD mode spectra from REQ_073) is an incremental addition to an existing page rather than a new requirement. The peer axis toggle is the reusable structural piece.
