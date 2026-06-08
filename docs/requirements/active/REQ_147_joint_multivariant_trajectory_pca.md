# REQ_147: Joint / Multi-Variant Trajectory PCA

**Status:** Draft (queued, 2026-06-07) — carved out of REQ_099 as a design requirement;
not yet in active execution.
**Priority:** Medium
**Branch:** TBD
**Dependencies:** REQ_099 (carve-out origin), REQ_110 (`miscope.query.open(family)`
cross-variant surface — the enabler), the `parameter_trajectory` / `pca_results` /
`global_centroid_pca` warehouse tables, REQ_107 (registry).
**Attribution:** Engineering Claude (under user direction, 2026-06-07)

---

## Problem Statement

`parameter_trajectory`'s PCA views compute their projections **inline in the renderer**
(an REQ_099 offender), and they do so **per variant** — each variant gets its own PCA
basis. The user wants the opposite of a cleanup here: a *new instrument*. Motivated by a
paper using **joint trajectory PCA**, the goal is to plot **multiple variants on a single
PCA plot** — projecting several variants' parameter-space trajectories into one **shared
basis** so their paths are directly comparable (do they share a route through parameter
space? where do they diverge?).

This is now feasible because REQ_110's `miscope.query.open(family)` gives a cross-variant
surface the old per-variant loader never had. So the migration off the renderer and the
new multi-variant capability are the same piece of work — do them together with
considered design rather than mechanically.

## Design direction (to refine at activation)

- The PCA fit over a **stacked set of variants' trajectories** (shared basis) is a joint
  computation — analyzer/derived territory, not renderer. Decide its home: a derived
  table / library function over the cross-variant `parameter_trajectory` (or weight-space)
  data, parameterized by the variant set.
- The basis-choice question is the real design content: whose trajectory defines the
  basis (one anchor variant? the joint stack? a reference run?), and how variants are
  aligned (sign-gauge / rotation ambiguity across variants — cf. the SVD sign-gauge work
  in REQ_133's notes).
- The renderer becomes plot-only: it receives projected coordinates per variant + render
  params and lays them out (one trace per variant in the shared basis).

## Conditions of Satisfaction (draft)

- [ ] A joint trajectory PCA over a chosen set of variants exists as a conformed,
  registry-declared computation reached through `miscope.query` — not computed in the
  renderer.
- [ ] A view plots N variants in the shared basis on one figure, with the basis-choice
  and alignment decisions explicit and auditable.
- [ ] `parameter_trajectory`'s PCA renderers are plot-only (no inline SVD/PCA).
- [ ] Single-variant behavior is preserved (N=1 reproduces today's per-variant plot
  within tolerance) so the existing view doesn't regress.

## Constraints

- **Universal-instrument invariant:** the multi-variant PCA is an instrument applicable
  to any family's variants; family context is a parameter.
- **Storage-encapsulation invariant:** cross-variant reads via `miscope.query.open(family)`;
  no path literals.
- **Validate against the pinned baselines** for the single-variant regression; the
  multi-variant capability is new (no prior output to match).

## Notes

- Carved out of REQ_099 on 2026-06-07 (user direction): the PCA work is a new instrument
  deserving considered design, not a mechanical compute-subtraction. REQ_099 keeps the
  unrelated `parameter_trajectory` **proximity** offender (a simple derived table).
- Relationship to REQ_146: both are "composite/coordination" instruments, but REQ_146
  conforms *streams within one variant* into aligned panels, while REQ_147 aligns
  *multiple variants* into one basis — different axes, separate requirements.
