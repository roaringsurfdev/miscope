# REQ_146: Composite Coordination Views (Conformed Multi-Stream Sources)

**Status:** Draft (queued, 2026-06-07) — carved out of REQ_099 as a design requirement;
not yet in active execution.
**Priority:** Medium
**Branch:** TBD
**Dependencies:** REQ_099 (carve-out origin + the plot-only rule these views must end up
obeying), REQ_110 (warehouse + `miscope.query` surface), REQ_141/144 (derived-table
pattern these new sources follow), REQ_107 (registry — conform new stream tables to the
existing `neuron_frequency_attribution` / `committed_counts` shape, don't re-derive).
**Attribution:** Engineering Claude (under user direction, 2026-06-07)

---

## Problem Statement

Two of the workbench's most information-dense instruments are **composite coordination
views** — single figures that line up several streams so the researcher can see, *in
one glance*, how a lot of moving parts coordinate during training:

- `multi_stream_specialization` (4-panel): frequency specialization accumulating across
  **embedding dimensions, attention heads, MLP neurons,** and **effective
  dimensionality**, on a shared epoch axis with a shared per-frequency color scheme.
- `dimensionality_dynamics` (3-panel timeseries + state-space): PR₃ / f_top3 across
  trajectory, class-centroid, and within-group-weight domains.

Their analytical power comes from the panels being aligned, not from any one panel — so
breaking them into separate plots is explicitly *not* the answer (the user's
requirement). But they are also REQ_099 offenders: they compute their panel quantities
**inline in the renderer**. The reason is structural — *no conformed data source feeds an
at-a-glance multi-stream view today*, so the renderer became the only place that could
assemble the aligned streams. Migrating them is therefore a **data-source design
problem**, not the mechanical subtraction REQ_099 handles, which is why they were carved
out here.

## The design insight (why this is well-scoped, not open-ended)

`multi_stream`'s three specialization panels are the *same semantic shape*: "how much
does stream *S* specialize in frequency *k* at epoch *t*," keyed `(epoch, stream,
frequency)`. The **MLP** stream is already conformed — `committed_counts` over
`neuron_frequency_attribution`. The real work is conforming the **embedding** and
**attention** streams into that same shape:

- embedding: per-`d_model`-dimension Fourier specialization of `W_E` (currently a full
  Fourier projection done in the renderer);
- attention: mean QK Fourier fraction across heads (currently `_compute_attn_aggregate`
  in the renderer, with no warehouse source).

Once those are conformed, the composite view stops owning compute and becomes a
`GROUP BY stream` **join-and-plot**. The universal-instrument payoff lands exactly where
the user wants to reuse the move: **adding a stream becomes adding a conformed table, not
editing a renderer** — which is the test the future DMD composite (deferred) will lean
on. `dimensionality_dynamics` rides along under the same *treatment* but a different
*source shape*: its rolling-window PR₃ / f_top3 is a per-`(epoch, site)` derived column
(rolling over PC projections), not a stream-conforming table.

## Conditions of Satisfaction (draft — refine at activation)

- [ ] A conformed **per-stream specialization** surface keyed `(epoch, stream,
  frequency)` exists on the query surface, with the embedding and attention streams
  joining the already-conformed MLP stream (same shape, registry-declared).
- [ ] The rolling trajectory PR₃ / f_top3 metric (`dimensionality_dynamics`) has a
  warehouse/derived home keyed `(epoch, site)`; the trivial PR₃ formula is a library fn.
- [ ] `multi_stream_specialization` and `dimensionality_dynamics` renderers are
  **plot-only** — no inline Fourier projection, no `np.var` rolling windows, no band
  counting. They take structured, already-conformed data and lay it out.
- [ ] Visual output unchanged within tolerance on the three baselines (the panels still
  line up the way they do today; this is an instrument, lines must not silently move).
- [ ] Adding-a-stream is demonstrably a table addition: a short note shows what a new
  stream (e.g. the eventual DMD one) would need, with no renderer change.

## Constraints

- **Universal-instrument invariant:** the composite view is a layout over conformed
  facts; streams are columns/rows, never view-owned compute. Family context enters as a
  parameter, never as ownership.
- **Storage-encapsulation invariant:** new sources are warehouse/derived tables reached
  through `miscope.query`; definitions in code, locations in config.
- **Must not** break the alignment that is the whole point — no decomposition into
  separate plots.

## Future (out of scope, motivating context)

A **DMD composite** is the deferred third instance of this pattern: a multi-panel view
aligning how different parameter groups and activation sites move, which today live on
separate pages and require flipping back and forth to compare. The user's working
hypothesis — DMD eigenvalue **compression/expansion** events line up with the
**dimensionality** compression/expansion events — is precisely the cross-instrument
coordination such a composite would let you confirm on one timeline. DMD is held out
deliberately (it carries its own complexity) so the conformed-stream pattern is proven
on the two existing views first; the DMD composite should then be a near-mechanical
"add a stream" once REQ_146 lands.

## Notes

- Carved out of REQ_099 on 2026-06-07 (user direction): the composite views are
  load-bearing instruments whose migration is design, not cleanup. REQ_099 keeps the
  mechanical offenders; this REQ owns the data-source design.
- The new tables here are born into the post-REQ_099 non-pinned variant refresh, so they
  populate across the corpus on first materialize (no separate backfill).
