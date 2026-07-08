# REQ_151: Searchable attention-entropy field at the decision query

**Status:** Draft — not started.
**Priority:** Medium — codifies a load-bearing research signal into a searchable analyzer field; no consumer is blocked, but it unlocks candidate-variant screening.
**Branch:** TBD (`feature/REQ_151_attention_query_entropy`).
**Attribution:** Engineering Claude (under user direction).

---

## Problem Statement

A reproducible cross-variant signal surfaced in `apps/research/notebooks/attention_head_pairing.ipynb`
and `flat_variant_entropy.ipynb`: the **Shannon entropy of attention at the decision query** (the `=`
position for modular addition), per head, tracks a structural motif. Across independently-trained
models the four heads split **2+2** — two distinct-frequency ("solo") heads carry a co-moving entropy
*excursion* (a rise toward uniform) during the late instability, while the doubled-frequency pair
stays sharp. The strength/coherence of that excursion is a **continuum** that distinguishes "escape
regime" (loss-spike) variants from "flat" ones, with no hard boundary.

Today that signal is only computable **ad hoc in notebooks**. The existing `attention_patterns`
analyzer captures the raw patterns but declares them as a single **tensor** field
(`patterns`, inner axes `(n_heads, query, key, a, b)`) — there is **no columnar, per-epoch entropy
field**, so the registry has nothing to search and the warehouse has no scalar to filter on. We
cannot answer "which variants show a coherent solo-head entropy excursion?" without re-deriving it by
hand each time — exactly the re-derivation the discoverability mandate (REQ_107) exists to prevent.

This requirement codifies the entropy as a first-class, registry-searchable field, and a derived
per-variant signature that makes candidate screening a single query.

## Conditions of Satisfaction

- [ ] **Columnar `query_entropy` field on `attention_patterns`** (extend the existing analyzer — it
  already runs the forward and holds the patterns; do **not** add a redundant analyzer or a second
  forward pass). Grid-mean Shannon entropy (nats) of the attention distribution over key positions,
  per `(head, query position)`. Declared on `AnalyzerSpec.outputs` (REQ_107) as `kind=columnar`,
  `dtype=float64`, `coords=(variant, epoch, head, query_pos)`. Emitting **all** query positions (not
  just the decision one) keeps the analyzer family-agnostic and is cheap (n_ctx is small); consumers
  slice the decision query.
- [ ] **Query-position coordinate.** Add a `QUERY_POS` (or equivalent) coordinate to the `Coord`
  vocabulary (`analysis/output_schema.py`) if not already present — mirroring how REQ_136 added
  `HEAD`. Confirm head/position counts come from the model config, not a hardcode.
- [ ] **Derived per-variant excursion signature** (REQ_141-style derived table, joining the entropy
  field with `weight_basis_projection`'s per-head dominant QK frequency): per-head post-grok
  excursion magnitude at the decision query, solo-vs-doubled head labels (from the doubled-frequency
  pair), and the within-solo-pair entropy-trajectory correlation. This is the searchable payoff —
  "filter variants whose solo heads show a coherent excursion above threshold." The **decision-query
  position is supplied as family context**, not hardcoded to index 2.
- [ ] **Parity check.** The analyzer field + derived signature reproduce the notebook numbers
  value-identically on p109/s485/ds598, p101/s999/ds999, p107/s999/ds42 (excursion magnitude,
  within-pair correlation) — the notebooks are the regression oracle.

## Constraints

- **Universal instrument (invariant 1).** "Attention entropy at a query position, per head" is a
  universal lens — it belongs to no family. The analyzer must not name `=` or hardcode query index 2.
- **Family as context provider (invariant 2).** *Which* query position is the "decision" query is
  family context, consumed only by the derived signature (and any view), never baked into the
  universal analyzer. The probe/grid is already family-supplied.
- **Storage internal to the API (invariant 3).** Emit through the declared schema; the warehouse
  consumes the declaration unchanged (REQ_136 pattern). No path literals; reach data via accessors.
- **Atomicity.** The analyzer emits the entropy *primitive* only. The frequency join, solo/doubled
  labeling, and excursion/correlation scalars live in the **derived layer** — do not pull the
  `weight_basis_projection` dependency into `attention_patterns`.

## Notes

- Grounding: committed notebooks `attention_head_pairing.ipynb` (event variants, 6 sections) and
  `flat_variant_entropy.ipynb` (flat-variant continuum), commit `611d285` on `develop`. Memory:
  `finding-attention-head-frequency-pairing`.
- The excursion is a **rise** toward the uniform line (ln n_ctx ≈ ln 3 for modadd) — a
  *de-sharpening*, the opposite of Zhai/σReparam entropy *collapse*. Worth a one-line note wherever
  the field is documented so it is not misread as a collapse diagnostic.
- `attention_patterns` is currently single-block (`blocks.0.attn.hook_pattern`). Multi-layer / `site`
  generality is out of scope here; if added later, fold `site` into the coords as REQ_136 did.
- The float64 forward seam (develop `ce70048`) is a **no-op** for this field; the analyzer uses the
  standard (float32) forward like every other analyzer.
- Adjacent open question (separate, cheaper screen — not this REQ): does rolling `PR_3(Attn)` of
  class centroids rise with the excursion? If so it is a coarse pre-screen that catches the loud
  (escape-regime) cases but likely misses the faint flat-variant drifts. Worth an overlay test before
  deciding whether PR_3 belongs in the searchable signature.
