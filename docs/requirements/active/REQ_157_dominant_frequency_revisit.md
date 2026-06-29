# REQ_157: Revisit dominant-frequency data (where does "the head's frequency" live?)

**Status:** Drafted (problem-first; surfaced by REQ_155).
**Priority:** Medium — blocks the copy/transform-vs-frequency orthogonality view
(REQ_155 OPEN item) and recurs wherever a per-head frequency is wanted.
**Attribution:** Surfaced by Engineering Claude during REQ_155; user-flagged as
queue-worthy ("the dominant frequency data needs a revisit").
**Related:** REQ_155 (deferred dominant-freq cross-reference), REQ_130 (fourier
quality repoint), the neuron→freq membership-rule and first-mover off-by-one
findings.

---

## Problem Statement

REQ_155 wanted to annotate the per-head copy/transform ranking bar with **each
head's dominant frequency** (the 2+2 pairing), to show that the copy↔transform axis
is orthogonal to the frequency axis. That annotation was left **OPEN** because the
data didn't fit cleanly:

- `weight_basis_projection.dominant_frequency` is keyed by `(variant, epoch, site,
  head, row_id)` — but `site` here is a **Fourier projection site** (resid_pre,
  attn_out, mlp_out, resid_post), **not** a circuit composition site (full_ov,
  full_qk, …). There is no clean, non-arbitrary mapping from "an attention head's
  frequency" to one of those Fourier sites.
- So "the head's dominant frequency" — a concept the findings (attention-head
  frequency pairing, 2+2 split) treat as well-defined — currently has **no
  first-class home** keyed by `(head)` in a way a circuit consumer can join to.

The question this requirement answers: **what is the canonical per-head frequency,
and where should it live**, such that a consumer (the ranking bar, a cross-variant
panel, a claim) can join it to a head without re-deriving or guessing a site?

## Conditions of Satisfaction

- [ ] A clear answer to "the head's dominant frequency" — a defined operand/site
  whose per-head Fourier dominant frequency *is* the quantity the 2+2 findings mean
  (candidates: the head's QK circuit, its OV circuit, or its attention-pattern
  spectrum). Pick one, justify it against the findings, record the decision.
- [ ] That per-head frequency is reachable through the API keyed by `(variant,
  epoch, head)` — either an existing field repointed, or a thin new one — without a
  consumer choosing a Fourier site by hand.
- [ ] The REQ_155 ranking bar can annotate/colour heads by dominant frequency from
  that source (closes the REQ_155 OPEN item).
- [ ] Sanity-check against the documented frequency-indexing gotchas (row index `k`
  = freq `k+1`; first-mover +1) so the reported frequency is the corrected one.

## Constraints

- **Universal instrument (invariant 1).** The Fourier transform is universal; this
  is about *which operand* to point it at and *how to key the output*, not a new
  bespoke analyzer. Reuse `weight_basis_projection` / the universal Fourier
  primitive over the chosen site.
- **Read through the API (invariant 3).** Whatever the home, consumers join via
  `miscope.query` / a keyed field — no per-consumer site guessing.

## Notes

- This is upstream of, not blocking, the rest of REQ_155 (which shipped without the
  annotation). It bumps in priority because the same per-head frequency is wanted by
  more than one surface.
- Likely interacts with the circuit_spectra sites (REQ_152/154): if the answer is
  "the head's QK circuit dominant frequency," that operand already exists as a
  circuit site — the Fourier instrument may just need repointing onto it, which
  would also brush the site-addition / refresh-signature gap (a forced rebuild,
  per the REQ_145 signature finding).
