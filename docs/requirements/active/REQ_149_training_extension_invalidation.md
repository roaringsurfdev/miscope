# REQ_149: Training-Extension Invalidation & Checkpoint Provenance

**Status:** Draft (stub — for discussion)
**Priority:** Medium-High — blocks trustworthy incremental refresh; surfaced a live over-invalidation during REQ_137-adjacent work.
**Branch:** authored directly on `develop` (per user direction); implementation branch TBD.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Extending training on `p109/s485/ds598` from 25K to 30K epochs triggered
re-analysis of **all 302 checkpoints**, not just the ~50 newly added ones. The
0–25K checkpoints are functionally unchanged, yet every per-epoch artifact was
marked stale and recomputed.

This is the first training extension since the REQ_145 signature-based
invalidation logic landed, and it exposed a cluster of latent issues that are
better decided together than patched piecemeal. The point of this REQ is to
capture them and force the design discussion, not to prescribe a fix.

## What was observed

- The extension re-trained **from scratch (0→30K)**, rewriting every checkpoint
  file (all 302 mtimes fall in one ~6-minute window, monotonic by epoch).
- The byte sizes are identical (`906633`) and, by seed/hyperparameter
  determinism, the 0–25K *weights* are almost certainly bit-identical to the
  prior run. Only the *files* (and their mtimes) are new.
- Per-epoch artifact signatures flipped for all epochs → planner marked all
  stale → full re-analysis.

## Contributing causes (each its own decision)

**1. Two checkpoint-writing paths with opposite overwrite semantics.**
`scripts/fill_checkpoints.py` is the designed incremental path: it *resumes* from
an existing checkpoint, *skips existing checkpoints*, and its docstring directs
"re-run the analysis pipeline with `force=False` — only the new epochs will be
processed." The **main training path has no such guard**: it re-trains from
epoch 0 and overwrites identical snapshots. "Extend training" used the main
path, so the no-overwrite rule that *was* believed to exist (and does, in
`fill_checkpoints`) did not apply.
*Open question:* should "extend/resume" be a first-class, non-destructive
training mode — and should from-scratch training **refuse to overwrite** an
existing variant's snapshots without an explicit flag? Should `fill_checkpoints`
become the only sanctioned extension entry point?

**2. The checkpoint fingerprint keys on mtime, not content.**
`Variant.checkpoint_fingerprint` returns `f"{epoch}:{st.st_size}:{st.st_mtime_ns}"`
(`families/variant.py`). mtime is a proxy for "changed" that breaks under
rewrite, copy, `git checkout`, rsync, and filesystem migration. Even the *safe*
`fill_checkpoints` path is one stray filesystem operation away from spuriously
invalidating an entire variant's artifacts.
*Open question:* content-address the checkpoint fingerprint
(`epoch:size:hash(bytes)`), with an mtime-keyed hash cache (git-index strategy:
trust the cached hash while mtime is unchanged; re-hash only when mtime moves,
and treat a matching hash as still-fresh). ~sub-second for 302×~900KB at plan
time.

**3. Checkpoint hashing vs. the deferred artifact-hashing decision.**
REQ_145 deliberately made signatures *input-derived* and **did not** hash
artifact *outputs* (signature.py "fork c", deferred). Checkpoints are *inputs*
(DAG leaves), so content-addressing them is **orthogonal** to that decision —
arguably the right asymmetry: leaves are content-addressed, derived nodes are
input-signature-addressed. This is the "awkward mixed file-validation strategy"
to accept explicitly, with a clear rationale, rather than avoid.

## Relationship to other requirements

- **REQ_145** owns the signature mechanism; this is a refinement at the
  fingerprint *leaf*, on the seam REQ_145 established. The planner/signature
  design itself is sound (per-epoch, input-derived, forward-propagating) — the
  fragility is solely in what the leaf fingerprint hashes.
- **REQ_137** (full variant refresh): per the user, its CoS is met **except the
  "re-train with dense checkpointing" clause** — existing variants were
  re-analyzed on their *existing* checkpoint schedules, but dense checkpointing
  has not been applied corpus-wide. Dense checkpointing is exactly the
  extend/fill workflow this REQ must make non-destructive **before** REQ_137 fans
  it out — otherwise the densification pass would re-invalidate every variant's
  artifacts (the very thing observed here, at corpus scale).

## Conditions of Satisfaction (to flesh out at scoping — discussion needed)

- [ ] Decide and implement the canonical non-destructive "extend training" path
      (resume + append; existing snapshots never silently overwritten).
- [ ] Establish a single "training never overwrites existing snapshots" guard
      (one enforcement point, not per-script convention).
- [ ] Decide whether to content-address the checkpoint fingerprint; if yes,
      define the hash, the mtime-keyed cache, and the checkpoint-hash /
      artifact-signature validation boundary.
- [ ] Re-extending a test variant processes **only** new epochs; pre-existing
      artifacts stay fresh (acceptance against a re-run of the p109 scenario).

## Notes / open questions

- **Determinism caveat** (shared with REQ_137 and the `fill_checkpoints` Adam-
  momentum note): a resumed run may not be bit-identical to a from-scratch run in
  the resume-settling window. A content hash would treat those bits as *changed*.
  This is the argument that **resume-without-overwrite** (don't touch existing
  files at all) is more fundamental than relying on reproducibility — content
  hashing is the general-purpose robustness layer underneath it, not a
  substitute.
- Confirmed during triage: no no-overwrite guard exists in the main training
  path; only `fill_checkpoints.py` enforces it.

## Out of scope

- REQ_145 "fork c" output-hashing for *artifacts* — a separate, still-deferred
  decision.
