# REQ_149: Training-Extension Invalidation & Checkpoint Provenance

**Status:** Scoped (2026-06-10) — ready for implementation
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

## Decision (scoped 2026-06-10)

The three forks above are resolved as follows. The deciding observation: the
checkpoint-schedule page's insert/extend path re-trains **from scratch**
(`variant.train(num_epochs=total_epochs, checkpoint_epochs=merged_checkpoint_epochs)`,
`pages/checkpoint_schedule.py`), and "Existing checkpoints are always included" in
the merged schedule — so every existing epoch is re-saved and overwritten today.

**1. No resume-mode.** Building resume-from-checkpoint training logic is explicitly
out of scope. The non-destructive guarantee comes from *not writing* over existing
files, not from resuming.

**2. Skip-existing writes + an `overwrite_all` flag — the load-bearing fix.**
`Variant._save_checkpoint` becomes non-destructive by default: it skips the write
when the checkpoint file already exists, unless an explicit `overwrite=True` is
passed. `Variant.train()` gains `overwrite_all: bool = False`, threaded to
`_save_checkpoint`. The schedule page keeps calling `train()` with the default, so
inserting density / extending becomes a no-op for existing epochs (bytes *and*
mtime preserved) and writes only the genuinely-new epochs — exactly what the
incremental planner needs. A deliberate "redo this variant" is the only caller that
passes `overwrite_all=True`. **Single enforcement point** in `_save_checkpoint`, not
per-script convention. No error-on-existing (which would stall a densification run);
skip is silent, overwrite is opt-in.

*Why skip-existing over content-hashing the fingerprint, for this workflow:* because
the page re-trains from scratch, a content-addressed fingerprint would only keep
0–25K fresh **if** the retrain reproduced bit-identical weights — which the
determinism caveat (below) says it may not. Skip-existing sidesteps the gamble
entirely: the original bytes are never touched, so there is nothing to re-hash and
nothing to invalidate.

**3. Content-addressed fingerprint — deferred to a robustness sub-item.** A
checkpoint-only `epoch:size:blake2b(bytes)` fingerprint with an mtime-keyed hash
cache (git-index strategy) is the right robustness layer for file operations that
bypass `_save_checkpoint` (backup/restore, machine migration, rsync). `git checkout`
is *not* a vector — checkpoints are gitignored. Those remaining vectors are real but
rare, and skip-existing already covers the training workflow, so this is carved out
as a deferred sub-item (REQ_149-A or a follow-up stub), picked up when that risk
becomes concrete. It stays checkpoints-only — REQ_145 "fork c" artifact
output-hashing remains separately deferred.

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

## Conditions of Satisfaction

- [ ] `Variant._save_checkpoint` skips the write when the checkpoint file already
      exists, unless `overwrite=True` is passed — the single non-destructive
      enforcement point.
- [ ] `Variant.train()` exposes `overwrite_all: bool = False`, threaded to
      `_save_checkpoint`. Default (`False`) preserves existing snapshots; `True`
      is the only path that re-writes them.
- [ ] `_save_checkpoint` reports written-vs-skipped so `train()`'s
      `saved_checkpoint_epochs` (and the schedule page's "Checkpoints saved: N")
      reflects epochs actually written, not merely intended.
- [ ] Re-running the p109 25K→30K scenario through the schedule page (default
      `overwrite_all=False`) leaves the 0–25K checkpoint files byte- *and*
      mtime-identical, and a subsequent `pipeline.run(force=False)` plans
      **only** the new epochs — pre-existing per-epoch artifacts stay fresh.
      Acceptance test asserts this directly.

**Deferred (sub-item, not this requirement):** content-addressed checkpoint
fingerprint (`epoch:size:blake2b`, mtime-keyed cache) for fs-operation robustness —
see Decision §3.

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
- **`fill_checkpoints.py` carries a separate, still-unfixed off-by-one** (user-
  flagged 2026-06-10): `FILL_WINDOW_START = RESUME_FROM_EPOCH + 1` then
  `range(FILL_WINDOW_START, …, 100)` shifts the grid off the round hundreds
  (resume @1200 → 1201/1301/1401 instead of 1300/1400). Orthogonal to this REQ —
  the chosen `train()` + skip-existing path takes round-number epochs from the
  schedule builder, not this `range`. fill_checkpoints remains the compute-efficient
  true-resume path but is left with this latent bug; fix is a small separate
  follow-up, not bundled here unless explicitly scoped in.

## Out of scope

- **Resume-from-checkpoint training logic** — the non-destructive guarantee comes
  from skip-existing writes, not from resuming a trajectory. Explicitly not built.
- **Content-addressed checkpoint fingerprint** — deferred robustness sub-item
  (Decision §3); not required to fix the observed over-invalidation or to unblock
  the REQ_137 densification fan-out.
- REQ_145 "fork c" output-hashing for *artifacts* — a separate, still-deferred
  decision.
