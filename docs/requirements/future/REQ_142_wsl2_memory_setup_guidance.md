# REQ_142: WSL2 Memory Setup Guidance for Researchers

**Status:** Future (parking lot — captured 2026-06-06 from the run_with_cache memory investigation).
**Priority:** Low — usability/docs, not a defect. Should land **before public release** (gate with REQ_103).
**Branch:** TBD (`feature/REQ_142_wsl2_setup_guidance`).
**Dependencies:** None code-wise. Related to REQ_103 (PyPI Publication Hardening — the release gate this should ride along with).
**Attribution:** Engineering Claude (under user direction).

---

## Problem Statement

A 2026-06-06 investigation (see
`docs/notes/infra_discovery/memory_investigations.md`) established that the
analysis path holds process RSS flat — there is no memory leak. The "slow
memory climb" observed during full runs is **reclaimable OS page cache**, which
on **WSL2 (default config)** the guest VM holds and is slow to return to the
Windows host. On native Linux/macOS the same cache is plainly reclaimable and
the kernel evicts it under pressure; only WSL2's default behavior makes it *look*
like a runaway.

The risk this addresses is reputational/usability, not correctness: a researcher
running MIScope under WSL2 may watch Windows Task Manager show memory climbing and
not recover, conclude the platform is unstable or "crashes machines," and abandon
it — when the fix is a one-time host configuration the project simply failed to
document. We do not want to ship something that appears to crash other machines
when an obvious fix exists.

This requirement adds the missing guidance to the setup documentation so a new
user on WSL2 (a) understands the climb is reclaimable cache, (b) knows how to
verify it, and (c) has the one-time config to cap and reclaim it.

## Conditions of Satisfaction

- [ ] Setup/README documentation gains a short "Memory on WSL2" subsection that:
  - [ ] States plainly that the analysis path does not leak; the climb is
    reclaimable page cache, and the number to watch is `available` (inside WSL)
    or process RSS — **not** Windows Task Manager "total used."
  - [ ] Gives the one-time `%UserProfile%\.wslconfig` snippet (`memory=` cap,
    `[experimental] autoMemoryReclaim=gradual`, `pageReporting=true`,
    `sparseVhd=true`) followed by `wsl --shutdown`.
  - [ ] Gives the on-demand reclaim check
    (`sync; echo 1 > /proc/sys/vm/drop_caches` from an elevated WSL shell) as the
    way to prove reclaimability.
  - [ ] Notes that native Linux/macOS users need none of this.
- [ ] The guidance links to `docs/notes/infra_discovery/memory_investigations.md`
  for the underlying evidence (or inlines the conclusion if the notes file is
  local-only at release time — see the public/local boundary policy).

## Constraints

**Must:**
- No code change. This is documentation only.
- Framed as environment configuration, not a platform limitation.

**Must avoid:**
- Overstating the issue. The default-WSL2 retention is benign as long as
  `available` stays healthy; the doc should calm, not alarm.
- Prescribing a specific `memory=` value as universal — give a sensible example
  and explain the tradeoff (cap too low starves large runs).

## Notes

- Pairs naturally with REQ_143 (warehouse materializer RSS profile): together they
  give a publication-ready memory story (analysis = flat; cache = reclaimable +
  documented; materializer = characterized).
- The genuine RSS-bound scaling surface is the warehouse materializer, **not** the
  analysis pipeline — keep that distinction in the doc so users profile the right
  thing if they do hit real pressure.
