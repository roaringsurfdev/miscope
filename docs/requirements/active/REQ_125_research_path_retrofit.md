# REQ_125: Research Notebook & Sketch Path/Storage Retrofit

**Status:** Drafted (stub — Problem Statement only; CoS to be filled in after REQ_123 lands and the API-side surface is settled).
**Priority:** Medium — pure cleanup, but high-leverage: existing notebooks act as exemplars that future research code copies from. Every day they keep the old shape, new code re-introduces the same leakage.
**Branch:** TBD
**Dependencies:** REQ_123 (Unified Data Root) lands first — this REQ retrofits research code against whatever API-facing accessors REQ_123 puts in place (e.g. an accessor for `variant_summary.json` / `variant_registry.json` so callers don't compose paths).
**Coordinates with:**
- *REQ_123 — Unified Data Root.* REQ_123 codifies the "storage layout is internal to the API" policy and closes the gap in API/dashboard code. REQ_125 closes the same gap in research code.
- *REQ_124 — Per-App Configuration Files.* Together with REQ_124, REQ_125 ensures the policy is enforced on both the app side and the research side.
**Attribution:** Engineering Claude (drafted as a follow-up captured during REQ_122 review on 2026-05-26).

---

## Problem Statement

Research code under `apps/research/notebooks/*.ipynb` and `apps/research/sketches/*.py` reaches past the MIScope API in a handful of recurring patterns:

- **Direct `ArtifactLoader` instantiation.** `ArtifactLoader` is currently a public class importable from `miscope.analysis`. Notebooks construct it with a path (`ArtifactLoader(str(some_path))`) instead of going through `variant.artifacts.load_epoch(...)`. The Variant API already provides the right accessor; the alternative path exists only because the loader class is public.
- **Path-literal reads.** `Path("results/modulo_addition_1layer/variant_registry.json")` and similar appear in notebooks and sketches, often to grab `variant_summary.json` or `variant_registry.json`. These files have no API accessor today, so the path composition is the only way in.
- **`variant.variant_dir`-based path composition.** Even when starting from a Variant, callers reach into `variant.variant_dir.parent / "something.json"` to read sibling artifacts the API doesn't expose. This is the same leakage with a Variant-shaped fig leaf.

The cost is two-fold: (1) future research notebooks copy from existing ones and propagate the pattern; (2) when REQ_123 changes the on-disk layout, every literal path needs hand-fixing. The fix is to retrofit research code to the API surface REQ_123 puts in place, and to make the API the *only* way in — including by making `ArtifactLoader` non-public.

---

## Notes

- The policy this REQ enforces ("no file paths outside config; storage layout is internal to the API") lands in `PROJECT.md` as part of REQ_123. REQ_125 is the enforcement pass through research code; REQ_124 is the enforcement pass through app code.
- Scope reminder: `apps/research/sketches/*.py` and `apps/research/notebooks/*.ipynb` are the primary surface. Scripts under `scripts/` were already migrated as part of REQ_122 and largely use the API, but should be re-audited here for any remaining path literals.
- Touch-and-go items likely uncovered during the audit (e.g. accessors that don't yet exist on Variant or Family — `variant.summary`, `family.variant_registry`) should be added to the API rather than worked around in research code. Anything that genuinely belongs in research-only scope (one-off plotting, scratch analysis) can stay local but should still use the API for storage access.

---

## Out of Scope

- The on-disk layout migration itself (REQ_123).
- The app-side path/config cleanup (REQ_124).
- Rewriting research analyses; this REQ is mechanical retrofit, not a re-derivation of any finding.
- Building a notebook linter or CI check — useful, but a separate REQ if pursued.
