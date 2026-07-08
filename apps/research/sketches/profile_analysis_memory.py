"""Profile the memory-heavy *post-run* stages of an analysis pass.

One-off diagnostic (Issue 2, 2026-06-06). The dashboard Force=True path is:

    pipeline.run()                      # per-epoch + cross-epoch (family.json scoped)
    VariantAnalysisSummary().analyze()  # -> neuron_frequency.load(...)
    build_variant_registry(family)      # -> neuron_frequency.load + query.open

The suspected spike is NOT the per-epoch loop (model rebuilt+freed each epoch) but
``neuron_frequency.load``'s self-heal: on a warehouse cache miss it calls
``variant.warehouse.materialize()``, which stacks *every registered analyzer's*
full trajectory into pandas in one pass (writer.materialize_variant_columnar,
iterating ``reg.index().analyzers`` — the registry, not family.json).

This script measures each post-run stage's RSS against the existing on-disk
artifacts (no model recompute), with a background peak sampler so transient
spikes during pandas concatenation are caught.

Usage:
    python apps/research/sketches/profile_analysis_memory.py --variant p101_seed485_dseed999
"""

from __future__ import annotations

import argparse
import gc
import threading
import time

import psutil

import miscope.analysis.analyzers  # noqa: F401 — triggers @register_analyzer
from miscope import load_family
from miscope.analysis.registry import AnalyzerRegistry

FAMILY = "modulo_addition_1layer"
_PROC = psutil.Process()


def rss_mb() -> float:
    return _PROC.memory_info().rss / 1024 / 1024


class PeakSampler:
    """Background thread tracking peak RSS across a stage."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, rss_mb())
            time.sleep(self.interval)

    def __enter__(self) -> PeakSampler:
        self.peak = rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()


def stage(label: str, fn) -> None:
    """Run fn, reporting RSS before/after and the in-stage peak."""
    gc.collect()
    before = rss_mb()
    with PeakSampler() as sampler:
        result = fn()
    after = rss_mb()
    note = ""
    if isinstance(result, str):
        note = f"  [{result}]"
    print(f"  {label:<34} {before:7.0f} -> {after:7.0f} MB   peak {sampler.peak:7.0f}"
          f"   (Δafter {after - before:+7.0f}){note}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="p101_seed485_dseed999")
    args = parser.parse_args()

    family = load_family(FAMILY)
    variant = next(v for v in family.variants if v.name == args.variant)
    registered = set(AnalyzerRegistry.list_all_names())
    in_family = {s.name for s in AnalyzerRegistry.list_for_family(family)}
    print(f"Variant: {variant.name}")
    print(f"Registered analyzers: {len(registered)} | family.json: {len(in_family)}")
    print(f"Registered but NOT in family.json: {sorted(registered - in_family)}")
    print(f"Baseline RSS: {rss_mb():.0f} MB\n")

    print("== Post-run stages (against existing on-disk artifacts) ==")

    from miscope.analysis import neuron_frequency as nf

    def do_materialize() -> str:
        rep = variant.warehouse.materialize()
        skipped = getattr(rep, "skipped_analyzers", [])
        return f"skipped: {skipped}" if skipped else "ok"

    def do_nf_load() -> str:
        attr = nf.load(variant)
        return f"{attr.dominant_freq.shape}"

    # Stage 1: the self-heal materialize in isolation (the suspected spike).
    stage("warehouse.materialize()", do_materialize)
    # Stage 2: nf.load now that the table exists (the steady-state read cost).
    stage("neuron_frequency.load()", do_nf_load)

    # Stage 3: the full summary build (now a warehouse-table read + roll-up).
    from miscope.analysis.variant_analysis_summary import (
        assemble_variant_registry,
        write_variant_summary,
    )

    stage("write_variant_summary(variant)", lambda: str(write_variant_summary(variant)) and "ok")
    stage("assemble_variant_registry(family)", lambda: str(len(assemble_variant_registry(family))))

    print(f"\n  Final RSS: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
