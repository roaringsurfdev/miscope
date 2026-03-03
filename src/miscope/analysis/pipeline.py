"""Analysis pipeline orchestrating analysis across checkpoints."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm.auto as tqdm

from miscope.analysis.protocols import (
    AnalysisRunConfig,
    Analyzer,
    CrossEpochAnalyzer,
    SecondaryAnalyzer,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miscope.families import Variant


class AnalysisPipeline:
    """Orchestrates analysis across checkpoints.

    Enforces the scientific invariant: same Variant + same Probe across
    all analyzed checkpoints. The only variable is the checkpoint (training moment).

    Artifacts are stored as one file per (analyzer, epoch):
        artifacts/{analyzer_name}/epoch_{NNNNN}.npz

    This mirrors the checkpoint storage pattern and enables:
    - Constant memory usage (no in-memory buffer)
    - Incremental computation (resume by checking file existence)
    - Parallel computation (independent files per epoch)
    - On-demand loading (load one epoch at a time for visualization)

    Analyzers may optionally produce summary statistics (REQ_022) — small
    per-epoch values accumulated in memory and saved as a single file:
        artifacts/{analyzer_name}/summary.npz

    Cross-epoch analyzers (REQ_038) run after per-epoch analysis completes,
    consuming per-epoch artifacts to produce cross-epoch results:
        artifacts/{analyzer_name}/cross_epoch.npz
    """

    def __init__(
        self,
        variant: Variant,
        config: AnalysisRunConfig | None = None,
    ):
        """Initialize the analysis pipeline.

        Args:
            variant: The Variant to analyze
            config: Analysis configuration. If None, uses defaults (all analyzers,
                    all checkpoints).
        """
        self.variant = variant
        self.config = config or AnalysisRunConfig()
        self.artifacts_dir = str(variant.artifacts_dir)

        self._analyzers: list[Analyzer] = []
        self._secondary_analyzers: list[SecondaryAnalyzer] = []
        self._cross_epoch_analyzers: list[CrossEpochAnalyzer] = []
        self._manifest: dict[str, Any] = {}

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._manifest = self._load_manifest()

    def register(self, analyzer: Analyzer) -> AnalysisPipeline:
        """Register an analyzer with the pipeline.

        Args:
            analyzer: Analyzer instance conforming to Analyzer protocol

        Returns:
            Self for method chaining
        """
        self._analyzers.append(analyzer)
        return self

    def register_secondary(
        self,
        analyzer: SecondaryAnalyzer,
    ) -> AnalysisPipeline:
        """Register a secondary analyzer with the pipeline.

        Secondary analyzers run after all per-epoch primary analysis completes
        (Phase 1.5). They consume per-epoch artifacts from a named primary
        analyzer and produce new per-epoch artifacts — without loading models.

        Args:
            analyzer: SecondaryAnalyzer instance

        Returns:
            Self for method chaining
        """
        self._secondary_analyzers.append(analyzer)
        return self

    def register_cross_epoch(
        self,
        analyzer: CrossEpochAnalyzer,
    ) -> AnalysisPipeline:
        """Register a cross-epoch analyzer with the pipeline.

        Cross-epoch analyzers run after all per-epoch analysis completes.
        They consume per-epoch artifacts to produce cross-epoch results.

        Args:
            analyzer: CrossEpochAnalyzer instance

        Returns:
            Self for method chaining
        """
        self._cross_epoch_analyzers.append(analyzer)
        return self

    def run(
        self,
        force: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Execute analysis pipeline across checkpoints.

        Each analyzer result is saved immediately to disk as a per-epoch file.
        No in-memory buffer is maintained across epochs.

        Args:
            force: If True, recompute even if artifacts exist
            progress_callback: Optional callback(progress, description) for UI updates.
                               Progress is a float from 0.0 to 1.0.
        """
        if not self._analyzers and not self._cross_epoch_analyzers:
            return

        available_epochs = self.variant.get_available_checkpoints()
        if self.config.checkpoints is not None:
            target_epochs = [e for e in self.config.checkpoints if e in available_epochs]
        else:
            target_epochs = available_epochs

        if not target_epochs:
            return

        # Phase 1: Per-epoch analysis
        work_queue = self._build_work_queue(target_epochs, force)
        context = self.variant.family.prepare_analysis_context(
            self.variant.params,
            self._device,
        )

        if work_queue:
            all_epochs_needed = sorted(set(e for _, needed in work_queue for e in needed))

            probe = self.variant.family.generate_analysis_dataset(
                self.variant.params,
                device=self._device,
            )

            summary_collectors = self._build_summary_collectors(work_queue)

            total_epochs = len(all_epochs_needed)
            for i, epoch in enumerate(tqdm.tqdm(all_epochs_needed, desc="Analyzing checkpoints")):
                if progress_callback:
                    progress_callback(
                        i / total_epochs,
                        f"Analyzing checkpoint {epoch} ({i + 1}/{total_epochs})",
                    )

                self._run_single_epoch(
                    epoch,
                    work_queue,
                    probe,
                    context,
                    summary_collectors,
                )

            for analyzer_name, collector in summary_collectors.items():
                if collector["epochs"]:
                    self._save_summary(analyzer_name, collector)

        # Phase 1.5: Secondary analysis (REQ_048)
        if self._secondary_analyzers:
            self._run_secondary_analyzers(context, force, progress_callback)

        # Phase 2: Cross-epoch analysis (REQ_038)
        if self._cross_epoch_analyzers:
            self._run_cross_epoch_analyzers(context, force, progress_callback)

        # Save manifest with metadata at end of run
        if work_queue:
            self._update_manifest(work_queue)
        self._save_manifest()

        if progress_callback:
            progress_callback(1.0, "Analysis complete")

    def get_completed_epochs(self, analyzer_name: str) -> list[int]:
        """Return list of epochs with completed analysis for given analyzer.

        Determines completion from file existence on disk.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Sorted list of completed epoch numbers
        """
        analyzer_dir = os.path.join(self.artifacts_dir, analyzer_name)
        if not os.path.isdir(analyzer_dir):
            return []

        epochs = []
        for filename in os.listdir(analyzer_dir):
            if filename.startswith("epoch_") and filename.endswith(".npz"):
                epoch_str = filename[len("epoch_") : -len(".npz")]
                try:
                    epochs.append(int(epoch_str))
                except ValueError:
                    continue
        return sorted(epochs)

    def _build_work_queue(
        self, target_epochs: list[int], force: bool
    ) -> list[tuple[Analyzer, list[int]]]:
        """Build work queue of (analyzer, epochs_needed) tuples."""
        work_queue = []

        for analyzer in self._analyzers:
            if force:
                missing = target_epochs
            else:
                completed = set(self.get_completed_epochs(analyzer.name))
                missing = [e for e in target_epochs if e not in completed]

            if missing:
                work_queue.append((analyzer, missing))

        return work_queue

    def _run_single_epoch(
        self,
        epoch: int,
        work_queue: list[tuple[Analyzer, list[int]]],
        probe: torch.Tensor,
        context: dict[str, Any],
        summary_collectors: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Run all relevant analyzers on a single checkpoint.

        Saves each result immediately to disk, then cleans up GPU memory.
        If summary_collectors is provided, computes and accumulates summary
        statistics for analyzers that support them.
        """
        state_dict = self.variant.load_checkpoint(epoch)
        model = self.variant.family.create_model(self.variant.params, device=self._device)
        model.load_state_dict(state_dict)

        with torch.inference_mode():
            _, cache = model.run_with_cache(probe)

        for analyzer, needed_epochs in work_queue:
            if epoch in needed_epochs:
                result = analyzer.analyze(model, probe, cache, context)
                self._save_epoch_artifact(analyzer.name, epoch, result)

                if summary_collectors and analyzer.name in summary_collectors:
                    summary = analyzer.compute_summary(result, context)  # type: ignore[attr-defined]
                    collector = summary_collectors[analyzer.name]
                    collector["epochs"].append(epoch)
                    for key, value in summary.items():
                        collector["values"][key].append(value)

        # Explicit cleanup to prevent GPU memory accumulation
        del model, cache, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_epoch_artifact(
        self, analyzer_name: str, epoch: int, result: dict[str, np.ndarray]
    ) -> None:
        """Save a single epoch's analysis result to disk.

        Writes to: artifacts/{analyzer_name}/epoch_{NNNNN}.npz

        Args:
            analyzer_name: Name of the analyzer
            epoch: Epoch number
            result: Dict of numpy arrays from the analyzer
        """
        analyzer_dir = os.path.join(self.artifacts_dir, analyzer_name)
        os.makedirs(analyzer_dir, exist_ok=True)

        artifact_path = os.path.join(analyzer_dir, f"epoch_{epoch:05d}.npz")
        temp_base = os.path.join(analyzer_dir, f".epoch_{epoch:05d}_tmp")
        np.savez_compressed(temp_base, **result)  # type: ignore[arg-type]
        temp_path = temp_base + ".npz"
        os.replace(temp_path, artifact_path)

    def _update_manifest(self, work_queue: list[tuple[Analyzer, list[int]]]) -> None:
        """Update manifest with metadata for all analyzers that ran."""
        if "analyzers" not in self._manifest:
            self._manifest["analyzers"] = {}

        for analyzer, _ in work_queue:
            completed = self.get_completed_epochs(analyzer.name)
            if not completed:
                continue

            # Load one epoch to get shapes and dtypes
            sample_path = os.path.join(
                self.artifacts_dir, analyzer.name, f"epoch_{completed[0]:05d}.npz"
            )
            sample = dict(np.load(sample_path))
            shapes = {k: list(v.shape) for k, v in sample.items()}
            dtypes = {k: str(v.dtype) for k, v in sample.items()}

            self._manifest["analyzers"][analyzer.name] = {
                "epochs_completed": completed,
                "shapes": shapes,
                "dtypes": dtypes,
                "updated_at": datetime.now(UTC).isoformat(),
            }

    def _save_manifest(self) -> None:
        """Save manifest to disk atomically."""
        self._manifest["variant_params"] = self.variant.params
        self._manifest["family_name"] = self.variant.family.name

        manifest_path = os.path.join(self.artifacts_dir, "manifest.json")
        temp_path = manifest_path + ".tmp"

        with open(temp_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

        os.replace(temp_path, manifest_path)

    def _load_manifest(self) -> dict:
        """Load manifest from disk, or return empty dict."""
        manifest_path = os.path.join(self.artifacts_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                return json.load(f)

        return {}

    def _build_summary_collectors(
        self, work_queue: list[tuple[Analyzer, list[int]]]
    ) -> dict[str, dict[str, Any]]:
        """Build in-memory collectors for analyzers that produce summary stats."""
        collectors: dict[str, dict[str, Any]] = {}
        for analyzer, _ in work_queue:
            if hasattr(analyzer, "get_summary_keys"):
                keys = analyzer.get_summary_keys()  # type: ignore[attr-defined]
                if keys:
                    collectors[analyzer.name] = {
                        "epochs": [],
                        "values": {k: [] for k in keys},
                    }
        return collectors

    def _save_summary(self, analyzer_name: str, collector: dict[str, Any]) -> None:
        """Save accumulated summary statistics to summary.npz.

        Merges with any existing summary data for gap-filling support.
        """
        new_epochs = np.array(collector["epochs"])
        new_values = {k: np.array(v) for k, v in collector["values"].items()}

        existing = self._load_existing_summary(analyzer_name)
        if existing is not None:
            old_epochs = existing["epochs"]
            # Find epochs not already present
            old_set = set(old_epochs.tolist())
            keep_mask = np.array([e not in old_set for e in new_epochs])

            if keep_mask.any():
                # Append genuinely new epochs; for new keys, pad old epochs with zeros.
                merged_epochs = np.concatenate([old_epochs, new_epochs[keep_mask]])
                merged_values = {}
                for k in new_values:
                    if k in existing:
                        merged_values[k] = np.concatenate([existing[k], new_values[k][keep_mask]])
                    else:
                        old_fill = np.zeros(len(old_epochs), dtype=new_values[k].dtype)
                        merged_values[k] = np.concatenate([old_fill, new_values[k][keep_mask]])
            else:
                # No new epochs — new_values covers the full existing epoch set
                # (e.g. force=True rerun). Prefer new_values; keep old for absent keys.
                merged_epochs = old_epochs
                merged_values = {}
                for k in new_values:
                    merged_values[k] = new_values[k]

            # Sort by epoch
            sort_idx = np.argsort(merged_epochs)
            new_epochs = merged_epochs[sort_idx]
            new_values = {k: v[sort_idx] for k, v in merged_values.items()}

        analyzer_dir = os.path.join(self.artifacts_dir, analyzer_name)
        summary_path = os.path.join(analyzer_dir, "summary.npz")
        temp_base = os.path.join(analyzer_dir, ".summary_tmp")
        np.savez_compressed(temp_base, epochs=new_epochs, **new_values)  # type: ignore[arg-type]
        os.replace(temp_base + ".npz", summary_path)

    def _load_existing_summary(self, analyzer_name: str) -> dict[str, np.ndarray] | None:
        """Load existing summary.npz if present, or return None."""
        summary_path = os.path.join(self.artifacts_dir, analyzer_name, "summary.npz")
        if not os.path.exists(summary_path):
            return None
        return dict(np.load(summary_path))

    # ------------------------------------------------------------------
    # Phase 1.5: Secondary analysis (REQ_048)
    # ------------------------------------------------------------------

    def _run_secondary_analyzers(
        self,
        context: dict[str, Any],
        force: bool,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Execute secondary analyzers after per-epoch primary phase completes.

        Each secondary analyzer loads per-epoch artifact data from its declared
        dependency and produces new per-epoch artifacts. No model loading occurs.

        Target epochs are derived from the dependency's completed set, not from
        config.checkpoints. If a dependency has no completed epochs, the analyzer
        is skipped with a warning.
        """
        from miscope.analysis.artifact_loader import ArtifactLoader

        loader = ArtifactLoader(self.artifacts_dir)

        config_names = set(self.config.analyzers) if self.config.analyzers else None

        for analyzer in self._secondary_analyzers:
            if config_names is not None and analyzer.name not in config_names:
                continue

            dependency_epochs = self.get_completed_epochs(analyzer.depends_on)
            if not dependency_epochs:
                logger.warning(
                    "Secondary analyzer '%s' depends on '%s' but no epochs have been computed "
                    "for that analyzer. Skipping.",
                    analyzer.name,
                    analyzer.depends_on,
                )
                continue

            if force:
                target_epochs = dependency_epochs
            else:
                completed = set(self.get_completed_epochs(analyzer.name))
                target_epochs = [e for e in dependency_epochs if e not in completed]

            if not target_epochs:
                continue

            summary_collectors: dict[str, dict[str, Any]] = {}
            if hasattr(analyzer, "get_summary_keys"):
                keys = analyzer.get_summary_keys()  # type: ignore[attr-defined]
                if keys:
                    summary_collectors[analyzer.name] = {
                        "epochs": [],
                        "values": {k: [] for k in keys},
                    }

            for i, epoch in enumerate(tqdm.tqdm(target_epochs, desc=f"Secondary: {analyzer.name}")):
                if progress_callback:
                    progress_callback(
                        0.8 + 0.1 * i / len(target_epochs),
                        f"Secondary analysis: {analyzer.name} epoch {epoch}",
                    )

                artifact = loader.load_epoch(analyzer.depends_on, epoch)
                result = analyzer.analyze(artifact, context)
                self._save_epoch_artifact(analyzer.name, epoch, result)

                if analyzer.name in summary_collectors:
                    summary = analyzer.compute_summary(result, context)  # type: ignore[attr-defined]
                    collector = summary_collectors[analyzer.name]
                    collector["epochs"].append(epoch)
                    for key, value in summary.items():
                        collector["values"][key].append(value)

            for analyzer_name, collector in summary_collectors.items():
                if collector["epochs"]:
                    self._save_summary(analyzer_name, collector)

    # ------------------------------------------------------------------
    # Phase 2: Cross-epoch analysis (REQ_038)
    # ------------------------------------------------------------------

    def _run_cross_epoch_analyzers(
        self,
        context: dict[str, Any],
        force: bool,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Execute cross-epoch analyzers after per-epoch phase completes.

        Validates that required per-epoch analyzers have completed,
        then runs each cross-epoch analyzer and saves results.
        """
        available_epochs = sorted(self.variant.get_available_checkpoints())

        for analyzer in self._cross_epoch_analyzers:
            # Skip if already computed (unless force)
            cross_epoch_path = os.path.join(
                self.artifacts_dir,
                analyzer.name,
                "cross_epoch.npz",
            )
            if os.path.exists(cross_epoch_path) and not force:
                continue

            # Validate dependencies
            for required in analyzer.requires:
                completed = self.get_completed_epochs(required)
                if not completed:
                    raise RuntimeError(
                        f"Cross-epoch analyzer '{analyzer.name}' requires "
                        f"'{required}' but no epochs have been analyzed."
                    )

            if progress_callback:
                progress_callback(
                    0.95,
                    f"Running cross-epoch analysis: {analyzer.name}",
                )

            result = analyzer.analyze_across_epochs(
                self.artifacts_dir,
                available_epochs,
                context,
            )
            self._save_cross_epoch_artifact(analyzer.name, result)

    def _save_cross_epoch_artifact(
        self,
        analyzer_name: str,
        result: dict[str, np.ndarray],
    ) -> None:
        """Save cross-epoch analysis result to disk.

        Writes to: artifacts/{analyzer_name}/cross_epoch.npz
        """
        analyzer_dir = os.path.join(self.artifacts_dir, analyzer_name)
        os.makedirs(analyzer_dir, exist_ok=True)

        cross_epoch_path = os.path.join(analyzer_dir, "cross_epoch.npz")
        temp_base = os.path.join(analyzer_dir, ".cross_epoch_tmp")
        np.savez_compressed(temp_base, **result)  # type: ignore[arg-type]
        os.replace(temp_base + ".npz", cross_epoch_path)
