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
from tqdm import tqdm

from miscope.analysis.artifact_loader import analyzer_dir
from miscope.analysis.inputs import ResolvedInputs
from miscope.analysis.parameters import EMPTY_PARAMETERIZATION, Parameterization
from miscope.analysis.planner import Plan, PlanItem, plan_analysis
from miscope.analysis.protocols import (
    AnalysisRunConfig,
    Analyzer,
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
        self._cross_epoch_analyzers: list[Analyzer] = []
        self._manifest: dict[str, Any] = {}

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # REQ_138: the run set + its per-analyzer recipe signatures. Default is the
        # empty parameterization (every analyzer at today's path); set in ``run()``.
        self._parameterization: Parameterization = EMPTY_PARAMETERIZATION
        self._recipe_map: dict[str, str] = {}

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._manifest = self._load_manifest()

    def _recipe_dir(self, analyzer_name: str) -> str:
        """Recipe-scoped write/scan directory for an analyzer (REQ_138 storage primitive)."""
        return analyzer_dir(
            self.artifacts_dir, analyzer_name, self._recipe_map.get(analyzer_name, "")
        )

    def register(self, analyzer: Analyzer) -> AnalysisPipeline:
        """Register an analyzer with the pipeline.

        A single registration verb (REQ_132): the pipeline routes the
        analyzer by the only structural axis, its registered Spec's
        ``output_scope`` (REQ_133). Per-epoch analyzers — model-driven or
        purely artifact-derived — share one bucket and one execution pass,
        topologically ordered by the planner. Callers do not pre-sort.

        Every analyzer must carry a registered Spec (via ``@register_analyzer``)
        — a Spec is mandatory at execution time. Registering an analyzer with
        no Spec raises ``ValueError``.

        Args:
            analyzer: Analyzer instance conforming to the Analyzer protocol.

        Returns:
            Self for method chaining.
        """
        from miscope.analysis.registry import AnalyzerRegistry

        if not AnalyzerRegistry.has_spec(analyzer.name):
            raise ValueError(
                f"Analyzer '{analyzer.name}' has no registered Spec. Every analyzer "
                "must register via @register_analyzer before use (REQ_132)."
            )
        spec = AnalyzerRegistry.get_spec(analyzer.name)
        if spec.output_scope == "cross_epoch":
            self._cross_epoch_analyzers.append(analyzer)
        else:
            self._analyzers.append(analyzer)
        return self

    def run(
        self,
        force: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
        parameterization: Parameterization | None = None,
        plan: Plan | None = None,
    ) -> None:
        """Execute analysis pipeline across checkpoints.

        Each analyzer result is saved immediately to disk as a per-epoch file.
        No in-memory buffer is maintained across epochs.

        Args:
            force: If True, recompute even if artifacts exist. Ignored when
                ``plan`` is provided (the plan already reflects the desired
                force state).
            progress_callback: Optional callback(progress, description) for UI updates.
                               Progress is a float from 0.0 to 1.0.
            parameterization: Optional run set (REQ_138). Its bindings project to a
                per-analyzer recipe; an analyzer with a non-default binding in its
                closure writes/reads under a recipe-scoped path and is planned
                independently, so parameterizations coexist. The default (``None`` →
                empty parameterization) resolves every analyzer to today's path and
                its declared parameter defaults — byte-identical to prior behavior.
            plan: Optional pre-built ``Plan`` describing the work to perform.
                If ``None``, the pipeline calls ``plan_analysis`` on its
                registered analyzers. Passing a plan lets callers preview
                work before execution (REQ_119).
        """
        if plan is None and not self._analyzers and not self._cross_epoch_analyzers:
            return

        self._parameterization = parameterization or EMPTY_PARAMETERIZATION
        self._recipe_map = self._build_recipe_map()
        self._resolved_params: dict[str, dict[str, Any]] = {}

        if plan is None:
            plan = self._build_plan(force)

        # When the caller hands in a Spec-built Plan, the pipeline may need
        # to instantiate analyzers from the Registry that were never passed
        # to register_*. Do so up front so phase loops can look them up.
        self._absorb_plan_references(plan)

        # Preserve the pre-REQ_119 early-return: when no checkpoints are
        # available, skip everything (including cross-epoch and secondary).
        if not plan.target_epochs:
            return

        # One per-epoch pass over all per-epoch analyzers (model-driven and
        # purely artifact-derived alike), in the planner's topological order so
        # an artifact-derived analyzer's upstream has written ``epoch_N`` before
        # it reads it in the same epoch iteration (REQ_133). Blocked items
        # (empty epochs, non-empty ``blocked_by``) are logged and skipped.
        per_epoch_by_name = {a.name: a for a in self._analyzers}
        for item in plan.per_epoch:
            if item.blocked_by:
                logger.warning(
                    "Per-epoch analyzer '%s' depends on %s but no epochs have been "
                    "computed for that upstream. Skipping.",
                    item.analyzer_name,
                    ", ".join(item.blocked_by),
                )
        work_queue: list[tuple[Analyzer, list[int]]] = [
            (per_epoch_by_name[item.analyzer_name], list(item.epochs))
            for item in plan.per_epoch
            if item.analyzer_name in per_epoch_by_name and item.epochs
        ]

        context = self.variant.family.prepare_analysis_context(
            self.variant.params,
            self._device,
        )

        if work_queue:
            all_epochs_needed = sorted(set(e for _, needed in work_queue for e in needed))

            # REQ_120: the Plan tells us whether any active primary analyzer
            # reads ctx.cache. When none do, skip the forward pass entirely.
            # ``needs_activation_cache`` defaults to True for any item whose
            # Spec flag is unknown (legacy Analyzer instances), preserving
            # today's behavior.
            needs_cache = plan.needs_activation_cache

            probe = self.variant.family.generate_analysis_dataset(
                self.variant.params,
                device=self._device,
            )

            summary_collectors = self._build_summary_collectors(work_queue)

            total_epochs = len(all_epochs_needed)
            for i, epoch in enumerate(tqdm(all_epochs_needed, desc="Analyzing checkpoints")):
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
                    needs_cache=needs_cache,
                )

            for analyzer_name, collector in summary_collectors.items():
                if collector["epochs"]:
                    self._save_summary(analyzer_name, collector)

        # Cross-epoch analysis (REQ_038) — runs after every per-epoch artifact
        # is on disk, in the planner's topological order so a cross→cross
        # upstream's ``cross_epoch.npz`` exists before its dependent reads it.
        if self._cross_epoch_analyzers and plan.cross_epoch:
            self._run_cross_epoch_from_plan(plan.cross_epoch, context, progress_callback)

        # Save manifest with metadata at end of run
        if work_queue:
            self._update_manifest(work_queue)
        self._save_manifest()

        self._record_run_set()

        if progress_callback:
            progress_callback(1.0, "Analysis complete")

    def _record_run_set(self) -> None:
        """Persist this run's run set to the registry (REQ_138; no-op for default)."""
        if self._parameterization.is_empty or not self._recipe_map:
            return
        from miscope.analysis.registry import AnalyzerRegistry
        from miscope.warehouse import run_sets

        specs = {s.name: s for s in AnalyzerRegistry.list_specs()}
        run_sets.record_run_set(
            self.variant,
            self._parameterization,
            self._recipe_map,
            self._resolved_params,
            specs,
        )

    def _absorb_plan_references(self, plan: Plan) -> None:
        """Instantiate Spec-only analyzers referenced by a Plan (REQ_120).

        When the caller built the Plan from Specs (via
        ``plan_analysis(variant, registry.list_for_family(family))``) but
        did not also call ``pipeline.register_*`` for each one, look up
        the missing factories in the Registry and register the instances.
        This makes the canonical entry-point pattern a one-liner.
        """
        from miscope.analysis.registry import AnalyzerRegistry

        per_epoch_names = {a.name for a in self._analyzers}
        cross_epoch_names = {a.name for a in self._cross_epoch_analyzers}

        for item in plan.per_epoch:
            if item.analyzer_name in per_epoch_names:
                continue
            if AnalyzerRegistry.has_spec(item.analyzer_name):
                self._analyzers.append(AnalyzerRegistry.create(item.analyzer_name))

        for item in plan.cross_epoch:
            if item.analyzer_name in cross_epoch_names:
                continue
            if AnalyzerRegistry.has_spec(item.analyzer_name):
                self._cross_epoch_analyzers.append(AnalyzerRegistry.create(item.analyzer_name))

    def _build_plan(self, force: bool) -> Plan:
        """Build a Plan from the pipeline's registered analyzers.

        Preserves the historical ``config.analyzers`` asymmetry (REQ_119): the
        filter narrows only the purely artifact-derived per-epoch analyzers (the
        former "secondary" set); model-driven and cross-epoch analyzers ignore
        it. The former-secondary set is reconstructed from each analyzer's Spec
        (per-epoch output, artifact inputs, no ``ModelInput``).
        """
        per_epoch = self._filter_per_epoch_by_config(self._analyzers)
        analyzers: list[Any] = [*per_epoch, *self._cross_epoch_analyzers]
        return plan_analysis(
            self.variant,
            analyzers,
            force=force,
            checkpoints=self.config.checkpoints,
            recipe_map=self._recipe_map,
        )

    def _build_recipe_map(self) -> dict[str, str]:
        """Per-analyzer recipe signatures for this run set (REQ_138).

        Empty for the default parameterization (every analyzer at today's path).
        Otherwise maps only the analyzers whose recipe is non-empty (a non-default
        binding in their closure) to their signature; everything else stays shared.
        """
        if self._parameterization.is_empty:
            return {}
        from miscope.analysis.recipe import project_recipe
        from miscope.analysis.registry import AnalyzerRegistry

        specs = {s.name: s for s in AnalyzerRegistry.list_specs()}
        names = [a.name for a in (*self._analyzers, *self._cross_epoch_analyzers)]
        out: dict[str, str] = {}
        for name in names:
            sig = project_recipe(name, self._parameterization, specs).signature()
            if sig:
                out[name] = sig
        return out

    def _filter_per_epoch_by_config(self, analyzers: list[Analyzer]) -> list[Analyzer]:
        """Drop artifact-derived per-epoch analyzers excluded by config.analyzers."""
        if not self.config.analyzers:
            return list(analyzers)
        from miscope.analysis.inputs import derive_has_model_input

        config_names = set(self.config.analyzers)
        kept: list[Analyzer] = []
        for analyzer in analyzers:
            spec = self._spec_for(analyzer)
            artifact_only = (
                spec is not None and not derive_has_model_input(spec.inputs) and bool(spec.requires)
            )
            if artifact_only and analyzer.name not in config_names:
                continue
            kept.append(analyzer)
        return kept

    def get_completed_epochs(self, analyzer_name: str) -> list[int]:
        """Return list of epochs with completed analysis for given analyzer.

        Determines completion from file existence on disk.  For cross-epoch
        analyzers that produce only cross_epoch.npz (no per-epoch files), the
        available checkpoint epochs are returned when cross_epoch.npz exists,
        so that downstream cross-epoch analyzers can treat the dependency as
        satisfied.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Sorted list of completed epoch numbers
        """
        scan_dir = self._recipe_dir(analyzer_name)
        if not os.path.isdir(scan_dir):
            return []

        epochs = []
        for filename in os.listdir(scan_dir):
            if filename.startswith("epoch_") and filename.endswith(".npz"):
                epoch_str = filename[len("epoch_") : -len(".npz")]
                try:
                    epochs.append(int(epoch_str))
                except ValueError:
                    continue

        if epochs:
            return sorted(epochs)

        # No per-epoch files — check for a cross-epoch artifact.  If present,
        # report the available checkpoint epochs so cross-epoch-to-cross-epoch
        # dependencies are satisfied.
        cross_epoch_path = os.path.join(scan_dir, "cross_epoch.npz")
        if os.path.exists(cross_epoch_path):
            return sorted(self.variant.get_available_checkpoints())

        return []

    def _run_single_epoch(
        self,
        epoch: int,
        work_queue: list[tuple[Analyzer, list[int]]],
        probe: torch.Tensor,
        context: dict[str, Any],
        summary_collectors: dict[str, dict[str, Any]] | None = None,
        needs_cache: bool = True,
    ) -> None:
        """Run all relevant analyzers on a single checkpoint.

        Saves each result immediately to disk, then cleans up GPU memory.
        If summary_collectors is provided, computes and accumulates summary
        statistics for analyzers that support them.

        An analyzer that declares ``required_hooks`` is filtered: if its
        declared canonical hooks are not all published by the current model
        it is skipped with an info-level log entry. Analyzers declaring no
        ``required_hooks`` run unconditionally.

        REQ_120: when ``needs_cache=False`` (no analyzer at this epoch
        reads ``ctx.cache`` or ``ctx.logits`` per their Specs), the
        forward pass is skipped — ``ctx.cache`` and ``ctx.logits`` are
        ``None``. Analyzers that quietly read the cache without declaring
        it on their Spec will fail; the audit step (Phase 2) is how that
        risk is bounded.
        """
        state_dict = self.variant.load_checkpoint(epoch)
        model = self.variant.family.create_model(self.variant.params, device=self._device)
        model.load_state_dict(state_dict)

        if needs_cache:
            with torch.inference_mode():
                logits, cache = model.run_with_cache(probe)
        else:
            logits, cache = None, None

        from miscope.analysis.registry import AnalyzerRegistry

        for analyzer, needed_epochs in work_queue:
            if epoch not in needed_epochs:
                continue

            spec = AnalyzerRegistry.get_spec(analyzer.name)

            required = spec.required_hooks
            if required:
                missing = [h for h in required if h not in model.hook_names()]
                if missing:
                    logger.info(
                        "Skipping %s on epoch %d: missing canonical hooks %s.",
                        analyzer.name,
                        epoch,
                        missing,
                    )
                    continue

            inputs = self._materialize_per_epoch_inputs(spec, epoch, model, cache, logits, probe)
            result = analyzer.analyze(inputs, context)

            self._save_epoch_artifact(analyzer.name, epoch, result)

            if summary_collectors and analyzer.name in summary_collectors:
                summary = analyzer.compute_summary(result, context)  # type: ignore[attr-defined]
                collector = summary_collectors[analyzer.name]
                collector["epochs"].append(epoch)
                for key, value in summary.items():
                    collector["values"][key].append(value)

        # Explicit cleanup to prevent GPU memory accumulation. ResolvedInputs
        # still carries the eager model side (model/cache/logits); the last
        # iteration's `inputs`/`result`/`summary` hold those references, so
        # clearing them lets the subsequent `del model, cache, ...` release the
        # objects and empty_cache() reclaim the GPU memory. (The artifact side
        # is now lazy via `deps`, so it no longer contributes here — REQ_128.)
        inputs = result = summary = None  # noqa: F841 — drop trailing refs
        del model, cache, logits, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _materialize_per_epoch_inputs(
        self,
        spec: Any,
        epoch: int,
        model: Any,
        cache: Any,
        logits: Any,
        probe: torch.Tensor,
    ) -> ResolvedInputs:
        """Build a ResolvedInputs for a per-epoch analyzer.

        Upstream artifacts are reached lazily through ``inputs.deps``; only the
        model side (from the forward pass the pipeline runs anyway) is eager.
        A registered Spec is mandatory at execution time (REQ_132): the deps
        scope and capability needs are derived entirely from ``spec.inputs``.
        """
        from miscope.analysis.artifact_loader import ArtifactLoader
        from miscope.analysis.deps import DepsAccessor
        from miscope.analysis.inputs import ResolvedInputs, derive_required_artifacts

        loader = ArtifactLoader(self.artifacts_dir, recipe_map=self._recipe_map)
        declared = frozenset(derive_required_artifacts(spec.inputs))
        deps = DepsAccessor(loader, declared)

        # Capability needs are derived from the Spec's ``inputs`` declaration
        # (each property ORs over every declared ModelInput).
        wants_model = spec.requires_model_weights
        wants_cache = spec.requires_activation_cache

        return ResolvedInputs(
            epoch=epoch,
            model=model if wants_model else None,
            cache=cache if wants_cache else None,
            logits=logits if wants_cache else None,
            probe=probe,
            deps=deps,
            parameters=self._resolve_parameters(spec, loader),
        )

    def _resolve_parameters(self, spec: Any, loader: Any) -> dict[str, Any]:
        """Resolve an analyzer's declared parameters under the run set (REQ_138).

        Each declared parameter resolves its binding — a run-set override if the
        parameterization supplies one, otherwise the declared default. A default is
        a binding, resolved the same way (no code fallback), which is what makes the
        p101 silent-default divergence impossible.
        """
        if not spec.parameters:
            return {}
        from miscope.analysis.recipe import RecipeResolver

        resolver = RecipeResolver(loader)
        resolved: dict[str, Any] = {}
        for param in spec.parameters:
            binding = self._parameterization.binding_for(spec.name, param.name) or param.default
            resolved[param.name] = resolver.resolve(binding)
        # Capture for the run-set registry (REQ_138 provenance).
        if resolved:
            self._resolved_params[spec.name] = resolved
        return resolved

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
        out_dir = self._recipe_dir(analyzer_name)
        os.makedirs(out_dir, exist_ok=True)

        artifact_path = os.path.join(out_dir, f"epoch_{epoch:05d}.npz")
        temp_base = os.path.join(out_dir, f".epoch_{epoch:05d}_tmp")
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
                self._recipe_dir(analyzer.name), f"epoch_{completed[0]:05d}.npz"
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

        out_dir = self._recipe_dir(analyzer_name)
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary.npz")
        temp_base = os.path.join(out_dir, ".summary_tmp")
        np.savez_compressed(temp_base, epochs=new_epochs, **new_values)  # type: ignore[arg-type]
        os.replace(temp_base + ".npz", summary_path)

    def _load_existing_summary(self, analyzer_name: str) -> dict[str, np.ndarray] | None:
        """Load existing summary.npz if present, or return None."""
        summary_path = os.path.join(self._recipe_dir(analyzer_name), "summary.npz")
        if not os.path.exists(summary_path):
            return None
        return dict(np.load(summary_path))

    def _spec_for(self, analyzer: Analyzer):
        """Look up an analyzer's Spec from the Registry (or None)."""
        from miscope.analysis.registry import AnalyzerRegistry

        return (
            AnalyzerRegistry.get_spec(analyzer.name)
            if AnalyzerRegistry.has_spec(analyzer.name)
            else None
        )

    # ------------------------------------------------------------------
    # Cross-epoch analysis (REQ_038)
    # ------------------------------------------------------------------

    def _run_cross_epoch_from_plan(
        self,
        items: list[PlanItem],
        context: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Execute cross-epoch analyzers as described by Plan items.

        A blocked item raises ``RuntimeError`` — preserving the existing
        behavior of the pipeline. The choice of raise vs. skip moves to a
        configurable knob in step 1.5 (REQ_119 Notes).
        """
        available_epochs = sorted(self.variant.get_available_checkpoints())
        # Inject variant so analyzers that load checkpoints directly can access it
        cross_epoch_context = {**context, "variant": self.variant}
        cross_epoch_by_name = {a.name: a for a in self._cross_epoch_analyzers}

        for item in items:
            analyzer = cross_epoch_by_name.get(item.analyzer_name)
            if analyzer is None:
                continue

            if item.blocked_by:
                missing = item.blocked_by[0]
                raise RuntimeError(
                    f"Cross-epoch analyzer '{analyzer.name}' requires "
                    f"'{missing}' but no epochs have been analyzed."
                )

            if progress_callback:
                progress_callback(
                    0.95,
                    f"Running cross-epoch analysis: {analyzer.name}",
                )

            from miscope.analysis.registry import AnalyzerRegistry

            spec = AnalyzerRegistry.get_spec(analyzer.name)
            inputs = self._materialize_cross_epoch_inputs(spec, available_epochs)
            result = analyzer.analyze(inputs, cross_epoch_context)
            self._save_cross_epoch_artifact(analyzer.name, result)

    def _materialize_cross_epoch_inputs(
        self,
        spec: Any,
        available_epochs: list[int],
    ) -> ResolvedInputs:
        """Build a ResolvedInputs for a cross-epoch analyzer.

        Upstreams are reached lazily through ``inputs.deps`` (scoped to the
        analyzer's declared ``ArtifactInput``s). The pipeline pre-materializes
        nothing — that eager whole-stack load was the REQ_128 memory suspect.
        A registered Spec is mandatory at execution time (REQ_132).
        """
        from miscope.analysis.artifact_loader import ArtifactLoader
        from miscope.analysis.deps import DepsAccessor
        from miscope.analysis.inputs import ResolvedInputs, derive_required_artifacts

        loader = ArtifactLoader(self.artifacts_dir, recipe_map=self._recipe_map)
        allowed = frozenset(derive_required_artifacts(spec.inputs))
        deps = DepsAccessor(loader, allowed)

        return ResolvedInputs(
            epoch=None,
            epochs=tuple(available_epochs),
            deps=deps,
            parameters=self._resolve_parameters(spec, loader),
        )

    def _save_cross_epoch_artifact(
        self,
        analyzer_name: str,
        result: dict[str, np.ndarray],
    ) -> None:
        """Save cross-epoch analysis result to disk.

        Writes to: artifacts/{analyzer_name}/cross_epoch.npz
        """
        out_dir = self._recipe_dir(analyzer_name)
        os.makedirs(out_dir, exist_ok=True)

        cross_epoch_path = os.path.join(out_dir, "cross_epoch.npz")
        temp_base = os.path.join(out_dir, ".cross_epoch_tmp")
        np.savez_compressed(temp_base, **result)  # type: ignore[arg-type]
        os.replace(temp_base + ".npz", cross_epoch_path)
