"""Internal storage primitive: ArtifactLoader.

Consumers should access a configured loader via ``variant.artifacts`` rather
than importing this class directly. The class is part of the storage layer
and is used internally by the analysis pipeline and cross-epoch analyzers;
only the public-API signal changed in REQ_125 — the class itself, its
methods, and its on-disk contract are unchanged.

Supports per-epoch artifact storage where each analyzer's results
are stored as individual files per epoch:
    artifacts/{analyzer_name}/epoch_{NNNNN}.npz

Also supports summary statistics (REQ_022) stored as a single file:
    artifacts/{analyzer_name}/summary.npz
"""

import json
import os
from typing import Any

import numpy as np

RECIPE_DIR_PREFIX = "__rs_"


def analyzer_dir(artifacts_dir: str, analyzer: str, recipe_sig: str = "") -> str:
    """The directory holding one analyzer's blobs, recipe-scoped (REQ_138).

    The single place the recipe path segment is composed (storage-encapsulation
    invariant 3): every reader, writer, and scanner reaches an analyzer's blob
    container through this helper rather than joining ``artifacts_dir`` itself.

    An **empty** ``recipe_sig`` (the default/all-defaults parameterization) returns
    today's path ``{artifacts_dir}/{analyzer}`` exactly, so the existing artifacts
    never relocate. A non-empty signature nests a ``__rs_{sig}`` segment so
    coexisting parameterizations of one analyzer never overwrite each other.
    """
    base = os.path.join(artifacts_dir, analyzer)
    if not recipe_sig:
        return base
    return os.path.join(base, f"{RECIPE_DIR_PREFIX}{recipe_sig}")


def iter_recipe_dirs(artifacts_dir: str):
    """Yield ``(analyzer, recipe_signature, path)`` for every on-disk recipe dir.

    The reverse of :func:`analyzer_dir`: walks each analyzer directory for
    ``__rs_<sig>`` recipe segments (REQ_138). Lives in the storage primitive so
    recipe-path *decomposition* is owned in the same place as composition
    (storage-encapsulation invariant 3) — the recipe-GC tooling reaches it here
    rather than globbing ``__rs_`` itself.
    """
    if not os.path.isdir(artifacts_dir):
        return
    for analyzer in sorted(os.listdir(artifacts_dir)):
        adir = os.path.join(artifacts_dir, analyzer)
        if not os.path.isdir(adir):
            continue
        for child in sorted(os.listdir(adir)):
            if child.startswith(RECIPE_DIR_PREFIX):
                yield analyzer, child[len(RECIPE_DIR_PREFIX) :], os.path.join(adir, child)


def _validate_fields(
    analyzer_name: str, available: list[str], requested: list[str], where: str
) -> None:
    """Raise if any requested field is absent from an artifact's keys.

    Validation reads the ``.npz`` index only (``NpzFile.files``), so it is
    cheap — no array data is loaded to check field names.
    """
    available_set = set(available)
    missing = [f for f in requested if f not in available_set]
    if missing:
        raise ValueError(
            f"Artifact '{analyzer_name}' ({where}) has no field(s) {missing}. "
            f"Available fields: {sorted(available_set)}."
        )


class ArtifactLoader:
    """Loads analysis artifacts for visualization components.

    Provides both per-epoch loading (for dashboard slider interaction)
    and multi-epoch loading (for cross-epoch views and notebooks).
    """

    def __init__(self, artifacts_dir: str, recipe_map: dict[str, str] | None = None):
        """Initialize the artifact loader.

        Args:
            artifacts_dir: Path to the artifacts directory
            recipe_map: Optional ``analyzer_name -> recipe signature`` map (REQ_138).
                Every path the loader composes for an analyzer is recipe-scoped via
                this map; an absent/empty entry resolves to today's path. The default
                (``None``) is an empty map — every analyzer reads/writes its
                unparameterized location, so existing behavior is byte-identical.
        """
        self.artifacts_dir = artifacts_dir
        self._recipe_map = recipe_map or {}
        self._manifest: dict[str, Any] | None = None

    def _dir(self, analyzer_name: str) -> str:
        """Recipe-scoped blob directory for an analyzer (storage primitive)."""
        return analyzer_dir(
            self.artifacts_dir, analyzer_name, self._recipe_map.get(analyzer_name, "")
        )

    @property
    def manifest(self) -> dict[str, Any]:
        """Load and cache the manifest."""
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def load_epoch(
        self, analyzer_name: str, epoch: int, fields: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Load analysis results for a single epoch.

        Args:
            analyzer_name: Name of the analyzer (e.g., "dominant_frequencies")
            epoch: Epoch number to load
            fields: Specific field names to load. None loads all fields.
                Requested fields are validated against the file's actual keys
                (cheap — reads the ``.npz`` index, not the arrays).

        Returns:
            Dict of numpy arrays (e.g., {"coefficients": ndarray})
            Does NOT include an "epochs" key — this is single-epoch data.

        Raises:
            FileNotFoundError: If artifact for this epoch doesn't exist
            ValueError: If a requested field is absent from the artifact
        """
        artifact_path = os.path.join(self._dir(analyzer_name), f"epoch_{epoch:05d}.npz")

        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"No artifact for '{analyzer_name}' at epoch {epoch}. Expected: {artifact_path}"
            )

        if fields is None:
            return dict(np.load(artifact_path))

        with np.load(artifact_path) as npz:
            _validate_fields(analyzer_name, npz.files, fields, f"epoch {epoch}")
            return {name: npz[name] for name in fields}

    def load_epochs(
        self,
        analyzer_name: str,
        epochs: list[int] | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Load and stack results across multiple epochs.

        Loads individual per-epoch files and stacks them along axis=0.
        Useful for cross-epoch visualizations and notebook exploration.

        Args:
            analyzer_name: Name of the analyzer
            epochs: Specific epochs to load. None means all available.
            fields: Specific field names to load from each epoch file.
                None loads all fields. Use this for large artifacts (e.g.,
                parameter_snapshot) where only one field is needed — avoids
                loading all weight matrices when only W_E is required.

        Returns:
            Dict with 'epochs' array and stacked data arrays.
            E.g., {"epochs": (n,), "coefficients": (n, n_fourier)}

        Raises:
            FileNotFoundError: If no epochs available for this analyzer
        """
        if epochs is None:
            epochs = self.get_epochs(analyzer_name)

        if not epochs:
            raise FileNotFoundError(f"No artifacts found for '{analyzer_name}'")

        epochs = sorted(epochs)

        if fields is not None:
            # Selective loading: open each npz lazily and only extract requested fields.
            # Avoids loading large arrays (e.g., W_in, W_out) when only W_E is needed.
            # Validate against the first epoch's keys for a clear early error.
            first_path = os.path.join(self._dir(analyzer_name), f"epoch_{epochs[0]:05d}.npz")
            with np.load(first_path) as npz0:
                _validate_fields(analyzer_name, npz0.files, fields, f"epoch {epochs[0]}")
            result: dict[str, list[np.ndarray]] = {k: [] for k in fields}
            for epoch in epochs:
                artifact_path = os.path.join(self._dir(analyzer_name), f"epoch_{epoch:05d}.npz")
                with np.load(artifact_path) as npz:
                    for k in fields:
                        result[k].append(npz[k])
        else:
            # Load all fields via load_epoch (original behavior)
            first = self.load_epoch(analyzer_name, epochs[0])
            keys = list(first.keys())
            result = {k: [first[k]] for k in keys}
            for epoch in epochs[1:]:
                data = self.load_epoch(analyzer_name, epoch)
                for k in keys:
                    result[k].append(data[k])

        stacked: dict[str, np.ndarray] = {"epochs": np.array(epochs)}
        for k, arrays in result.items():
            stacked[k] = np.stack(arrays, axis=0)

        return stacked

    def load(self, analyzer_name: str) -> dict[str, np.ndarray]:
        """Load all epochs for an analyzer (backward-compatible alias).

        Equivalent to load_epochs(analyzer_name, epochs=None).

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Dict with 'epochs' array and stacked data arrays
        """
        return self.load_epochs(analyzer_name)

    def artifact_path(self, analyzer_name: str, epoch: int | None = None) -> str:
        """Filesystem path to an analyzer's blob container.

        ``epoch`` selects the per-epoch ``epoch_{NNNNN}.npz``; ``None`` selects
        the cross-epoch ``cross_epoch.npz``. Path composition lives here in the
        storage primitive (the storage-encapsulation invariant): the tensor
        catalog (REQ_110B) records this address as a descriptor ``uri`` rather
        than composing the path itself.

        Args:
            analyzer_name: Name of the analyzer.
            epoch: Epoch number for a per-epoch blob, or ``None`` for cross-epoch.

        Returns:
            Absolute path to the ``.npz`` container (existence not checked).
        """
        fname = "cross_epoch.npz" if epoch is None else f"epoch_{epoch:05d}.npz"
        return os.path.join(self._dir(analyzer_name), fname)

    def get_available_analyzers(self) -> list[str]:
        """List available analyzers by checking for subdirectories with artifacts.

        Returns:
            List of analyzer names with saved artifacts
        """
        if not os.path.isdir(self.artifacts_dir):
            return []

        analyzers = []
        for entry in os.listdir(self.artifacts_dir):
            # Recipe-scoped (REQ_138): a parameterized loader checks the analyzer's
            # recipe dir, so per-epoch availability reflects the active parameterization.
            scan_dir = self._dir(entry)
            if os.path.isdir(scan_dir):
                # Check that it contains at least one epoch file
                if any(f.startswith("epoch_") and f.endswith(".npz") for f in os.listdir(scan_dir)):
                    analyzers.append(entry)

        return sorted(analyzers)

    def get_epochs(self, analyzer_name: str) -> list[int]:
        """Get list of available epochs for an analyzer from filesystem.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Sorted list of epoch numbers
        """
        scan_dir = self._dir(analyzer_name)
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

        return sorted(epochs)

    def get_metadata(self, analyzer_name: str) -> dict[str, Any]:
        """Get metadata for an analyzer from manifest.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Dict with shapes, dtypes, updated_at, etc.

        Raises:
            KeyError: If analyzer not found in manifest
        """
        analyzers = self.manifest.get("analyzers", {})

        if analyzer_name not in analyzers:
            available = list(analyzers.keys())
            raise KeyError(
                f"Analyzer '{analyzer_name}' not found in manifest. Available: {available}"
            )

        return analyzers[analyzer_name]

    def load_summary(self, analyzer_name: str) -> dict[str, np.ndarray]:
        """Load summary statistics for an analyzer.

        Summary files contain cross-epoch aggregate values computed inline
        during analysis (REQ_022). These are small values (scalars or small
        arrays per epoch) stored in a single file for efficient access.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Dict with 'epochs' array and one array per summary statistic.
            E.g., {"epochs": (N,), "mean_coarseness": (N,), "blob_count": (N,)}

        Raises:
            FileNotFoundError: If no summary exists for this analyzer
        """
        summary_path = os.path.join(self._dir(analyzer_name), "summary.npz")

        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"No summary for '{analyzer_name}'. Expected: {summary_path}")

        return dict(np.load(summary_path))

    def has_summary(self, analyzer_name: str) -> bool:
        """Check whether summary statistics exist for an analyzer.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            True if summary.npz exists for this analyzer
        """
        summary_path = os.path.join(self._dir(analyzer_name), "summary.npz")
        return os.path.exists(summary_path)

    def load_cross_epoch(
        self, analyzer_name: str, fields: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Load cross-epoch analysis results.

        Cross-epoch files contain results from analyzers that operate across
        all checkpoints (REQ_038), e.g. PCA trajectory projections.

        Args:
            analyzer_name: Name of the cross-epoch analyzer
            fields: Specific field names to load. None loads all fields.
                Requested fields are validated against the file's actual keys.

        Returns:
            Dict of numpy arrays from cross_epoch.npz

        Raises:
            FileNotFoundError: If no cross-epoch results exist
            ValueError: If a requested field is absent from the artifact
        """
        cross_epoch_path = os.path.join(self._dir(analyzer_name), "cross_epoch.npz")

        if not os.path.exists(cross_epoch_path):
            raise FileNotFoundError(
                f"No cross-epoch results for '{analyzer_name}'. Expected: {cross_epoch_path}"
            )

        if fields is None:
            return dict(np.load(cross_epoch_path))

        with np.load(cross_epoch_path) as npz:
            _validate_fields(analyzer_name, npz.files, fields, "cross_epoch.npz")
            return {name: npz[name] for name in fields}

    def has_cross_epoch(self, analyzer_name: str) -> bool:
        """Check whether cross-epoch results exist for an analyzer.

        Args:
            analyzer_name: Name of the cross-epoch analyzer

        Returns:
            True if cross_epoch.npz exists for this analyzer
        """
        cross_epoch_path = os.path.join(self._dir(analyzer_name), "cross_epoch.npz")
        return os.path.exists(cross_epoch_path)

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration from manifest.

        Returns:
            Dict with model config (prime, seed, etc.)
        """
        return self.manifest.get("model_config", {})

    def _load_manifest(self) -> dict[str, Any]:
        """Load manifest from disk."""
        manifest_path = os.path.join(self.artifacts_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            return {}

        with open(manifest_path) as f:
            return json.load(f)
