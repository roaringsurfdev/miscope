"""Base ModelFamily implementation loaded from JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from miscope.families.types import (
    AnalysisDatasetSpec,
    ArchitectureSpec,
    ParameterSpec,
    VariantState,
)
from miscope.families.variant import Variant

if TYPE_CHECKING:
    from miscope.families.intervention_variant import InterventionVariant


class BaseModelFamily:
    """ModelFamily implementation loaded from a family.json file.

    This is a data-only implementation that stores configuration.
    Actual model creation and dataset generation are delegated to
    family-specific implementations (see modulo_addition_1layer).

    For families that need custom logic, subclass this or implement
    the ModelFamily protocol directly.
    """

    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path | None = None,
        data_root: Path | str | None = None,
    ):
        """Initialize from config dict.

        Args:
            config: Parsed family.json content
            config_path: Path to the family.json file (for error messages)
            data_root: Unified data root containing per-family subdirectories.
                Defaults to ``Path("data")`` if not provided.
        """
        self._config = config
        self._config_path = config_path
        self._data_root = Path(data_root) if data_root is not None else Path("data")
        self._validate_config()

    @classmethod
    def from_json(
        cls,
        path: Path | str,
        data_root: Path | str | None = None,
    ) -> BaseModelFamily:
        """Load a BaseModelFamily from a family.json file.

        Args:
            path: Path to family.json
            data_root: Unified data root containing per-family subdirectories.

        Returns:
            BaseModelFamily instance
        """
        path = Path(path)
        with open(path) as f:
            config = json.load(f)
        return cls(config, config_path=path, data_root=data_root)

    def _validate_config(self) -> None:
        """Validate required fields are present."""
        required_fields = [
            "name",
            "display_name",
            "description",
            "architecture",
            "domain_parameters",
            "analyzers",
            "variant_pattern",
        ]
        missing = [f for f in required_fields if f not in self._config]
        if missing:
            location = f" in {self._config_path}" if self._config_path else ""
            raise KeyError(f"Missing required fields{location}: {missing}")

    @property
    def name(self) -> str:
        """Unique identifier, used as directory key."""
        return self._config["name"]

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        return self._config["display_name"]

    @property
    def description(self) -> str:
        """Brief description of the family."""
        return self._config["description"]

    @property
    def architecture(self) -> ArchitectureSpec:
        """Architectural properties."""
        return self._config["architecture"]

    @property
    def domain_parameters(self) -> dict[str, ParameterSpec]:
        """Parameters that vary across variants."""
        return self._config["domain_parameters"]

    @property
    def analyzers(self) -> list[str]:
        """Analyzer identifiers valid for this family."""
        return self._config["analyzers"]

    @property
    def secondary_analyzers(self) -> list[str]:
        """Secondary analyzer identifiers valid for this family."""
        return self._config.get("secondary_analyzers", [])

    @property
    def cross_epoch_analyzers(self) -> list[str]:
        """Cross-epoch analyzer identifiers valid for this family."""
        return self._config.get("cross_epoch_analyzers", [])

    @property
    def analysis_dataset(self) -> AnalysisDatasetSpec:
        """Specification for the analysis dataset."""
        return self._config.get("analysis_dataset", {})

    @property
    def variant_pattern(self) -> str:
        """Pattern for variant directory names."""
        return self._config["variant_pattern"]

    @property
    def data_root(self) -> Path:
        """Unified data root containing per-family subdirectories."""
        return self._data_root

    @property
    def family_dir(self) -> Path:
        """This family's directory: ``{data_root}/{name}/``."""
        return self._data_root / self.name

    @property
    def variants_dir(self) -> Path:
        """Directory containing this family's variants: ``{family_dir}/variants/``."""
        return self.family_dir / "variants"

    @property
    def variant_registry_path(self) -> Path:
        """Path to the compiled aggregate ``variant_registry.json``."""
        return self.family_dir / "variant_registry.json"

    @property
    def variant_registry(self) -> list[dict[str, Any]]:
        """Parsed variant_registry.json — list of per-variant summary entries.

        Reads on each access. Assign to a variable to avoid repeated disk reads.

        Raises:
            FileNotFoundError: If the registry has not been built yet.
        """
        if not self.variant_registry_path.exists():
            raise FileNotFoundError(
                f"No variant_registry.json for family {self.name!r} at "
                f"{self.variant_registry_path}. "
                "Build it via build_variant_registry(family) or the dashboard's "
                "analysis run."
            )
        with open(self.variant_registry_path) as f:
            return json.load(f)

    # --- Variant lookup ------------------------------------------------

    def get_variant(self, **params: Any) -> Variant:
        """Get a trained variant by domain parameter values.

        Args:
            **params: Domain parameters (e.g., prime=113, seed=999)

        Returns:
            Variant object with convenience access to checkpoints,
            artifacts, metadata, and forward passes.

        Raises:
            ValueError: If the variant doesn't exist or isn't trained.
        """
        variant = self.create_variant(params)
        if variant.state == VariantState.UNTRAINED:
            available = self.variant_parameters
            raise ValueError(
                f"Variant with params {params} not found or not trained. "
                f"Available variants: {available}"
            )
        return variant

    @property
    def variants(self) -> list[Variant]:
        """All discovered variants for this family.

        Scans ``{variants_dir}/`` for directories matching this family's
        ``variant_pattern``.
        """
        variants_dir = self.variants_dir
        if not variants_dir.exists():
            return []

        pattern_regex = _pattern_to_regex(self.variant_pattern, self.domain_parameters)
        variants: list[Variant] = []
        for variant_dir in variants_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            match = pattern_regex.match(variant_dir.name)
            if match:
                params = _extract_params(match, self.domain_parameters)
                variants.append(Variant(self, params))
        return variants

    @property
    def variant_parameters(self) -> list[dict[str, Any]]:
        """Parameter dicts for all discovered variants."""
        return [v.params for v in self.variants]

    def create_variant(self, params: dict[str, Any]) -> Variant:
        """Construct a Variant for this family without checking for files.

        Use this when you intend to create a new variant (e.g. for training),
        or when you want a Variant handle regardless of training state.
        """
        return Variant(self, params)

    def create_intervention_variant(
        self,
        parent_params: dict[str, Any],
        intervention_config: dict[str, Any],
    ) -> InterventionVariant:
        """Create an intervention variant nested under the parent variant.

        Args:
            parent_params: Domain parameters identifying the parent variant.
            intervention_config: Intervention parameter dict (passed to
                ``Variant.create_intervention_variant``).
        """
        parent = self.get_variant(**parent_params)
        return parent.create_intervention_variant(intervention_config)

    @property
    def ui_trainable(self) -> bool:
        """Whether this family can be trained through the generic training UI.

        Families that require programmatic variant construction (e.g., intervention
        families) should set this to false in their family.json.
        """
        return self._config.get("ui_trainable", True)

    def create_model(self, params: dict[str, Any], device: str | torch.device | None = None) -> Any:
        """Create a model instance.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Args:
            params: Domain parameter values

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"create_model() not implemented for {self.name}. Use a family-specific implementation."
        )

    def generate_analysis_dataset(
        self, params: dict[str, Any], device: str | torch.device | None = None
    ) -> torch.Tensor:
        """Generate the analysis dataset.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Args:
            params: Domain parameter values

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"generate_analysis_dataset() not implemented for {self.name}. "
            "Use a family-specific implementation."
        )

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.3,
        data_seed: int = 598,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test split for training.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Args:
            params: Domain parameter values
            training_fraction: Fraction of data to use for training
            data_seed: Random seed for train/test split
            device: Device to place tensors on

        Returns:
            Tuple of (train_data, train_labels, test_data, test_labels,
                     train_indices, test_indices)

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"generate_training_dataset() not implemented for {self.name}. "
            "Use a family-specific implementation."
        )

    def get_training_config(self) -> dict[str, Any]:
        """Return default training hyperparameters.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Returns:
            Dict with learning_rate, weight_decay, betas, etc.

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"get_training_config() not implemented for {self.name}. "
            "Use a family-specific implementation."
        )

    def prepare_analysis_context(
        self,
        params: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        """Prepare precomputed values needed for analysis.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Args:
            params: Domain parameter values
            device: Device for tensor computations

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"prepare_analysis_context() not implemented for {self.name}. "
            "Use a family-specific implementation."
        )

    def get_default_params(self) -> dict[str, Any]:
        """Get default parameter values from domain_parameters.

        Returns:
            Dict of parameter name to default value
        """
        return {
            name: spec.get("default")
            for name, spec in self.domain_parameters.items()
            if "default" in spec
        }

    def make_probe(
        self,
        params: dict[str, Any],
        inputs: list[list[int]],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Construct a probe tensor from raw input values.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Args:
            params: Domain parameter values
            inputs: List of input sequences
            device: Device to place the tensor on

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"make_probe() not implemented for {self.name}. Use a family-specific implementation."
        )

    def create_optimizer(
        self,
        model: Any,
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer using this family's training config.

        Args:
            model: Model instance to optimize.

        Returns:
            AdamW optimizer configured from get_training_config().
        """
        config = self.get_training_config()
        lr = config.get("learning_rate", 1e-3)
        wd = config.get("weight_decay", 1.0)
        betas = config.get("betas", (0.9, 0.98))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses must override this method.

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"compute_loss() not implemented for {self.name}. Use a family-specific implementation."
        )

    def build_config_dict(
        self,
        model: Any,
        params: dict[str, Any],
        data_seed: int,
        training_fraction: float,
    ) -> dict[str, Any]:
        """Build the config.json dict for a trained model.

        Base implementation includes domain params and training metadata.
        Family subclasses should override to add architecture-specific fields.

        Args:
            model: Trained model instance (architecture-specific)
            params: Domain parameter values
            data_seed: Data split seed used during training
            training_fraction: Fraction of data used for training

        Returns:
            Dict suitable for JSON serialization as config.json
        """
        return {
            **params,
            "data_seed": data_seed,
            "training_fraction": training_fraction,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def _pattern_to_regex(pattern: str, domain_parameters: dict[str, Any]) -> re.Pattern[str]:
    """Convert a variant pattern to a regex for matching directory names."""
    regex_pattern = re.escape(pattern)
    for param_name, spec in domain_parameters.items():
        placeholder = re.escape("{" + param_name + "}")
        param_type = spec.get("type", "str")
        if param_type == "int":
            capture_group = f"(?P<{param_name}>\\d+)"
        elif param_type == "float":
            capture_group = f"(?P<{param_name}>\\d+\\.?\\d*)"
        else:
            capture_group = f"(?P<{param_name}>[^_]+)"
        regex_pattern = regex_pattern.replace(placeholder, capture_group)
    return re.compile(f"^{regex_pattern}$")


def _extract_params(match: re.Match[str], domain_parameters: dict[str, Any]) -> dict[str, Any]:
    """Extract typed parameters from a regex match."""
    params: dict[str, Any] = {}
    for param_name, spec in domain_parameters.items():
        raw_value = match.group(param_name)
        param_type = spec.get("type", "str")
        if param_type == "int":
            params[param_name] = int(raw_value)
        elif param_type == "float":
            params[param_name] = float(raw_value)
        else:
            params[param_name] = raw_value
    return params
