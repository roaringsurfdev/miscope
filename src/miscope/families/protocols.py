"""Protocol definitions for model families."""

from typing import Any, Protocol, runtime_checkable

import torch
from transformer_lens import HookedTransformer

from miscope.families.types import AnalysisDatasetSpec, ArchitectureSpec, ParameterSpec


@runtime_checkable
class ModelFamily(Protocol):
    """Protocol defining the contract for a model family.

    A ModelFamily groups structurally similar models that share:
    - Architecture (layer count, head count, activation functions)
    - Analyzers (which analysis functions are valid)
    - Visualizations (which visualizations can be rendered)
    - Probe schema (what kind of probe input is valid)

    The `name` property serves as the directory key for both
    `model_families/{name}/` and `results/{name}/`.
    """

    @property
    def name(self) -> str:
        """Unique identifier, used as directory key."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        ...

    @property
    def description(self) -> str:
        """Brief description of the family."""
        ...

    @property
    def architecture(self) -> ArchitectureSpec:
        """Architectural properties (n_layers, n_heads, etc.)."""
        ...

    @property
    def domain_parameters(self) -> dict[str, ParameterSpec]:
        """Parameters that vary across variants."""
        ...

    @property
    def analyzers(self) -> list[str]:
        """Analyzer identifiers valid for this family."""
        ...

    @property
    def secondary_analyzers(self) -> list[str]:
        """Secondary analyzer identifiers valid for this family.

        These analyzers run after all per-epoch primary analysis completes
        (Phase 1.5) and derive new per-epoch artifacts from existing ones.
        No model loading occurs during secondary analysis.
        """
        ...

    @property
    def cross_epoch_analyzers(self) -> list[str]:
        """Cross-epoch analyzer identifiers valid for this family.

        These analyzers run after all per-epoch analysis completes
        and consume per-epoch artifacts to produce cross-epoch results.
        """
        ...

    @property
    def visualizations(self) -> list[str]:
        """Visualization identifiers valid for this family."""
        ...

    @property
    def analysis_dataset(self) -> AnalysisDatasetSpec:
        """Specification for the analysis dataset."""
        ...

    @property
    def variant_pattern(self) -> str:
        """Pattern for variant directory names.

        Example: "modulo_addition_1layer_p{prime}_seed{seed}"
        """
        ...

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> HookedTransformer:
        """Instantiate a model with the given domain parameters.

        Args:
            params: Domain parameter values (e.g., {"prime": 113, "seed": 42})
            device: Device to place the model on

        Returns:
            A HookedTransformer configured for this family
        """
        ...

    def generate_analysis_dataset(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate the analysis dataset (probe) for a variant.

        Args:
            params: Domain parameter values
            device: Device to place the dataset on

        Returns:
            Tensor of inputs for analysis forward passes
        """
        ...

    def get_variant_directory_name(self, params: dict[str, Any]) -> str:
        """Generate variant directory name from parameters.

        Args:
            params: Domain parameter values

        Returns:
            Directory name (e.g., "modulo_addition_1layer_p113_seed42")
        """
        ...

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.3,
        data_seed: int = 598,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test split for training.

        Args:
            params: Domain parameter values (e.g., {"prime": 113, "seed": 42})
            training_fraction: Fraction of data to use for training
            data_seed: Random seed for train/test split
            device: Device to place tensors on

        Returns:
            Tuple of (train_data, train_labels, test_data, test_labels,
                     train_indices, test_indices)
        """
        ...

    def get_training_config(self) -> dict[str, Any]:
        """Return default training hyperparameters.

        Returns:
            Dict with learning_rate, weight_decay, betas, etc.
        """
        ...

    def get_default_params(self) -> dict[str, Any]:
        """Get default parameter values from domain_parameters.

        Returns:
            Dict of parameter name to default value
        """
        ...

    def make_probe(
        self,
        params: dict[str, Any],
        inputs: list[list[int]],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Construct a probe tensor from raw input values.

        Formats inputs according to the family's expected probe structure.
        For example, modular addition takes [a, b] pairs and appends
        the equals token to produce [a, b, p].

        Args:
            params: Domain parameter values (e.g., {"prime": 113})
            inputs: List of input sequences (e.g., [[3, 29], [5, 7]])
            device: Device to place the tensor on

        Returns:
            Probe tensor ready for model.run_with_cache()
        """
        ...

    def prepare_analysis_context(
        self,
        params: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        """Prepare precomputed values needed for analysis.

        This method allows families to provide domain-specific precomputed
        values that analyzers need, without the pipeline having to know
        what those values are.

        The returned context dict should include:
        - 'params': The variant's domain parameters
        - Any family-specific precomputed values (e.g., 'fourier_basis' for
          Modulo Addition families)

        Args:
            params: Domain parameter values (e.g., {"prime": 113, "seed": 42})
            device: Device for tensor computations

        Returns:
            Dict containing 'params' and any precomputed analysis context
        """
        ...
