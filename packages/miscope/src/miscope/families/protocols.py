"""Protocol definitions for model families."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from miscope.core.basis_projection import BasisProjectionSite
from miscope.families.types import AnalysisDatasetSpec, ParameterSpec

if TYPE_CHECKING:
    from miscope.architectures import HookedModel  # noqa: F401
    from miscope.families.intervention_variant import InterventionVariant
    from miscope.families.variant import Variant


@runtime_checkable
class ModelFamily(Protocol):
    """Protocol defining the contract for a model family.

    A ModelFamily groups structurally similar models that share:
    - Architecture (layer count, head count, activation functions)
    - Analyzers (which analysis functions are valid)
    - Visualizations (which visualizations can be rendered)
    - Probe schema (what kind of probe input is valid)

    The `name` property serves as the directory key under the unified data
    root (`data/{name}/`).
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
    def domain_parameters(self) -> dict[str, ParameterSpec]:
        """Parameters that vary across variants."""
        ...

    @property
    def analyzers(self) -> list[str]:
        """Analyzer identifiers valid for this family.

        A single flat list. The pipeline derives execution order from each
        analyzer's declared ``inputs`` (REQ_132); the family does not group
        analyzers by execution phase.
        """
        ...

    @property
    def weight_basis_projection_sites(self) -> tuple[BasisProjectionSite, ...]:
        """Weight-side sites for ``weight_basis_projection`` (REQ_126).

        Each site bundles a composer ``(parameter_snapshot_dict, context)
        -> ndarray`` and a period-axis spec. The projection itself is
        performed by the analyzer via REQ_109 primitives — never inline
        in the family. Families without a registered basis return an
        empty tuple.

        See :class:`miscope.core.basis_projection.BasisProjectionSite`.
        """
        ...

    @property
    def activation_basis_projection_sites(self) -> tuple[BasisProjectionSite, ...]:
        """Activation-side sites for ``activation_basis_projection`` (REQ_126).

        Parallel to ``weight_basis_projection_sites`` but the composer
        takes ``(activation_cache, context) -> ndarray`` and each site
        declares the canonical hooks its composer reads. The analyzer
        aggregates ``required_hooks`` across sites for its Spec.
        """
        ...

    @property
    def analysis_dataset(self) -> AnalysisDatasetSpec:
        """Specification for the analysis dataset."""
        ...

    @property
    def variant_pattern(self) -> str:
        """Pattern for variant directory names.

        Example: ``"p{prime}_seed{seed}_dseed{data_seed}"``
        """
        ...

    @property
    def data_root(self) -> Path:
        """Unified data root containing per-family subdirectories."""
        ...

    @property
    def family_dir(self) -> Path:
        """This family's directory: ``{data_root}/{name}/``."""
        ...

    @property
    def variants_dir(self) -> Path:
        """Directory containing this family's variants: ``{family_dir}/variants/``."""
        ...

    @property
    def variant_registry(self) -> list[dict[str, Any]]:
        """Parsed ``variant_registry.json`` — list of per-variant summary entries.

        Raises ``FileNotFoundError`` if the registry has not been built yet.
        """
        ...

    def get_variant(self, **params: Any) -> Variant:
        """Look up a trained variant by domain parameter values."""
        ...

    @property
    def variants(self) -> list[Variant]:
        """All discovered variants for this family."""
        ...

    @property
    def variant_parameters(self) -> list[dict[str, Any]]:
        """Parameter dicts for all discovered variants."""
        ...

    def create_variant(self, params: dict[str, Any]) -> Variant:
        """Construct a Variant for this family without checking for files."""
        ...

    def create_intervention_variant(
        self,
        parent_params: dict[str, Any],
        intervention_config: dict[str, Any],
    ) -> InterventionVariant:
        """Create an intervention variant nested under the parent variant."""
        ...

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> HookedModel:
        """Instantiate a model with the given domain parameters.

        Args:
            params: Domain parameter values (e.g., {"prime": 113, "seed": 42})
            device: Device to place the model on

        Returns:
            A ``HookedModel`` configured for this family.
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

    def create_optimizer(
        self,
        model: Any,
    ) -> torch.optim.Optimizer:
        """Create an optimizer for model training.

        Args:
            model: Model instance created by create_model()

        Returns:
            Configured optimizer for this family's training regime
        """
        ...

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss from model output logits and target labels.

        Args:
            logits: Model output. Shape is architecture-dependent:
                    (batch, seq_len, vocab_size) for transformers,
                    (batch, vocab_size) for plain MLPs.
            labels: Target class indices of shape (batch,).

        Returns:
            Scalar loss tensor.
        """
        ...

    def build_config_dict(
        self,
        model: Any,
        params: dict[str, Any],
        data_seed: int,
        training_fraction: float,
    ) -> dict[str, Any]:
        """
        TODO: Add spec comments for this method
        """
        ...
