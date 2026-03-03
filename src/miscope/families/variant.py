"""Variant class representing a specific trained model within a family."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import tqdm.auto as tqdm
from safetensors.torch import load_file, save_file

from miscope.families.types import VariantState

if TYPE_CHECKING:
    from transformer_lens import ActivationCache, HookedTransformer

    from miscope.analysis.artifact_loader import ArtifactLoader
    from miscope.families.protocols import ModelFamily
    from miscope.views.catalog import BoundView, EpochContext


@dataclass
class TrainingResult:
    """Result of a training run."""

    train_losses: list[float]
    test_losses: list[float]
    checkpoint_epochs: list[int]
    final_train_loss: float
    final_test_loss: float
    variant_dir: Path


class Variant:
    """A specific trained model within a family.

    Variants share the same architecture and analysis logic but differ
    in domain-specific parameters (e.g., modulus, seed).

    State is derived from filesystem presence:
    - UNTRAINED: No checkpoints directory or empty
    - TRAINED: Checkpoints exist but no artifacts
    - ANALYZED: Both checkpoints and artifacts exist
    """

    def __init__(
        self,
        family: ModelFamily,
        params: dict[str, Any],
        results_dir: Path,
    ):
        """Initialize a Variant.

        Args:
            family: The ModelFamily this variant belongs to
            params: Domain parameter values (e.g., {"prime": 113, "seed": 42})
            results_dir: Root results directory (typically "results/")
        """
        self._family = family
        self._params = params
        self._results_dir = Path(results_dir)

    @property
    def family(self) -> ModelFamily:
        """The ModelFamily this variant belongs to."""
        return self._family

    @property
    def params(self) -> dict[str, Any]:
        """Domain parameter values for this variant."""
        return self._params.copy()

    @property
    def name(self) -> str:
        """Variant directory name derived from family pattern."""
        return self._family.get_variant_directory_name(self._params)

    @property
    def variant_dir(self) -> Path:
        """Path to this variant's directory.

        Structure: results/{family.name}/{variant_name}/
        """
        return self._results_dir / self._family.name / self.name

    @property
    def checkpoints_dir(self) -> Path:
        """Path to checkpoints subdirectory."""
        return self.variant_dir / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        """Path to analysis artifacts subdirectory."""
        return self.variant_dir / "artifacts"

    @property
    def metadata_path(self) -> Path:
        """Path to training metadata JSON file."""
        return self.variant_dir / "metadata.json"

    @property
    def config_path(self) -> Path:
        """Path to model config JSON file."""
        return self.variant_dir / "config.json"

    @property
    def state(self) -> VariantState:
        """Current state based on filesystem presence."""
        has_checkpoints = self._has_checkpoints()
        has_artifacts = self._has_artifacts()

        if has_checkpoints and has_artifacts:
            return VariantState.ANALYZED
        elif has_checkpoints:
            return VariantState.TRAINED
        else:
            return VariantState.UNTRAINED

    def _has_checkpoints(self) -> bool:
        """Check if any checkpoint files exist."""
        if not self.checkpoints_dir.exists():
            return False
        # Look for .safetensors files
        return any(self.checkpoints_dir.glob("checkpoint_epoch_*.safetensors"))

    def _has_artifacts(self) -> bool:
        """Check if any artifact files exist."""
        if not self.artifacts_dir.exists():
            return False
        # Look for any .npz files (analysis artifacts)
        return any(self.artifacts_dir.glob("**/*.npz"))

    def get_available_checkpoints(self) -> list[int]:
        """List checkpoint epochs available on disk.

        Returns:
            Sorted list of epoch numbers with available checkpoints
        """
        epochs = []
        if self.checkpoints_dir.exists():
            for path in self.checkpoints_dir.glob("checkpoint_epoch_*.safetensors"):
                # Extract epoch from filename: checkpoint_epoch_00100.safetensors
                epoch_str = path.stem.replace("checkpoint_epoch_", "")
                try:
                    epochs.append(int(epoch_str))
                except ValueError:
                    continue
        return sorted(epochs)

    def load_checkpoint(self, epoch: int) -> dict[str, Any]:
        """Load model state_dict from a specific checkpoint.

        Args:
            epoch: The epoch number of the checkpoint to load

        Returns:
            Model state_dict

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:05d}.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found for epoch {epoch} at {checkpoint_path}")
        return load_file(checkpoint_path)

    def load_model_at_checkpoint(self, epoch: int) -> HookedTransformer:
        """Load a HookedTransformer with weights from a specific checkpoint.

        Args:
            epoch: The epoch number of the checkpoint to load

        Returns:
            HookedTransformer with checkpoint weights loaded
        """
        model = self._family.create_model(self._params)
        state_dict = self.load_checkpoint(epoch)
        model.load_state_dict(state_dict)
        return model

    # --- Notebook convenience properties (REQ_037) ---

    @property
    def artifacts(self) -> ArtifactLoader:
        """ArtifactLoader for this variant's analysis artifacts.

        Returns a new loader each call (no caching). Assign to a variable
        if loading multiple artifacts in sequence.
        """
        from miscope.analysis.artifact_loader import ArtifactLoader

        return ArtifactLoader(str(self.artifacts_dir))

    @property
    def metadata(self) -> dict[str, Any]:
        """Training metadata (losses, checkpoint epochs, indices).

        Reads from metadata.json on each access. Assign to a variable
        to avoid repeated disk reads.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"No metadata found at {self.metadata_path}")
        with open(self.metadata_path) as f:
            return json.load(f)

    @property
    def model_config(self) -> dict[str, Any]:
        """Model configuration (architecture, domain params, training params).

        Reads from config.json on each access.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"No config found at {self.config_path}")
        with open(self.config_path) as f:
            return json.load(f)

    @property
    def train_losses(self) -> list[float]:
        """Per-epoch training losses. Shortcut for metadata['train_losses']."""
        return self.metadata["train_losses"]

    @property
    def test_losses(self) -> list[float]:
        """Per-epoch test losses. Shortcut for metadata['test_losses']."""
        return self.metadata["test_losses"]

    def run_with_cache(
        self,
        probe: torch.Tensor,
        epoch: int,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, ActivationCache]:
        """Load model at checkpoint and run a forward pass with activation cache.

        Args:
            probe: Input tensor for the forward pass
            epoch: Checkpoint epoch to load
            device: Device for the model (default: auto-detect)

        Returns:
            Tuple of (logits, cache) from transformer_lens run_with_cache
        """
        model = self.load_model_at_checkpoint(epoch)
        if device is not None:
            model.to(device)  # in-place device move
        logits, cache = model.run_with_cache(probe)
        return logits, cache  # type: ignore[return-value]

    def make_probe(
        self,
        inputs: list[list[int]],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Construct a probe tensor from input values.

        Delegates to the family's make_probe implementation for
        family-specific input formatting.

        Args:
            inputs: List of input sequences (e.g., [[3, 29]] for modular addition)
            device: Device for the tensor

        Returns:
            Properly formatted probe tensor
        """
        return self._family.make_probe(self._params, inputs, device=device)

    def analysis_dataset(
        self,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate the full analysis dataset (probe) for this variant.

        For modular addition, this is the full p x p grid of all input pairs.

        Args:
            device: Device for the tensor

        Returns:
            Analysis probe tensor
        """
        return self._family.generate_analysis_dataset(self._params, device=device)

    def analysis_context(
        self,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Prepare family-specific analysis context.

        Returns precomputed values needed for custom analysis (e.g.,
        fourier_basis for modular addition).

        Args:
            device: Device for tensor computations (default: auto-detect)

        Returns:
            Dict with family-specific context (params, fourier_basis, etc.)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._family.prepare_analysis_context(self._params, device=device)

    def at(self, epoch: int | None) -> EpochContext:
        """Pin a training moment and return an EpochContext for view access.

        The returned context fixes the epoch for all views dispatched through
        it, enforcing a shared cursor across lenses.

        Args:
            epoch: Training epoch to pin. None defers resolution per-view
                (per-epoch views resolve to first available; cross-epoch
                views receive None as a "no cursor" signal).

        Returns:
            EpochContext bound to this variant and epoch.
        """
        from miscope.views.catalog import EpochContext

        return EpochContext(variant=self, epoch=epoch)

    def view(self, name: str) -> BoundView:
        """Convenience shortcut for variant.at(epoch=None).view(name).

        For per-epoch views, resolves to the first available artifact epoch.
        For cross-epoch and metadata-based views, epoch is None (no cursor).

        Args:
            name: View identifier (e.g., "loss_curve", "dominant_frequencies").

        Returns:
            BoundView ready to call .show(), .figure(), or .export().

        Raises:
            KeyError: If view name is not found in the catalog.
        """
        return self.at(epoch=None).view(name)

    # --- End notebook convenience properties ---

    def ensure_directories(self) -> None:
        """Create variant directories if they don't exist."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        num_epochs: int | None = None,
        checkpoint_epochs: list[int] | None = None,
        training_fraction: float = 0.3,
        data_seed: int = 598,
        device: str | torch.device | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TrainingResult:
        """Train this variant's model.

        Creates the model via family.create_model(), generates training data,
        runs the training loop, and saves checkpoints and metadata.

        Args:
            num_epochs: Total training epochs (default: from family config)
            checkpoint_epochs: Epochs at which to save checkpoints
                              (default: from family config)
            training_fraction: Fraction of data for training (default: 0.3)
            data_seed: Random seed for train/test split (default: 598)
            device: Device for training (default: auto-detect CUDA)
            progress_callback: Optional callback for progress updates
                              (fraction: float, description: str)

        Returns:
            TrainingResult with losses and checkpoint info
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get training config from family
        training_config = self._family.get_training_config()

        if num_epochs is None:
            num_epochs = int(training_config.get("num_epochs", 25000))

        if checkpoint_epochs is None:
            checkpoint_epochs = list(training_config.get("default_checkpoint_epochs", []))

        # Filter and sort checkpoint epochs
        checkpoint_epochs = sorted([e for e in checkpoint_epochs if e < num_epochs])
        checkpoint_epochs_set = set(checkpoint_epochs)

        # Ensure directories exist
        self.ensure_directories()

        # Create model via family
        model = self._family.create_model(self._params, device=device)

        # Generate training data via family
        train_data, train_labels, test_data, test_labels, train_indices, test_indices = (
            self._family.generate_training_dataset(
                self._params,
                training_fraction=training_fraction,
                data_seed=data_seed,
                device=device,
            )
        )

        # Setup optimizer
        lr = training_config.get("learning_rate", 1e-3)
        wd = training_config.get("weight_decay", 1.0)
        betas = training_config.get("betas", (0.9, 0.98))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

        train_losses: list[float] = []
        test_losses: list[float] = []
        saved_checkpoint_epochs: list[int] = []

        # Training loop
        for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
            # Forward pass
            train_logits = model(train_data)
            train_loss = self._loss_function(train_logits, train_labels)

            # Backward pass
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(train_loss.item())

            # Evaluate on test set
            with torch.inference_mode():
                test_logits = model(test_data)
                test_loss = self._loss_function(test_logits, test_labels)
                test_losses.append(test_loss.item())

            # Save checkpoint if scheduled
            if epoch in checkpoint_epochs_set:
                self._save_checkpoint(model.state_dict(), epoch)
                saved_checkpoint_epochs.append(epoch)

            # Progress callback
            if progress_callback and epoch % 100 == 0:
                progress_callback(
                    epoch / num_epochs,
                    f"Epoch {epoch}/{num_epochs} - Train: {train_loss.item():.4f}, Test: {test_loss.item():.4f}",
                )

        # Save final model as latest checkpoint
        final_epoch = num_epochs - 1
        if final_epoch not in saved_checkpoint_epochs:
            self._save_checkpoint(model.state_dict(), final_epoch)
            saved_checkpoint_epochs.append(final_epoch)

        # Save config
        self._save_config(model.cfg, data_seed, training_fraction)

        # Save metadata
        self._save_metadata(
            train_losses=train_losses,
            test_losses=test_losses,
            train_indices=train_indices.tolist(),
            test_indices=test_indices.tolist(),
            checkpoint_epochs=saved_checkpoint_epochs,
            num_epochs=num_epochs,
        )

        if progress_callback:
            progress_callback(1.0, "Training complete!")

        return TrainingResult(
            train_losses=train_losses,
            test_losses=test_losses,
            checkpoint_epochs=saved_checkpoint_epochs,
            final_train_loss=train_losses[-1],
            final_test_loss=test_losses[-1],
            variant_dir=self.variant_dir,
        )

    def _loss_function(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for sequence prediction.

        Args:
            logits: Model output logits, shape (batch, seq_len, vocab) or (batch, vocab)
            labels: Target labels, shape (batch,)

        Returns:
            Mean negative log probability of correct labels
        """
        if len(logits.shape) == 3:
            logits = logits[:, -1]  # Take last position
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()

    def _save_checkpoint(self, state_dict: dict[str, Any], epoch: int) -> None:
        """Save a checkpoint to disk as safetensors.

        Args:
            state_dict: Model state dict to save
            epoch: Epoch number for filename
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:05d}.safetensors"
        save_file(state_dict, str(checkpoint_path))

    def _save_config(
        self,
        cfg: Any,
        data_seed: int,
        training_fraction: float,
    ) -> None:
        """Save model configuration as JSON.

        Args:
            cfg: HookedTransformerConfig object
            data_seed: Data split random seed
            training_fraction: Training data fraction
        """
        config_dict = {
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "d_head": cfg.d_head,
            "d_mlp": cfg.d_mlp,
            "act_fn": cfg.act_fn,
            "normalization_type": cfg.normalization_type,
            "d_vocab": cfg.d_vocab,
            "d_vocab_out": cfg.d_vocab_out,
            "n_ctx": cfg.n_ctx,
            "seed": cfg.seed,
            # Domain parameters
            **self._params,
            # Training parameters
            "model_seed": self._params.get("seed", cfg.seed),
            "data_seed": data_seed,
            "training_fraction": training_fraction,
        }
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _save_metadata(self, **kwargs: Any) -> None:
        """Save training metadata as JSON.

        Args:
            **kwargs: Metadata to save (losses, indices, etc.)
        """
        with open(self.metadata_path, "w") as f:
            json.dump(kwargs, f)

    def __repr__(self) -> str:
        return (
            f"Variant(family={self._family.name!r}, name={self.name!r}, state={self.state.value})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variant):
            return NotImplemented
        return self._family.name == other._family.name and self._params == other._params

    def __hash__(self) -> int:
        return hash((self._family.name, tuple(sorted(self._params.items()))))
