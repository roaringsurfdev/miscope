"""Modulo Addition 1-Layer family implementation.

This module provides the concrete implementation for the modulo addition
single-layer transformer family, based on Neel Nanda's grokking experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import einops
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from miscope.analysis.library import get_fourier_basis
from miscope.families.base_model_family import BaseModelFamily


class ModuloAddition1LayerFamily(BaseModelFamily):
    """Implementation of ModelFamily for 1-layer modular addition transformer.

    This family represents single-layer transformers trained on the modular
    addition task: given inputs (a, b), predict (a + b) mod p.

    The model architecture matches Neel Nanda's grokking experiment:
    - 1 layer, 4 heads, d_model=128, d_mlp=512
    - ReLU activation, no layer norm
    - Vocabulary size = p + 1 (0 to p-1 for numbers, p for equals token)

    Domain parameters:
    - prime: The modulus p for the addition task
    - seed: Random seed for model initialization
    """

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> HookedTransformer:
        """Create a HookedTransformer for modular addition.

        Args:
            params: Domain parameters containing 'prime' and optionally 'seed'
            device: Device to place the model on (default: None, uses default device)

        Returns:
            HookedTransformer configured for modular addition
        """
        p = params["prime"]
        seed = params.get("seed", self.get_default_params().get("seed", 999))

        arch = self.architecture

        cfg = HookedTransformerConfig(
            n_layers=arch.get("n_layers", 1),
            n_heads=arch.get("n_heads", 4),
            d_model=arch.get("d_model", 128),
            d_head=arch.get("d_head", 32),
            d_mlp=arch.get("d_mlp", 512),
            act_fn=arch.get("act_fn", "relu"),
            normalization_type=arch.get("normalization_type"),
            d_vocab=p + 1,  # 0 to p-1 for numbers, p for equals token
            d_vocab_out=p,  # Output is 0 to p-1
            n_ctx=arch.get("n_ctx", 3),  # a, b, =
            init_weights=True,
            device=str(device) if device is not None else None,
            seed=seed,
        )

        model = HookedTransformer(cfg)

        # Disable biases (matches original experiment)
        for name, param in model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        return model

    def generate_analysis_dataset(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate full (a, b) input grid for analysis.

        Creates all p^2 input combinations for modular addition analysis.

        Args:
            params: Domain parameters containing 'prime'
            device: Device to place the dataset on

        Returns:
            Tensor of shape (p^2, 3) containing [a, b, equals_token] rows
        """
        p = params["prime"]

        # Create all (a, b) pairs
        a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        equals_vector = einops.repeat(torch.tensor(p), " -> (i j)", i=p, j=p)

        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)

        if device:
            dataset = dataset.to(device)

        return dataset

    def get_labels(self, params: dict[str, Any]) -> torch.Tensor:
        """Get ground truth labels for the analysis dataset.

        Args:
            params: Domain parameters containing 'prime'

        Returns:
            Tensor of shape (p^2,) containing (a + b) mod p
        """
        p = params["prime"]
        dataset = self.generate_analysis_dataset(params)
        return (dataset[:, 0] + dataset[:, 1]) % p

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.3,
        data_seed: int = 598,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test split for training.

        Creates all p^2 input combinations and splits them into train/test sets.

        Args:
            params: Domain parameters containing 'prime'
            training_fraction: Fraction of data to use for training (default: 0.3)
            data_seed: Random seed for reproducible train/test split (default: 598)
            device: Device to place tensors on

        Returns:
            Tuple of (train_data, train_labels, test_data, test_labels,
                     train_indices, test_indices)
        """
        p = params["prime"]

        # Generate full dataset (same as analysis dataset)
        dataset = self.generate_analysis_dataset(params, device=device)
        labels = (dataset[:, 0] + dataset[:, 1]) % p

        # Create reproducible train/test split
        torch.manual_seed(data_seed)
        indices = torch.randperm(p * p)
        cutoff = int(p * p * training_fraction)
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]

        train_data = dataset[train_indices]
        train_labels = labels[train_indices]
        test_data = dataset[test_indices]
        test_labels = labels[test_indices]

        return train_data, train_labels, test_data, test_labels, train_indices, test_indices

    def get_training_config(self) -> dict[str, Any]:
        """Return default training hyperparameters.

        These match the original Neel Nanda grokking experiment settings.

        Returns:
            Dict with learning_rate, weight_decay, betas, num_epochs,
            and checkpoint configuration
        """
        return {
            "learning_rate": 1e-3,
            "weight_decay": 1.0,
            "betas": (0.9, 0.98),
            "num_epochs": 25000,
            "default_checkpoint_epochs": sorted(
                list(
                    set(
                        [
                            *range(0, 1500, 100),  # Early training - sparse
                            *range(1500, 9000, 500),  # Mid training - moderate
                            *range(9000, 13000, 100),  # Grokking region - dense
                            *range(13000, 25000, 500),  # Post-grokking - moderate
                        ]
                    )
                )
            ),
        }

    def prepare_analysis_context(
        self,
        params: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        """Prepare precomputed values for modular addition analysis.

        For modular addition, this includes:
        - params: The variant's domain parameters
        - fourier_basis: Precomputed Fourier basis for the given prime

        Args:
            params: Domain parameters containing 'prime'
            device: Device for tensor computations

        Returns:
            Dict with 'params', 'fourier_basis', and 'loss_fn'
        """
        p = params["prime"]
        fourier_basis, _ = get_fourier_basis(p, device)

        def loss_fn(model, probe):
            """Cross-entropy loss on modular addition probe dataset.

            Derives labels from probe inputs: (a + b) mod p.
            Uses last-position logits matching training loss.
            """
            labels = (probe[:, 0] + probe[:, 1]) % p
            with torch.no_grad():
                logits = model(probe)
            log_probs = logits[:, -1].log_softmax(dim=-1)
            loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean()
            return loss.item()

        a_vals = torch.arange(p).repeat_interleave(p)
        b_vals = torch.arange(p).repeat(p)
        labels = ((a_vals + b_vals) % p).numpy()

        return {
            "params": params,
            "fourier_basis": fourier_basis,
            "loss_fn": loss_fn,
            "labels": labels,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss on last-position transformer logits.

        Args:
            logits: Shape (batch, seq_len, vocab_size) from HookedTransformer.
            labels: Target class indices of shape (batch,).

        Returns:
            Scalar mean negative log-probability of correct labels.
        """
        last_logits = logits[:, -1].to(torch.float64)
        log_probs = last_logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()

    def build_config_dict(
        self,
        model: Any,
        params: dict[str, Any],
        data_seed: int,
        training_fraction: float,
    ) -> dict[str, Any]:
        """Build config.json dict from HookedTransformerConfig."""
        cfg = model.cfg
        return {
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
            **params,
            "model_seed": params.get("seed", cfg.seed),
            "data_seed": data_seed,
            "training_fraction": training_fraction,
        }

    def run_forward_pass(
        self,
        model: Any,
        probe: torch.Tensor,
    ) -> Any:
        """Run a forward pass and return a TransformerLensBundle.

        Args:
            model: HookedTransformer instance created by create_model()
            probe: Analysis dataset tensor from generate_analysis_dataset()

        Returns:
            TransformerLensBundle wrapping the model, cache, and logits
        """
        from miscope.analysis.bundle import TransformerLensBundle

        with torch.inference_mode():
            logits, cache = model.run_with_cache(probe)
        return TransformerLensBundle(model, cache, logits)

    def make_probe(
        self,
        params: dict[str, Any],
        inputs: list[list[int]],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Construct a probe tensor from (a, b) input pairs.

        Appends the equals token (value = prime) to each pair,
        matching the model's expected input format [a, b, =].

        Args:
            params: Domain parameters containing 'prime'
            inputs: List of [a, b] pairs (e.g., [[3, 29], [5, 7]])
            device: Device to place the tensor on

        Returns:
            Tensor of shape (n, 3) with [a, b, equals_token] rows
        """
        p = params["prime"]
        rows = [[a, b, p] for a, b in inputs]
        tensor = torch.tensor(rows, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


def load_modulo_addition_1layer_family(
    model_families_dir: Path | str = "model_families",
) -> ModuloAddition1LayerFamily:
    """Load the modulo addition 1-layer family from the standard location.

    Args:
        model_families_dir: Path to model_families directory

    Returns:
        ModuloAddition1LayerFamily instance
    """
    family_json = Path(model_families_dir) / "modulo_addition_1layer" / "family.json"
    family = ModuloAddition1LayerFamily.from_json(family_json)
    assert isinstance(family, ModuloAddition1LayerFamily)
    return family
