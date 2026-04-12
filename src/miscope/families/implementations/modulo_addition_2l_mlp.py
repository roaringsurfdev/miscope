"""Two-layer MLP family for modular addition.

Implements the 2L MLP architecture: same modular addition task as the
1-layer transformer, but stripped to two fully-connected layers with no
attention, no positional embedding, and no residual stream.

Input encoding: one-hot concatenation of (a, b) — two vectors of size p,
concatenated to 2p. No equals token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from miscope.analysis.library import get_fourier_basis
from miscope.families.base_model_family import BaseModelFamily


class ModuloAddition2LMLP(nn.Module):
    """Two-layer MLP for modular addition.

    Architecture: Linear(2p, d_hidden) → ReLU → Linear(d_hidden, p) → logits

    Input is a flat one-hot encoding of (a, b) concatenated: size 2p.
    Output is logits over the p residue classes (0 to p-1).
    """

    def __init__(
        self,
        vocab_size: int,
        d_hidden: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.vocab_size = vocab_size
        self.d_hidden = d_hidden

        self.W_in = nn.Linear(2 * vocab_size, d_hidden, bias=False)
        self.relu = nn.ReLU()
        self.W_out = nn.Linear(d_hidden, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded input of shape (batch, 2 * vocab_size)

        Returns:
            Logits of shape (batch, vocab_size)
        """
        hidden = self.relu(self.W_in(x))
        return self.W_out(hidden)


class ModuloAddition2LMLPActivationBundle:
    """ActivationBundle implementation for ModuloAddition2LMLP.

    Captures hidden activations via a forward hook registered during
    run_forward_pass(). Exposes them via the ActivationBundle protocol.
    """

    _WEIGHT_LOOKUP = {
        "W_in": lambda m: m.W_in.weight,
        "W_out": lambda m: m.W_out.weight,
    }

    def __init__(
        self,
        model: ModuloAddition2LMLP,
        hidden_acts: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self._model = model
        self._hidden_acts = hidden_acts
        self._logits = logits

    def mlp_post(self, layer: int, position: int) -> torch.Tensor:
        """Return post-ReLU hidden activations.

        Args:
            layer: Must be 0 (the only hidden layer).
            position: Ignored — MLP has no sequence dimension.

        Returns:
            Tensor of shape (batch, d_hidden).

        Raises:
            ValueError: If layer != 0.
        """
        if layer != 0:
            raise ValueError(
                f"ModuloAddition2LMLPActivationBundle has one hidden layer (layer 0), got layer={layer}"
            )
        return self._hidden_acts

    def weight(self, name: str) -> torch.Tensor:
        """Return a named weight matrix.

        Supported: W_in (first layer), W_out (second layer).
        Raises KeyError for transformer-specific weight names.

        Args:
            name: Weight matrix name.

        Returns:
            Weight tensor.

        Raises:
            KeyError: If name is not available for this architecture.
        """
        if name not in self._WEIGHT_LOOKUP:
            raise KeyError(f"Weight '{name}' not available in ModuloAddition2LMLPActivationBundle")
        return self._WEIGHT_LOOKUP[name](self._model)

    def attention_pattern(self, layer: int) -> torch.Tensor:
        raise NotImplementedError("ModuloAddition2LMLPActivationBundle has no attention patterns")

    def residual_stream(self, layer: int, position: int, location: str) -> torch.Tensor:
        raise NotImplementedError("ModuloAddition2LMLPActivationBundle has no residual stream")

    def logits(self, position: int) -> torch.Tensor:
        """Return output logits.

        Args:
            position: Ignored — MLP has no sequence dimension.

        Returns:
            Tensor of shape (batch, vocab_size).
        """
        return self._logits

    def supports_site(self, extractor: str) -> bool:
        """MLP bundles only support 'mlp' extraction; no residual stream or attention."""
        return extractor == "mlp"


class ModuloAddition2LMLPFamily(BaseModelFamily):
    """ModelFamily implementation for the 2-layer MLP on modular addition.

    Same task as ModuloAddition1LayerFamily, but the architecture is
    stripped to two fully-connected layers with no attention, no positional
    embedding, and no residual stream.

    Domain parameters: prime, seed, data_seed
    Input: one-hot concatenation of (a, b), size 2p
    Output: logits over p residue classes
    """

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> ModuloAddition2LMLP:
        """Create a ModuloAddition2LMLP for modular addition.

        Args:
            params: Domain parameters containing 'prime', 'seed'
            device: Device to place the model on

        Returns:
            ModuloAddition2LMLP configured for this prime
        """
        p = params["prime"]
        seed = params.get("seed", self.get_default_params().get("seed", 999))
        d_hidden = self.architecture.get("d_hidden", 512)

        model = ModuloAddition2LMLP(vocab_size=p, d_hidden=d_hidden, seed=seed)

        if device is not None:
            model = model.to(device)

        return model

    def generate_analysis_dataset(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate full (a, b) one-hot grid for analysis.

        Creates all p² input combinations encoded as one-hot concatenations.

        Args:
            params: Domain parameters containing 'prime'
            device: Device to place the dataset on

        Returns:
            Float tensor of shape (p², 2p) — one-hot (a, b) pairs
        """
        p = params["prime"]
        a_vals = torch.arange(p).repeat_interleave(p)
        b_vals = torch.arange(p).repeat(p)

        one_hot_a = torch.zeros(p * p, p)
        one_hot_b = torch.zeros(p * p, p)
        one_hot_a.scatter_(1, a_vals.unsqueeze(1), 1.0)
        one_hot_b.scatter_(1, b_vals.unsqueeze(1), 1.0)

        dataset = torch.cat([one_hot_a, one_hot_b], dim=1)

        if device is not None:
            dataset = dataset.to(device)

        return dataset

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.3,
        data_seed: int = 598,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test split for training.

        Uses the same data_seed mechanism as the 1-layer transformer family.

        Args:
            params: Domain parameters containing 'prime'
            training_fraction: Fraction of data to use for training
            data_seed: Random seed for train/test split
            device: Device to place tensors on

        Returns:
            Tuple of (train_data, train_labels, test_data, test_labels,
                     train_indices, test_indices)
        """
        p = params["prime"]

        dataset = self.generate_analysis_dataset(params, device=device)
        a_vals = torch.arange(p).repeat_interleave(p)
        b_vals = torch.arange(p).repeat(p)
        labels = (a_vals + b_vals) % p
        if device is not None:
            labels = labels.to(device)

        torch.manual_seed(data_seed)
        indices = torch.randperm(p * p)
        cutoff = int(p * p * training_fraction)
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]

        return (
            dataset[train_indices],
            labels[train_indices],
            dataset[test_indices],
            labels[test_indices],
            train_indices,
            test_indices,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss on MLP logits (batch, vocab_size).

        Args:
            logits: Shape (batch, vocab_size) from ModuloAddition2LMLP.
            labels: Target class indices of shape (batch,).

        Returns:
            Scalar mean negative log-probability of correct labels.
        """
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()

    def build_config_dict(
        self,
        model: ModuloAddition2LMLP,
        params: dict[str, Any],
        data_seed: int,
        training_fraction: float,
    ) -> dict[str, Any]:
        """Build config.json dict for a ModuloAddition2LMLP."""
        return {
            "architecture": "two_layer_mlp",
            "vocab_size": model.vocab_size,
            "d_hidden": model.d_hidden,
            "act_fn": "relu",
            **params,
            "model_seed": params.get("seed"),
            "data_seed": data_seed,
            "training_fraction": training_fraction,
        }

    def run_forward_pass(
        self,
        model: ModuloAddition2LMLP,
        probe: torch.Tensor,
    ) -> ModuloAddition2LMLPActivationBundle:
        """Run a forward pass and return a ModuloAddition2LMLPActivationBundle.

        Registers a forward hook on the ReLU to capture hidden activations,
        then runs the probe through the model.

        Args:
            model: ModuloAddition2LMLP instance created by create_model()
            probe: One-hot encoded probe tensor of shape (batch, 2p)

        Returns:
            ModuloAddition2LMLPActivationBundle with hidden activations and logits
        """
        captured: dict[str, torch.Tensor] = {}

        def _hook(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
            captured["hidden"] = output

        hook = model.relu.register_forward_hook(_hook)
        try:
            with torch.inference_mode():
                logits = model(probe)
        finally:
            hook.remove()

        return ModuloAddition2LMLPActivationBundle(model, captured["hidden"], logits)

    def prepare_analysis_context(
        self,
        params: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        """Prepare precomputed values for modular addition analysis.

        Provides the same fourier_basis key as the 1-layer transformer family,
        enabling all Fourier-based analyzers to run without modification.

        Args:
            params: Domain parameters containing 'prime'
            device: Device for tensor computations

        Returns:
            Dict with 'params', 'fourier_basis', and 'loss_fn'
        """
        p = params["prime"]
        fourier_basis, _ = get_fourier_basis(p, device)

        def loss_fn(model: ModuloAddition2LMLP, probe: torch.Tensor) -> float:
            """Cross-entropy loss on one-hot modular addition probe."""
            a_vals = probe[:, :p].argmax(dim=1)
            b_vals = probe[:, p:].argmax(dim=1)
            labels = (a_vals + b_vals) % p
            with torch.no_grad():
                logits = model(probe)
            log_probs = logits.log_softmax(dim=-1)
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

    def make_probe(
        self,
        params: dict[str, Any],
        inputs: list[list[int]],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Construct a one-hot probe tensor from (a, b) input pairs.

        Args:
            params: Domain parameters containing 'prime'
            inputs: List of [a, b] pairs (e.g., [[3, 29], [5, 7]])
            device: Device to place the tensor on

        Returns:
            Float tensor of shape (n, 2p) with one-hot (a, b) rows
        """
        p = params["prime"]
        n = len(inputs)

        one_hot_a = torch.zeros(n, p)
        one_hot_b = torch.zeros(n, p)
        for i, (a, b) in enumerate(inputs):
            one_hot_a[i, a] = 1.0
            one_hot_b[i, b] = 1.0

        probe = torch.cat([one_hot_a, one_hot_b], dim=1)
        if device is not None:
            probe = probe.to(device)
        return probe

    def get_training_config(self) -> dict[str, Any]:
        """Return default training hyperparameters.

        Calibrated to match He et al. (2602.16849) §3.3:
          - lr=1e-4, weight_decay=2.0, training_fraction=0.75
        These settings reliably produce grokking in 2L MLPs on modular addition.
        Weight decay is the primary driver of sparsification and generalization.

        Returns:
            Dict with learning_rate, weight_decay, betas, num_epochs,
            and checkpoint configuration
        """
        return {
            "learning_rate": 1e-4,
            "weight_decay": 1.0,
            "betas": (0.9, 0.98),
            "num_epochs": 50000,
            "default_checkpoint_epochs": sorted(
                list(
                    set(
                        [
                            *range(0, 1500, 100),
                            *range(1500, 12000, 500),
                            *range(12000, 22000, 100),
                            *range(22000, 50000, 500),
                        ]
                    )
                )
            ),
        }


def load_modulo_addition_2l_mlp_family(
    model_families_dir: Path | str = "model_families",
) -> ModuloAddition2LMLPFamily:
    """Load the 2-layer MLP family from the standard location.

    Args:
        model_families_dir: Path to model_families directory

    Returns:
        ModuloAddition2LMLPFamily instance
    """
    family_json = Path(model_families_dir) / "modulo_addition_2layer_mlp" / "family.json"
    family = ModuloAddition2LMLPFamily.from_json(family_json)
    assert isinstance(family, ModuloAddition2LMLPFamily)
    return family
