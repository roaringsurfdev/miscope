"""Learned-Embedding MLP family for modular addition.

Architecture: two separate learned embeddings for inputs a and b,
summed into a shared d_embed-dimensional representation, then passed
through a single hidden layer with ReLU activation.

The sum embedding forces the network to encode a + b in a shared space,
creating representational pressure analogous to the transformer's
residual stream — but without the attention routing mechanism.

This is rung 2 of the architecture ladder:
  1. One-hot MLP  (no learned repr, no attention)  → no geometry
  2. Learned-embedding MLP (learned repr, no attention) → ?
  3. 1-layer Transformer (learned repr + attention)  → geometry

Input encoding: integer indices a, b ∈ {0, ..., p-1} (not one-hot).
The embedding layers handle encoding; output is logits over p classes.

REQ_113 migration
-----------------
The standalone ``ModuloAdditionEmbedMLP`` ``nn.Module`` and its
``ModuloAdditionEmbedMLPActivationBundle`` are retired. The model is
now :class:`miscope.architectures.HookedEmbeddingMLP`. Per-input
embeddings are still published as ``embed.embed_a`` / ``embed.embed_b``
(never ``embed.W_E``); see :mod:`miscope.architectures.hooked_embedding_mlp`
for the embedding-identity invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from miscope.analysis.library import get_fourier_basis
from miscope.architectures import HookedEmbeddingMLP, HookedEmbeddingMLPConfig
from miscope.families.base_model_family import BaseModelFamily


class ModuloAdditionEmbedMLPFamily(BaseModelFamily):
    """ModelFamily for the learned-embedding MLP on modular addition.

    Same task as the one-hot MLP and 1-layer transformer, but with
    learned embeddings and no attention or residual stream. The two
    input embeddings (a, b) are summed before the hidden layer.

    Domain parameters: prime, seed, data_seed
    Input: integer indices a, b ∈ {0, ..., p-1}
    Output: logits over p residue classes
    """

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> HookedEmbeddingMLP:
        p = params["prime"]
        seed = params.get("seed", self.get_default_params().get("seed", 999))
        d_embed = self.architecture.get("d_embed", 16)
        d_hidden = self.architecture.get("d_hidden", 512)

        model = HookedEmbeddingMLP(
            HookedEmbeddingMLPConfig(vocab_size=p, d_embed=d_embed, d_hidden=d_hidden, seed=seed)
        )
        if device is not None:
            model = model.to(device)
        return model

    def generate_analysis_dataset(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate full (a, b) integer index grid for analysis.

        Returns:
            Long tensor of shape (p², 2) — each row is (a, b) index pair.
        """
        p = params["prime"]
        a_vals = torch.arange(p).repeat_interleave(p)
        b_vals = torch.arange(p).repeat(p)
        probe = torch.stack([a_vals, b_vals], dim=1)
        if device is not None:
            probe = probe.to(device)
        return probe

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.75,
        data_seed: int = 598,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test split for training.

        Returns:
            6-tuple of (train_data, train_labels, test_data, test_labels,
                        train_indices, test_indices) matching the ModelFamily protocol.
            train_data / test_data are (N, 2) long tensors of (a, b) index pairs.
        """
        p = params["prime"]
        probe = self.generate_analysis_dataset(params, device=device)
        a_vals, b_vals = probe.unbind(1)
        labels = (a_vals + b_vals) % p

        torch.manual_seed(data_seed)
        indices = torch.randperm(p * p)
        if device is not None:
            indices = indices.to(device)
        cutoff = int(p * p * training_fraction)
        train_idx = indices[:cutoff]
        test_idx = indices[cutoff:]

        return (
            probe[train_idx],
            labels[train_idx],
            probe[test_idx],
            labels[test_idx],
            train_idx,
            test_idx,
        )

    def prepare_analysis_context(
        self,
        params: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        """Prepare precomputed values for modular addition analysis."""
        p = params["prime"]
        fourier_basis, _ = get_fourier_basis(p, device)

        def loss_fn(model: HookedEmbeddingMLP, probe: torch.Tensor) -> float:
            a_vals, b_vals = probe.unbind(1)
            labels = (a_vals + b_vals) % p
            with torch.no_grad():
                logits = model(probe)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean()
            return loss.item()

        ab = self.generate_analysis_dataset(params)
        a_vals, b_vals = ab.unbind(1)
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
        """Construct a (N, 2) index probe tensor from (a, b) input pairs.

        Returns:
            Long tensor of shape (n, 2) — each row is (a, b).
        """
        probe = torch.tensor(inputs, dtype=torch.long)
        if device is not None:
            probe = probe.to(device)
        return probe

    def get_training_config(self) -> dict[str, Any]:
        """Return default training hyperparameters.

        Calibrated to match He et al. (2602.16849) §3.3 settings used for
        the one-hot 2L MLP. Embedding parameters may behave differently
        under weight decay — monitor training curves and adjust if needed.
        d_embed=16 is chosen to sit below the transformer's observed W_E
        participation ratio (23.45) while staying above the theoretical
        floor (2|F|=8 for 4 frequencies at p=113).
        """
        return {
            "learning_rate": 1e-4,
            "weight_decay": 2.0,
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

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss on MLP logits (batch, vocab_size).

        Args:
            logits: Shape (batch, vocab_size) from the model.
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
        model: HookedEmbeddingMLP,
        params: dict[str, Any],
        data_seed: int,
        training_fraction: float,
    ) -> dict[str, Any]:
        """Build config.json dict for the model."""
        return {
            "architecture": "learned_emb_mlp",
            "vocab_size": model.vocab_size,
            "d_embed": model.d_embed,
            "d_hidden": model.d_hidden,
            "act_fn": "relu",
            **params,
            "model_seed": params.get("seed"),
            "data_seed": data_seed,
            "training_fraction": training_fraction,
        }


def load_modulo_addition_embed_mlp_family(
    data_root: Path | str = "data",
) -> ModuloAdditionEmbedMLPFamily:
    """Load the learned-embedding MLP family from the unified data root."""
    family_json = Path(data_root) / "modulo_addition_learned_emb_mlp" / "family.json"
    family = ModuloAdditionEmbedMLPFamily.from_json(family_json, data_root=data_root)
    assert isinstance(family, ModuloAdditionEmbedMLPFamily)
    return family


from miscope.families.discovery import register_family_implementation  # noqa: E402

register_family_implementation("modulo_addition_learned_emb_mlp", ModuloAdditionEmbedMLPFamily)
