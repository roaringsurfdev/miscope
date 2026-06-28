"""Modulo Addition 1-Layer family implementation.

This module provides the concrete implementation for the modulo addition
single-layer transformer family, based on Neel Nanda's grokking experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch

from miscope.analysis.library import get_fourier_basis
from miscope.analysis.library.fourier import (
    compose_neuron_fourier_weights,
    extract_frequency_pairs,
)
from miscope.analysis.library.grouping import group_neurons
from miscope.architectures import HookedTransformer, HookedTransformerConfig
from miscope.core.basis_projection import BasisProjectionSite
from miscope.core.grouping import GroupAssignment
from miscope.families.base_model_family import BaseModelFamily

# REQ_118: confidence threshold (variance fraction of W_in Fourier
# projection) for the modadd grouping override. Calibrated empirically
# against canon (p113/s999/ds598) at epoch 24999 — the four documented
# canon frequencies {9, 33, 38, 55} are recovered cleanly across the
# 0.1–0.5 range; 0.3 is the sweet spot where canon frequencies are
# fully recovered while ~4% of neurons with truly diffuse projections
# are correctly marked UNASSIGNED. Weight-side variance fractions are
# naturally lower than the activation-side fractions used by the
# project's neuron_dynamics convention (0.7), since W_in is
# higher-dimensional than per-input activation profiles.
_DEFAULT_FOURIER_GROUPING_THRESHOLD = 0.3


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
        dtype: torch.dtype | None = None,
    ) -> HookedTransformer:
        """Create a miscope ``HookedTransformer`` for modular addition.

        Returns the canonical-name-aware subclass that quarantines the
        underlying TransformerLens dependency (REQ_112). Existing
        checkpoints load unchanged because the subclass preserves TL's
        parameter layout and state-dict format.

        Args:
            params: Domain parameters containing 'prime' and optionally 'seed'
            device: Device to place the model on (default: None, uses default device)
            dtype: Forward-pass dtype. ``None`` keeps TransformerLens' float32
                default; ``torch.float64`` builds weights and runs the forward
                pass in double precision (the float64-instability experiment).

        Returns:
            ``HookedTransformer`` configured for modular addition.
        """
        p = params["prime"]
        seed = params.get("seed", self.get_default_params().get("seed", 999))

        arch = self.architecture

        cfg_kwargs: dict[str, Any] = dict(
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
        # Only override TransformerLens' own dtype default (float32) when a
        # caller explicitly requests another precision (e.g. float64).
        if dtype is not None:
            cfg_kwargs["dtype"] = dtype

        cfg = HookedTransformerConfig(**cfg_kwargs)
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
            # REQ_118: family-supplied neuron grouping override. The
            # NeuronGrouping analyzer dispatches on this when present and
            # bypasses the universal kmeans path.
            "neuron_grouping_override": self._neuron_grouping_override,
            # REQ_126: family-supplied sites for basis-projection analyzers.
            "weight_basis_projection_sites": self.weight_basis_projection_sites,
            "activation_frequency_norm_sites": self.activation_frequency_norm_sites,
            # REQ_152: family-supplied composed-circuit sites for full_ov_circuit.
            "circuit_spectra_sites": self.circuit_spectra_sites,
        }

    def _neuron_grouping_override(
        self,
        artifact: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[GroupAssignment, np.ndarray]:
        """REQ_118: Modadd-family Fourier-based neuron grouping override.

        Each neuron is grouped by the Fourier frequency that dominates its
        composed input weight (W_E[:p] @ W_in[:, neuron], projected onto
        the family's Fourier basis). Neurons whose dominant-frequency
        magnitude fraction falls below a confidence threshold are marked
        UNASSIGNED.

        Returns:
            (assignment, features) where features are the per-neuron
            per-frequency magnitude matrix used for the assignment and
            for the downstream `group_neurons_summary` computation.
        """
        prime = int(context["params"]["prime"])
        fourier_basis = context["fourier_basis"]
        threshold = float(
            context.get(
                "neuron_grouping_confidence_threshold",
                _DEFAULT_FOURIER_GROUPING_THRESHOLD,
            )
        )

        # Compose input-side weight: theta[token, neuron].
        theta_t, _ = compose_neuron_fourier_weights(artifact, prime)
        if isinstance(theta_t, torch.Tensor):
            theta_np = theta_t.detach().cpu().numpy()
        else:
            theta_np = np.asarray(theta_t)
        # Project onto Fourier basis.
        if isinstance(fourier_basis, torch.Tensor):
            basis_np = fourier_basis.detach().cpu().numpy()
        else:
            basis_np = np.asarray(fourier_basis)
        fourier_coeffs = basis_np @ theta_np  # (prime, d_mlp)
        # Per-neuron per-frequency magnitudes.
        magnitudes, _ = extract_frequency_pairs(fourier_coeffs, prime)  # (d_mlp, K)

        assignment = group_neurons(
            magnitudes,
            n_groups=int(magnitudes.shape[1]),
            method="argmax_by_basis",
            feature_basis_name="fourier_w_in",
            confidence_threshold=threshold,
        )
        return assignment, magnitudes

    @property
    def weight_basis_projection_sites(self) -> tuple[BasisProjectionSite, ...]:
        """REQ_126: weight-side sites for ``weight_basis_projection``.

        Five sites cover the absorbed Fourier analyzers:
        - ``embedding`` reproduces ``dominant_frequencies``.
        - ``attn_v`` and ``attn_qk`` together reproduce ``attention_fourier``.
        - ``mlp_in`` reproduces ``neuron_fourier``'s theta-side and
          ``fourier_nucleation``'s one-shot projection.
        - ``mlp_out`` reproduces ``neuron_fourier``'s xi-side.
        """
        return (
            BasisProjectionSite(
                name="embedding",
                compose=_compose_embedding,
                period_axes=(0,),
                description="W_E[:p] — token embeddings, excluding equals token",
            ),
            BasisProjectionSite(
                name="attn_v",
                compose=_compose_attn_v,
                period_axes=(1,),
                description="(W_E[:p] @ W_V[h]) per head: (n_heads, p, d_head)",
            ),
            BasisProjectionSite(
                name="attn_qk",
                compose=_compose_attn_qk,
                period_axes=(1, 2),
                description="(W_E[:p] @ W_Q[h]) (W_E[:p] @ W_K[h])^T per head: (n_heads, p, p)",
            ),
            BasisProjectionSite(
                name="mlp_in",
                compose=_compose_mlp_in,
                period_axes=(0,),
                description="W_E[:p] @ W_in — composed neuron input weight: (p, d_mlp)",
            ),
            BasisProjectionSite(
                name="mlp_out",
                compose=_compose_mlp_out,
                period_axes=(0,),
                description="(W_out @ W_U)^T — composed neuron output weight: (p, d_mlp)",
            ),
            # REQ_152: the full OV circuit as a 2D Fourier site → its task-conditional
            # dominant_frequency comes from this universal instrument (CHEAP repoint),
            # not re-derived in full_ov_circuit.
            _FULL_OV_SITE,
        )

    @property
    def circuit_spectra_sites(self) -> tuple[BasisProjectionSite, ...]:
        """REQ_152: composed-circuit sites for the ``full_ov_circuit`` analyzer.

        Currently the single ``full_ov`` path; sibling circuits (OV, QK, full QK,
        direct path) are added here as the data model's Layer 4 buildout proceeds.
        """
        return (_FULL_OV_SITE,)

    @property
    def activation_frequency_norm_sites(self) -> tuple[BasisProjectionSite, ...]:
        """REQ_126: activation-side sites for ``activation_frequency_norm``.

        Two sites absorb the activation-side Fourier analyzers:
        - ``attn_pattern`` reproduces ``attention_freq`` (post-softmax
          attention pattern on the equals-token row, reshaped to a (p, p)
          (a, b) grid per head).
        - ``mlp_out`` reproduces ``neuron_freq_norm`` (MLP last-position
          activations reshaped to a (p, p) (a, b) grid per neuron) and is
          the REQ_102 gate for retiring ``coarseness``.
        """
        return (
            BasisProjectionSite(
                name="attn_pattern",
                compose=_compose_attn_pattern_activation,
                period_axes=(1, 2),
                description=("attention pattern at (to=2, from=0) reshaped to (n_heads, p, p)"),
                required_hooks=("blocks.0.attn.hook_pattern",),
            ),
            BasisProjectionSite(
                name="mlp_out",
                compose=_compose_mlp_out_activation,
                period_axes=(1, 2),
                description="MLP last-position activations reshaped to (d_mlp, p, p)",
                required_hooks=("blocks.0.mlp.hook_out",),
            ),
        )

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


def _compose_embedding(snapshot: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """``embedding`` site: token embedding rows for 0..p-1."""
    p = int(context["params"]["prime"])
    return np.asarray(snapshot["W_E"])[:p]


def _compose_mlp_in(snapshot: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """``mlp_in`` site: per-token composed input weight ``W_E[:p] @ W_in``.

    Result shape: ``(p, d_mlp)``. Each column is one neuron's effective
    response to each token value 0..p-1 — the same matrix used by
    ``neuron_fourier`` (theta) and ``fourier_nucleation``'s one-shot
    projection.
    """
    p = int(context["params"]["prime"])
    W_E = np.asarray(snapshot["W_E"])
    W_in = np.asarray(snapshot["W_in"])
    return W_E[:p] @ W_in


def _compose_mlp_out(snapshot: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """``mlp_out`` site: per-token composed output weight ``(W_out @ W_U)^T``.

    Result shape: ``(p, d_mlp)``. Matches ``neuron_fourier``'s xi convention.
    """
    p = int(context["params"]["prime"])
    W_out = np.asarray(snapshot["W_out"])
    W_U = np.asarray(snapshot["W_U"])
    return (W_out @ W_U[:, :p]).T  # type: ignore[no-any-return]


def _compose_full_ov(snapshot: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """``full_ov`` site: per-head end-to-end OV circuit in token space (REQ_152).

    ``M[h] = W_E[:p] @ W_V[h] @ W_O[h] @ W_U[:, :p]`` — the source-token →
    output-logit map through head ``h``'s OV path, restricted to the ``p`` task
    tokens (excludes the equals token). ``M[h][src, out]`` is the logit that source
    token ``src`` contributes to output token ``out`` via head ``h``. Result shape
    ``(n_heads, p, p)``; period axes 1 and 2 (2D Fourier). Spectral invariants
    (SVD, eigenvalues) are transpose-invariant, so the src/out orientation does not
    affect ``full_ov_circuit``'s measurements.
    """
    p = int(context["params"]["prime"])
    W_E = np.asarray(snapshot["W_E"])[:p]  # (p, d_model)
    W_V = np.asarray(snapshot["W_V"])  # (n_heads, d_model, d_head)
    W_O = np.asarray(snapshot["W_O"])  # (n_heads, d_head, d_model)
    W_U = np.asarray(snapshot["W_U"])[:, :p]  # (d_model, p)
    ev = np.einsum("td,hdk->htk", W_E, W_V)  # (n_heads, p, d_head)
    evo = np.einsum("htk,hkm->htm", ev, W_O)  # (n_heads, p, d_model)
    return np.einsum("htm,mo->hto", evo, W_U)  # (n_heads, p, p)


# REQ_152: the full OV circuit, declared once and shared by the Fourier instrument
# (weight_basis_projection_sites → dominant_frequency) and the spectral instrument
# (circuit_spectra_sites → copying_score / effective_rank / operator_norm), so the
# composition has a single source of truth.
_FULL_OV_SITE = BasisProjectionSite(
    name="full_ov",
    compose=_compose_full_ov,
    period_axes=(1, 2),
    description="Full OV circuit W_E[:p] W_V[h] W_O[h] W_U[:,:p] per head: (n_heads, p, p)",
)


def _compose_attn_v(snapshot: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """``attn_v`` site: per-head value projection of token embeddings.

    Result shape: ``(n_heads, p, d_head)``. Period axis is axis 1.
    """
    p = int(context["params"]["prime"])
    W_E = np.asarray(snapshot["W_E"])[:p]  # (p, d_model)
    W_V = np.asarray(snapshot["W_V"])  # (n_heads, d_model, d_head)
    return np.einsum("td,hdk->htk", W_E, W_V)


def _compose_attn_qk(snapshot: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """``attn_qk`` site: per-head ``Q K^T`` in token space.

    Result shape: ``(n_heads, p, p)``. Period axes are 1 and 2 (2D Fourier).
    """
    p = int(context["params"]["prime"])
    W_E = np.asarray(snapshot["W_E"])[:p]  # (p, d_model)
    W_Q = np.asarray(snapshot["W_Q"])  # (n_heads, d_model, d_head)
    W_K = np.asarray(snapshot["W_K"])  # (n_heads, d_model, d_head)
    Q = np.einsum("td,hdk->htk", W_E, W_Q)  # (n_heads, p, d_head)
    K = np.einsum("td,hdk->htk", W_E, W_K)  # (n_heads, p, d_head)
    return np.einsum("htk,hsk->hts", Q, K)


def _compose_attn_pattern_activation(cache: Any, context: dict[str, Any]) -> np.ndarray:
    """``attn_pattern`` activation site: per-head equals-token attention reshaped to (a, b) grid.

    Pulls the post-softmax pattern at ``blocks.0.attn.hook_pattern`` shape
    ``(p^2, n_heads, n_pos, n_pos)``, selects the ``(to=2, from=0)`` cell,
    and reshapes the leading batch axis as a ``(p, p)`` (a, b) grid.

    Result shape: ``(n_heads, p, p)``. Period axes: (1, 2).
    """
    p = int(context["params"]["prime"])
    attn = cache["blocks.0.attn.hook_pattern"]
    attn_pair = attn[:, :, 2, 0]  # (p^2, n_heads)
    if hasattr(attn_pair, "detach"):
        attn_pair = attn_pair.detach().cpu().numpy()
    else:
        attn_pair = np.asarray(attn_pair)
    grid = attn_pair.reshape(p, p, -1).transpose(2, 0, 1)  # (n_heads, p, p)
    return grid


def _compose_mlp_out_activation(cache: Any, context: dict[str, Any]) -> np.ndarray:
    """``mlp_out`` activation site: last-position MLP activations on the (a, b) grid.

    Pulls ``blocks.0.mlp.hook_out`` shape ``(p^2, seq_len, d_mlp)``,
    selects the last position, and reshapes to a ``(p, p)`` (a, b) grid
    per neuron.

    Result shape: ``(d_mlp, p, p)``. Period axes: (1, 2).
    """
    p = int(context["params"]["prime"])
    acts = cache["blocks.0.mlp.hook_out"]
    if hasattr(acts, "ndim") and acts.ndim == 3:
        acts = acts[:, -1, :]
    if hasattr(acts, "detach"):
        acts = acts.detach().cpu().numpy()
    else:
        acts = np.asarray(acts)
    grid = acts.reshape(p, p, -1).transpose(2, 0, 1)  # (d_mlp, p, p)
    return grid


def load_modulo_addition_1layer_family(
    data_root: Path | str = "data",
) -> ModuloAddition1LayerFamily:
    """Load the modulo addition 1-layer family from the unified data root.

    Args:
        data_root: Path to the unified data root (default: ``data``).

    Returns:
        ModuloAddition1LayerFamily instance
    """
    family_json = Path(data_root) / "modulo_addition_1layer" / "family.json"
    family = ModuloAddition1LayerFamily.from_json(family_json, data_root=data_root)
    assert isinstance(family, ModuloAddition1LayerFamily)
    return family


# Self-register on import so the discovery helper can route family.json
# files with name="modulo_addition_1layer" to this implementation.
from miscope.families.discovery import register_family_implementation  # noqa: E402

register_family_implementation("modulo_addition_1layer", ModuloAddition1LayerFamily)
