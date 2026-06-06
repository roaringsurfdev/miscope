"""REQ_077: Site Gradient Convergence Analyzer.

Post-hoc analyzer that computes per-site per-frequency gradient energy across
sampled training epochs. Samples are anchored to variant_summary.json window
boundaries; checkpoints are loaded directly (no per-epoch artifact dependency).

Artifact stored as: artifacts/gradient_site/cross_epoch.npz
"""

import shutil
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.inputs import ResolvedInputs
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.families.variant import Variant

_WINDOW_NAMES = [
    "first_descent_window",
    "plateau_window",
    "second_descent_window",
    "final_window",
]

_SITES = ("embedding", "attention", "mlp")


# Gradient energy/magnitude per component site (embedding/attention/mlp) and
# pairwise gradient-direction similarities, sampled over a training window.
# On-disk keys are {field}_{site} / {field}_{pair}; declared as logical fields
# keyed by `site` (component) or `group` (the component pair).
SPEC = AnalyzerSpec(
    name="gradient_site",
    output_scope="cross_epoch",
    inputs=(),  # loads checkpoints directly via context["variant"]
    outputs=(
        F.columnar(
            "epochs",
            "int64",
            ("variant", "epoch"),
            "Sampled epoch axis labels for the gradient trajectories.",
        ),
        F.columnar(
            "energy",
            "float64",
            ("variant", "epoch", "site", "frequency"),
            "Per-frequency gradient energy at a component site over the window.",
        ),
        F.columnar(
            "magnitude",
            "float64",
            ("variant", "epoch", "site"),
            "Gradient magnitude at a component site over the window.",
        ),
        F.columnar(
            "similarity",
            "float64",
            ("variant", "epoch", "group"),
            "Cosine similarity of gradient directions between a component pair "
            "(group ∈ {emb_attn, emb_mlp, attn_mlp}).",
        ),
        F.columnar(
            "key_frequencies",
            "int64",
            ("variant", "frequency"),
            "Frequencies flagged as key during the analysis window.",
        ),
        F.columnar(
            "window_epochs",
            "int64",
            ("variant", "row_id"),
            "Boundary epochs delimiting the analysis window.",
        ),
        F.columnar(
            "prime",
            "int64",
            ("variant",),
            "Modulus the analysis was run against.",
        ),
    ),
)


@register_analyzer(SPEC)
class GradientSiteAnalyzer:
    """Cross-epoch analyzer for site-level gradient convergence.

    Computes per-site per-frequency gradient energy at window-sampled epochs.
    Stores direction-normalized energy, raw magnitudes, and pairwise cosine
    similarities between site frequency spectra.

    Run through the pipeline (as a cross-epoch analyzer) or standalone
    via GradientSiteAnalyzer().run(variant).
    """

    name = "gradient_site"
    # Reads transformer weights via canonical names (embed.W_E, attn Q/K/V,
    # mlp.in.W). Family registration filters non-transformer architectures.
    requires: list[str] = []  # loads checkpoints directly; no per-epoch deps

    def __init__(self, n_interior: int = 2, full_resolution: bool = False) -> None:
        self.n_interior = n_interior
        self.full_resolution = full_resolution

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute site gradient convergence artifact for a variant.

        Requires context["variant"] — injected by the pipeline when calling
        cross-epoch analyzers. Use run(variant) for standalone execution.

        Args:
            inputs: ``inputs.epochs`` (available checkpoints, used for snapping);
                checkpoints/data are loaded directly via ``context["variant"]``.
            context: Must include "variant"; "fourier_basis" used if present

        Returns:
            Artifact dict suitable for npz storage.
        """
        variant: Variant = context["variant"]
        prime = int(variant.model_config["prime"])
        n_freqs = prime // 2
        epochs = list(inputs.epochs or ())

        fourier_basis = context.get("fourier_basis")
        if fourier_basis is None:
            from miscope.analysis.library import get_fourier_basis

            fourier_basis, _ = get_fourier_basis(prime)

        summary = _load_variant_summary(variant)
        if self.full_resolution:
            sampled_epochs = sorted(epochs)
        else:
            requested = _sample_window_epochs(summary, self.n_interior)
            sampled_epochs = _snap_to_available(requested, epochs)

        td, tl, *_ = variant.generate_training_dataset()

        raw_energies: dict[str, list[np.ndarray]] = {s: [] for s in _SITES}
        magnitudes: dict[str, list[float]] = {s: [] for s in _SITES}

        for epoch in sampled_epochs:
            model = variant.load_model_at_checkpoint(epoch)
            model.eval()
            device = next(model.parameters()).device
            site_energies = _fourier_gradient_by_site(
                model, td.to(device), tl.to(device), prime, fourier_basis, n_freqs
            )
            model.zero_grad()
            del model
            for site in _SITES:
                raw = site_energies[site]
                magnitudes[site].append(float(np.linalg.norm(raw)))
                raw_energies[site].append(raw)

        n_sampled = len(sampled_epochs)

        # Direction-normalized energy for storage (visualization primary signal)
        energy_arrays = {
            site: np.stack(
                [_normalize_or_zero(raw_energies[site][i]) for i in range(n_sampled)], axis=0
            )
            for site in _SITES
        }

        # Pairwise cosine similarities using normalized vectors; NaN for zero-norm epochs
        def _cosine_sims(a_key: str, b_key: str) -> np.ndarray:
            out = np.full(n_sampled, np.nan)
            for i in range(n_sampled):
                ma, mb = magnitudes[a_key][i], magnitudes[b_key][i]
                if ma > 1e-30 and mb > 1e-30:
                    a_norm = energy_arrays[a_key][i]
                    b_norm = energy_arrays[b_key][i]
                    out[i] = float(np.dot(a_norm, b_norm))
            return out

        return {
            "epochs": np.array(sampled_epochs, dtype=np.int64),
            "energy_embedding": energy_arrays["embedding"],
            "energy_attention": energy_arrays["attention"],
            "energy_mlp": energy_arrays["mlp"],
            "magnitude_embedding": np.array(magnitudes["embedding"]),
            "magnitude_attention": np.array(magnitudes["attention"]),
            "magnitude_mlp": np.array(magnitudes["mlp"]),
            "similarity_emb_attn": _cosine_sims("embedding", "attention"),
            "similarity_emb_mlp": _cosine_sims("embedding", "mlp"),
            "similarity_attn_mlp": _cosine_sims("attention", "mlp"),
            "key_frequencies": np.array(_get_key_frequencies(summary), dtype=np.int64),
            "window_epochs": np.array(_get_window_boundary_epochs(summary), dtype=np.int64),
            "prime": np.array([prime], dtype=np.int64),
        }

    def run(self, variant: "Variant", force: bool = False) -> None:
        """Standalone runner — computes and saves the artifact for a variant.

        Args:
            variant: Variant instance to analyze
            force: If True, recompute even if artifact already exists
        """
        cross_epoch_path = variant.artifacts_dir / "gradient_site" / "cross_epoch.npz"
        if cross_epoch_path.exists() and not force:
            return

        prime = int(variant.model_config["prime"])
        from miscope.analysis.library import get_fourier_basis

        fourier_basis, _ = get_fourier_basis(prime)
        context: dict[str, Any] = {
            "variant": variant,
            "fourier_basis": fourier_basis,
            "params": variant.params,
        }
        available_epochs = variant.get_available_checkpoints()
        inputs = ResolvedInputs(epochs=tuple(available_epochs))
        result = self.analyze(inputs, context)

        cross_epoch_path.parent.mkdir(parents=True, exist_ok=True)
        temp_base = str(cross_epoch_path.parent / ".cross_epoch_tmp")
        np.savez_compressed(temp_base, **result)  # type: ignore[reportArgumentType]
        shutil.move(temp_base + ".npz", str(cross_epoch_path))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_variant_summary(variant: "Variant") -> dict[str, Any]:
    return variant.summary


def _sample_window_epochs(summary: dict[str, Any], n_interior: int) -> list[int]:
    """Return sorted unique epochs spanning window boundaries plus interior samples."""
    epochs: set[int] = set()
    for name in _WINDOW_NAMES:
        w = summary.get(name)
        if w is None:
            continue
        start, end = w["start_epoch"], w["end_epoch"]
        epochs.add(start)
        epochs.add(end)
        if n_interior > 0 and end > start:
            step = (end - start) / (n_interior + 1)
            for i in range(1, n_interior + 1):
                epochs.add(round(start + i * step))
    return sorted(epochs)


def _snap_to_available(requested: list[int], available: list[int]) -> list[int]:
    """Map each requested epoch to the nearest available checkpoint."""
    avail_sorted = sorted(available)
    snapped = {min(avail_sorted, key=lambda a: abs(a - ep)) for ep in requested}
    return sorted(snapped)


def _get_key_frequencies(summary: dict[str, Any]) -> list[int]:
    """Return dominant committed frequencies from the final window."""
    final = summary.get("final_window")
    if final is None:
        return []
    return final.get("committed_frequencies_end") or []


def _get_window_boundary_epochs(summary: dict[str, Any]) -> list[int]:
    """Return sorted unique epoch values that mark window start/end boundaries."""
    epochs: set[int] = set()
    for name in _WINDOW_NAMES:
        w = summary.get(name)
        if w:
            epochs.add(w["start_epoch"])
            epochs.add(w["end_epoch"])
    return sorted(epochs)


def _fourier_gradient_by_site(
    model: Any,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    prime: int,
    fourier_basis: torch.Tensor,
    n_freqs: int,
) -> dict[str, np.ndarray]:
    """Compute per-frequency gradient energy at embedding, attention, and MLP sites.

    All three sites are projected through W_E[:p] into token space before
    Fourier decomposition, making spectra directly comparable. Energy per
    frequency = RMS over all output dimensions.

    Args:
        model: ``HookedModel`` with weights loaded from a checkpoint.
            Must publish ``embed.W_E``, attention Q/K/V, and MLP in
            weights — i.e., a transformer-class architecture.
        train_data: Training input tensor, shape (n, seq_len)
        train_labels: Training label tensor, shape (n,)
        prime: Modulus p (number of token positions in frequency projection)
        fourier_basis: Fourier basis tensor, shape (p+1, p)
        n_freqs: Number of frequency components (= p // 2)

    Returns:
        Dict with keys 'embedding', 'attention', 'mlp', each ndarray(n_freqs,).
    """
    model.zero_grad()
    logits = model(train_data)[:, -1, :prime]
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    loss.backward()

    W_E_param = model.get_weight("embed.W_E")
    W_E = W_E_param.detach()  # (d_vocab, d_model)
    F_dev = fourier_basis.to(W_E.device)

    def _freq_energy(fourier_projected: torch.Tensor) -> np.ndarray:
        fg = fourier_projected.detach().cpu().numpy()
        energy = np.zeros(n_freqs)
        for k in range(1, n_freqs + 1):
            sin_row = fg[2 * k - 1]
            cos_row = fg[2 * k]
            energy[k - 1] = np.sqrt(np.mean(sin_row**2 + cos_row**2))
        return energy

    # Embedding: grad_W_E[:p] is (p, d_model) — already in token space
    grad_W_E = W_E_param.grad[:prime]
    emb_energy = _freq_energy(F_dev @ grad_W_E)

    # Attention: Q, K, V each (n_heads, d_model, d_head); project per head, combine as RMS
    attn_sq = np.zeros(n_freqs)
    n_contributions = 0
    for canonical_w in (
        "blocks.0.attn.q.W",
        "blocks.0.attn.k.W",
        "blocks.0.attn.v.W",
    ):
        grad_W = model.get_weight(canonical_w).grad  # (n_heads, d_model, d_head)
        for h in range(grad_W.shape[0]):
            projected = W_E[:prime] @ grad_W[h]  # (p, d_head)
            fg = (F_dev @ projected).detach().cpu().numpy()
            for k in range(1, n_freqs + 1):
                attn_sq[k - 1] += np.mean(fg[2 * k - 1] ** 2 + fg[2 * k] ** 2)
            n_contributions += 1
    attn_energy = np.sqrt(attn_sq / n_contributions)

    # MLP: W_in is (d_model, d_mlp); project gradient through W_E
    grad_W_in = model.get_weight("blocks.0.mlp.in.W").grad
    mlp_energy = _freq_energy(F_dev @ (W_E[:prime] @ grad_W_in))

    return {"embedding": emb_energy, "attention": attn_energy, "mlp": mlp_energy}


def _normalize_or_zero(vec: np.ndarray) -> np.ndarray:
    """Normalize vector by L2 norm; return zeros for negligible-norm vectors."""
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-30 else np.zeros_like(vec)
