"""Canonical weight names for miscope architectures.

Distinct from ``miscope.core.architecture`` (which names *activation* hook
points). This module names *learned parameters* — the tensors a model
exposes via ``HookedModel.get_weight(canonical_name)``.

Spec adopted: 2026-04-17 (TL 3.0 release).
https://transformerlensorg.github.io/TransformerLens/content/model_structure.html

Naming convention
-----------------
Top-level components (``embed``, ``pos_embed``, ``unembed``) use the
historical ``W_E`` / ``W_pos`` / ``W_U`` and ``b_E`` / ``b_pos`` / ``b_U``
suffixes — that's the spelling researchers already know. Block-internal
components (``attn.q``, ``mlp.in``, etc.) use the unsuffixed ``W`` / ``b``
because the sub-component path already disambiguates.

Embedding-MLP variant
---------------------
Models with two per-input embedding matrices (e.g.,
``HookedEmbeddingMLP`` from REQ_113) publish ``embed.embed_a`` and
``embed.embed_b`` rather than ``embed.W_E``. The distinction is
load-bearing: an analyzer that reads ``embed.W_E`` against an
embedding-MLP must fail loudly (``KeyError``) rather than be silently
aliased to one of the per-input matrices.
"""

from __future__ import annotations

from miscope.core.architecture import (
    ATTN,
    BLOCKS,
    EMBED,
    MLP,
    POS_EMBED,
    UNEMBED,
)

# Generic suffixes (used for block-internal sub-components)
W = "W"
B = "b"

# Top-level weight suffixes (legacy notation, kept for researcher familiarity)
W_E = "W_E"
B_E = "b_E"
W_POS = "W_pos"
B_POS = "b_pos"
W_U = "W_U"
B_U = "b_U"

# Per-input embeddings (embedding-MLP variant; not aliased to W_E)
EMBED_A = "embed_a"
EMBED_B = "embed_b"

# Top-level canonical paths
EMBED_WEIGHT = f"{EMBED}.{W_E}"
EMBED_BIAS = f"{EMBED}.{B_E}"
POS_EMBED_WEIGHT = f"{POS_EMBED}.{W_POS}"
POS_EMBED_BIAS = f"{POS_EMBED}.{B_POS}"
UNEMBED_WEIGHT = f"{UNEMBED}.{W_U}"
UNEMBED_BIAS = f"{UNEMBED}.{B_U}"
EMBED_A_PATH = f"{EMBED}.{EMBED_A}"
EMBED_B_PATH = f"{EMBED}.{EMBED_B}"


def block_attn_weight(layer: int, sub: str, suffix: str = W) -> str:
    """e.g. ``block_attn_weight(0, "q") == "blocks.0.attn.q.W"``."""
    return f"{BLOCKS}.{layer}.{ATTN}.{sub}.{suffix}"


def block_attn_bias(layer: int, sub: str) -> str:
    """e.g. ``block_attn_bias(0, "q") == "blocks.0.attn.q.b"``."""
    return block_attn_weight(layer, sub, suffix=B)


def block_mlp_weight(layer: int, sub: str, suffix: str = W) -> str:
    """e.g. ``block_mlp_weight(0, "in") == "blocks.0.mlp.in.W"``."""
    return f"{BLOCKS}.{layer}.{MLP}.{sub}.{suffix}"


def block_mlp_bias(layer: int, sub: str) -> str:
    """e.g. ``block_mlp_bias(0, "in") == "blocks.0.mlp.in.b"``."""
    return block_mlp_weight(layer, sub, suffix=B)
