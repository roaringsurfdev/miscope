"""Canonical component and hook names for miscope architectures.

This module is the platform's single source of truth for component names
and hook point names. Analyzers, renderers, and other consumers reference
the constants and helpers defined here; raw string literals like
``"blocks.0.mlp.hook_pre"`` are reserved for concrete ``HookedModel``
subclasses that translate canonical names to a backend's spelling.

The vocabulary mirrors the TransformerLens 3.0 Model Structure page:
https://transformerlensorg.github.io/TransformerLens/content/model_structure.html

Spec adopted: 2026-04-17 (TL 3.0 release).

Why adopt verbatim
------------------
TL 3.0's hierarchy — uniform ``hook_in`` / ``hook_out`` suffixes, explicit
Q / K / V / O sub-paths, separate ``ln1`` / ``attn`` / ``ln2`` / ``mlp``
block decomposition — is a clean conceptual design. Inventing a parallel
vocabulary would be churn for no gain and would make researcher onboarding
harder (they already know TL's names).

Independence from runtime backend
---------------------------------
Adoption here is independent of the underlying TransformerLens runtime
version. Concrete ``HookedModel`` subclasses (``HookedTransformer``,
``HookedOneHotMLP``, ``HookedEmbeddingMLP``) translate canonical names to
whatever backend they run on. If a runtime moves to TL 3.x or off TL
entirely, the canonical names in this module do not change.
"""

from __future__ import annotations

import re

# Top-level components
EMBED = "embed"
POS_EMBED = "pos_embed"
LN_FINAL = "ln_final"
UNEMBED = "unembed"
BLOCKS = "blocks"

# Block-internal components
LN1 = "ln1"
LN2 = "ln2"
ATTN = "attn"
MLP = "mlp"

# Attention sub-components
ATTN_Q = "q"
ATTN_K = "k"
ATTN_V = "v"
ATTN_O = "o"
ATTN_QKV = "qkv"

# MLP sub-components
MLP_IN = "in"
MLP_PRE = "pre"
MLP_OUT = "out"

# Hook point suffixes
HOOK_IN = "hook_in"
HOOK_OUT = "hook_out"
HOOK_PRE = "hook_pre"
HOOK_PATTERN = "hook_pattern"
HOOK_ATTN_SCORES = "hook_attn_scores"
HOOK_HIDDEN_STATES = "hook_hidden_states"

HOOK_SUFFIXES: tuple[str, ...] = (
    HOOK_IN,
    HOOK_OUT,
    HOOK_PRE,
    HOOK_PATTERN,
    HOOK_ATTN_SCORES,
    HOOK_HIDDEN_STATES,
)


def block(layer: int) -> str:
    """Path to a block, e.g. ``block(0) == "blocks.0"``."""
    return f"{BLOCKS}.{layer}"


def block_component(layer: int, component: str) -> str:
    """Path to a component within a block.

    e.g. ``block_component(0, ATTN) == "blocks.0.attn"``.
    """
    return f"{block(layer)}.{component}"


def block_attn_subcomponent(layer: int, sub: str) -> str:
    """Path to an attention sub-component.

    e.g. ``block_attn_subcomponent(0, ATTN_Q) == "blocks.0.attn.q"``.
    """
    return f"{block(layer)}.{ATTN}.{sub}"


def block_mlp_subcomponent(layer: int, sub: str) -> str:
    """Path to an MLP sub-component.

    e.g. ``block_mlp_subcomponent(0, MLP_PRE) == "blocks.0.mlp.pre"``.
    """
    return f"{block(layer)}.{MLP}.{sub}"


def hook(*parts: str | int) -> str:
    """Compose a canonical hook path from arbitrary parts.

    Example:
        ``hook(BLOCKS, 0, ATTN, ATTN_Q, HOOK_OUT) == "blocks.0.attn.q.hook_out"``
    """
    return ".".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Grammar validation and enumeration
# ---------------------------------------------------------------------------

_TOP_LEVEL = {EMBED, POS_EMBED, LN_FINAL, UNEMBED}
_BLOCK_COMPONENTS = {LN1, LN2, ATTN, MLP}
_ATTN_SUBS = {ATTN_Q, ATTN_K, ATTN_V, ATTN_O, ATTN_QKV}
_MLP_SUBS = {MLP_IN, MLP_PRE, MLP_OUT}
_HOOK_SUFFIXES_SET = set(HOOK_SUFFIXES)

_BLOCK_PATH_RE = re.compile(r"^blocks\.(\d+)((?:\.[^.]+)+)$")


def is_canonical_hook_name(name: str) -> bool:
    """Return True if ``name`` conforms to the canonical hook-name grammar.

    Accepted forms:
      - ``{top_level}.{hook_suffix}`` for top-level components
        (``embed``, ``pos_embed``, ``ln_final``, ``unembed``).
      - ``blocks.{i}.{hook_suffix}`` for block-level hooks.
      - ``blocks.{i}.{component}.{hook_suffix}`` for block sub-components
        (``ln1``, ``ln2``, ``attn``, ``mlp``).
      - ``blocks.{i}.attn.{q|k|v|o|qkv}.{hook_suffix}`` for attention sub-components.
      - ``blocks.{i}.attn.{hook_pattern|hook_attn_scores}`` for attention-only hooks.
      - ``blocks.{i}.mlp.{in|pre|out}.{hook_suffix}`` for MLP sub-components.

    This validates the *shape* of the name. Whether a particular
    architecture actually publishes a given hook is determined by that
    architecture's ``hook_names()`` at runtime.
    """
    parts = name.split(".")
    if len(parts) < 2:
        return False
    suffix = parts[-1]
    if suffix not in _HOOK_SUFFIXES_SET:
        return False

    # Top-level: {component}.{suffix}
    if len(parts) == 2 and parts[0] in _TOP_LEVEL:
        return True

    # Block paths: blocks.{i}....{suffix}
    if parts[0] != BLOCKS or len(parts) < 3:
        return False
    if not parts[1].isdigit():
        return False

    middle = parts[2:-1]

    # blocks.{i}.{suffix}
    if not middle:
        return True

    # blocks.{i}.{component}.{suffix}
    if len(middle) == 1:
        return middle[0] in _BLOCK_COMPONENTS

    # blocks.{i}.{component}.{sub}.{suffix}
    if len(middle) == 2:
        component, sub = middle
        if component == ATTN:
            return sub in _ATTN_SUBS
        if component == MLP:
            return sub in _MLP_SUBS
        return False

    return False


def enumerate_hooks(
    n_layers: int,
    *,
    has_embed: bool = True,
    has_pos_embed: bool = True,
    has_ln_final: bool = True,
    has_unembed: bool = True,
    has_attn: bool = True,
    has_mlp: bool = True,
    has_block_layernorms: bool = True,
) -> tuple[str, ...]:
    """Enumerate canonical hook paths for an architecture shape.

    Returns the standard set of ``hook_in`` / ``hook_out`` (and related)
    paths that a model with the given component flags would naturally
    publish. Subclasses may publish a subset of this list — what matters
    at runtime is what the model's ``hook_names()`` actually returns.

    Used by tests and by subclasses that want to derive their hook list
    from architecture flags rather than enumerating manually.
    """
    out: list[str] = []
    if has_embed:
        out.extend([f"{EMBED}.{HOOK_IN}", f"{EMBED}.{HOOK_OUT}"])
    if has_pos_embed:
        out.extend([f"{POS_EMBED}.{HOOK_IN}", f"{POS_EMBED}.{HOOK_OUT}"])
    for i in range(n_layers):
        bp = block(i)
        out.extend([f"{bp}.{HOOK_IN}", f"{bp}.{HOOK_OUT}"])
        if has_block_layernorms:
            out.extend(
                [
                    f"{bp}.{LN1}.{HOOK_IN}",
                    f"{bp}.{LN1}.{HOOK_OUT}",
                    f"{bp}.{LN2}.{HOOK_IN}",
                    f"{bp}.{LN2}.{HOOK_OUT}",
                ]
            )
        if has_attn:
            out.extend(
                [
                    f"{bp}.{ATTN}.{HOOK_IN}",
                    f"{bp}.{ATTN}.{HOOK_OUT}",
                    f"{bp}.{ATTN}.{HOOK_PATTERN}",
                    f"{bp}.{ATTN}.{HOOK_ATTN_SCORES}",
                ]
            )
            for sub in (ATTN_Q, ATTN_K, ATTN_V, ATTN_O):
                out.extend(
                    [
                        f"{bp}.{ATTN}.{sub}.{HOOK_IN}",
                        f"{bp}.{ATTN}.{sub}.{HOOK_OUT}",
                    ]
                )
        if has_mlp:
            out.extend(
                [
                    f"{bp}.{MLP}.{HOOK_IN}",
                    f"{bp}.{MLP}.{HOOK_OUT}",
                    f"{bp}.{MLP}.{HOOK_PRE}",
                ]
            )
    if has_ln_final:
        out.extend([f"{LN_FINAL}.{HOOK_IN}", f"{LN_FINAL}.{HOOK_OUT}"])
    if has_unembed:
        out.extend([f"{UNEMBED}.{HOOK_IN}", f"{UNEMBED}.{HOOK_OUT}"])
    return tuple(out)
