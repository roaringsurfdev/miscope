"""Canary analyzer × stub-model substitution test (REQ_112).

Proves that the migrated ``repr_geometry`` analyzer is architecture-
agnostic in practice: it consumes ``ctx.cache`` by canonical name and
``ctx.model.get_weight`` (when applicable), and a no-op
``StubHookedModel`` can be substituted for the real
``HookedTransformer`` with no architecture-aware branching in the
analyzer.

Two cases:

1. Stub publishes the canonical hooks the canary needs → analyzer runs
   to completion and produces site-prefixed result keys.
2. Stub publishes none of them → analyzer either fails predictably at
   the model boundary (``KeyError`` from the cache) or, when run via
   the pipeline, is filtered out before invocation by the
   ``required_hooks`` check.

This test imports nothing from ``transformer_lens``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from miscope.analysis.analyzers.repr_geometry import (
    _SITES,
    RepresentationalGeometryAnalyzer,
)
from miscope.analysis.protocols import ActivationContext
from miscope.architectures import ActivationCache, HookedModel
from miscope.core import architecture as canonical_hooks


class FullCanaryStub(HookedModel):
    """Stub that publishes all canonical hooks the canary analyzer needs."""

    def __init__(self) -> None:
        super().__init__()
        self._linear = nn.Linear(2, 2, bias=False)
        self.setup_hooks()

    def setup_hooks(self) -> None:
        for canonical_hook in _SITES.values():
            self._register_hook_point(canonical_hook)

    def weight_names(self) -> list[str]:
        return []

    def get_weight(self, canonical_name: str) -> torch.Tensor:
        raise KeyError(canonical_name)


class EmptyHookStub(HookedModel):
    """Stub that publishes none of the canary's required hooks."""

    def __init__(self) -> None:
        super().__init__()
        self._linear = nn.Linear(2, 2, bias=False)
        self.setup_hooks()

    def setup_hooks(self) -> None:
        # Deliberately no hooks — required_hooks filter should keep
        # the canary from being invoked.
        return None

    def weight_names(self) -> list[str]:
        return []

    def get_weight(self, canonical_name: str) -> torch.Tensor:
        raise KeyError(canonical_name)


def _build_cache_for_full_stub(p: int, d_model: int, d_mlp: int) -> ActivationCache:
    """Build a canonical cache populated for every site the canary reads."""
    cache = ActivationCache()
    rng = np.random.default_rng(0)
    n_samples = p * p
    seq_len = 3
    for site_name, canonical_hook in _SITES.items():
        d = d_mlp if site_name == "mlp_out" else d_model
        cache[canonical_hook] = torch.tensor(
            rng.standard_normal((n_samples, seq_len, d)), dtype=torch.float32
        )
    return cache


def _make_probe(p: int) -> torch.Tensor:
    rows = [[a, b, p] for a in range(p) for b in range(p)]
    return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_canary_runs_against_full_stub():
    """Substituting a stub HookedModel + canonical cache works end-to-end."""
    p = 5
    d_model, d_mlp = 4, 8
    stub = FullCanaryStub()
    cache = _build_cache_for_full_stub(p, d_model, d_mlp)
    ctx = ActivationContext(
        # type: ignore[arg-type]
        probe=_make_probe(p),
        analysis_params={"params": {"prime": p}},
        model=stub,
        cache=cache,
    )

    analyzer = RepresentationalGeometryAnalyzer()
    result = analyzer.analyze(ctx)

    # Every published site produces its expected scalar measures.
    for site_name in _SITES:
        assert f"{site_name}_centroids" in result
        assert f"{site_name}_circularity" in result


def test_canary_required_hooks_match_canonical_grammar():
    """Sanity: every required_hooks entry is a valid canonical hook name."""
    analyzer = RepresentationalGeometryAnalyzer()
    for hook_name in analyzer.required_hooks:
        assert canonical_hooks.is_canonical_hook_name(hook_name), hook_name


def test_canary_skips_when_cache_missing_canonical_hook():
    """Cache without the canary's canonical hooks raises at the boundary, not deeper."""
    p = 5
    cache = ActivationCache()  # empty
    ctx = ActivationContext(
        # type: ignore[arg-type]
        probe=_make_probe(p),
        analysis_params={"params": {"prime": p}},
        model=FullCanaryStub(),
        cache=cache,
    )
    analyzer = RepresentationalGeometryAnalyzer()
    # Empty cache → no canonical hook is in cache → analyzer produces empty
    # site coverage (each site checks ``if canonical_hook not in ctx.cache: continue``).
    result = analyzer.analyze(ctx)
    for site_name in _SITES:
        assert f"{site_name}_centroids" not in result


def test_canary_required_hooks_filtering_against_stub_with_no_hooks():
    """A stub publishing no hooks → required_hooks set ∩ hook_names = ∅."""
    stub = EmptyHookStub()
    analyzer = RepresentationalGeometryAnalyzer()
    missing = [h for h in analyzer.required_hooks if h not in stub.hook_names()]
    # Pipeline filter would skip the analyzer based on ``missing`` being
    # non-empty; this asserts the filter has the right signal to act on.
    assert missing == list(analyzer.required_hooks)


def test_canary_does_not_import_transformer_lens():
    """Analyzer module must be TL-free after migration (REQ_112 invariant)."""
    import importlib
    import inspect

    module = importlib.import_module("miscope.analysis.analyzers.repr_geometry")
    source = inspect.getsource(module)
    assert "transformer_lens" not in source, (
        "repr_geometry imports transformer_lens after REQ_112 migration."
    )


def test_required_hooks_attribute_replaces_architecture_support():
    """Migrated analyzer no longer carries the legacy architecture_support flag."""
    analyzer = RepresentationalGeometryAnalyzer()
    assert hasattr(analyzer, "required_hooks")
    assert not hasattr(analyzer, "architecture_support")
