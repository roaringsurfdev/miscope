"""ActivationCache — canonical-name-keyed activation read interface.

A minimal wrapper over ``dict[str, torch.Tensor]``. Returned by
``HookedModel.run_with_cache()``. Analyzers read activations as
``cache[canonical_name]``; missing keys raise ``KeyError`` with the set
of available names attached to the message.

The cache is intentionally a read-only-style surface. It exposes no
analyzer-facing computation methods — those belong in the analysis
layer, not on the cache. Adding bracket-key sugar like
``cache["post", 0, "mlp"]`` is deferred to REQ_112 if migration of the
canary analyzer surfaces a real ergonomic gap.

This module has no imports from the underlying TransformerLens
library. The grep quarantine test (REQ_112) depends on this.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

import torch


class ActivationCache:
    """Architecture-agnostic activation cache keyed by canonical hook names."""

    def __init__(self, data: Mapping[str, torch.Tensor] | None = None) -> None:
        self._data: dict[str, torch.Tensor] = dict(data) if data else {}

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            return self._data[name]
        except KeyError:
            raise KeyError(
                f"Canonical hook name {name!r} is not in this cache. "
                f"Available: {sorted(self._data)}"
            ) from None

    def __setitem__(self, name: str, value: torch.Tensor) -> None:
        self._data[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ActivationCache(keys={sorted(self._data)})"

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[torch.Tensor]:
        return self._data.values()

    def items(self) -> ItemsView[str, torch.Tensor]:
        return self._data.items()

    def get(
        self,
        name: str,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        return self._data.get(name, default)
