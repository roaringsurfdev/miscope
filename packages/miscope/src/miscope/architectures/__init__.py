"""miscope's HookedModel boundary.

This subpackage contains the platform's architecture-agnostic interface
for trained models:

- :class:`~miscope.architectures.hooked_model.HookedModel` — the abstract
  base class concrete architecture subclasses extend.
- :class:`~miscope.architectures.activation_cache.ActivationCache` — the
  canonical-name-keyed activation cache returned by
  ``run_with_cache()``.
- :class:`~miscope.architectures.hooks.HookPoint` — the building block
  for registering capture points within a model.

Concrete subclasses (``HookedTransformer``, ``HookedOneHotMLP``,
``HookedEmbeddingMLP``) land under REQ_112 / REQ_113.
"""

from miscope.architectures.activation_cache import ActivationCache
from miscope.architectures.hooked_embedding_mlp import (
    HookedEmbeddingMLP,
    HookedEmbeddingMLPConfig,
)
from miscope.architectures.hooked_model import HookedModel
from miscope.architectures.hooked_one_hot_mlp import (
    HookedOneHotMLP,
    HookedOneHotMLPConfig,
)
from miscope.architectures.hooked_transformer import HookedTransformer, HookedTransformerConfig
from miscope.architectures.hooks import HookPoint

__all__ = [
    "ActivationCache",
    "HookedEmbeddingMLP",
    "HookedEmbeddingMLPConfig",
    "HookedModel",
    "HookedOneHotMLP",
    "HookedOneHotMLPConfig",
    "HookedTransformer",
    "HookedTransformerConfig",
    "HookPoint",
]
