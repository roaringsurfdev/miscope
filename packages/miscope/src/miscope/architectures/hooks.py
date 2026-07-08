"""HookPoint — the per-activation capture primitive.

A ``HookPoint`` is an identity ``nn.Module`` that subclasses route
activations through. Forward hooks registered on the point fire when the
activation passes through, giving consumers a chance to capture or
modify it.

Implementation
--------------
Aliased from ``transformer_lens.hook_points.HookPoint`` (TL 2.x) for now.
TL is in the dependency tree regardless via the runtime
:class:`~miscope.architectures.hooked_transformer.HookedTransformer`
subclass (REQ_112), so re-implementing here would only cost code without
removing the dependency. If a future REQ drops TL from the dep tree,
``HookPoint`` can be re-implemented in this module — the canonical
miscope import path stays the same.
"""

from __future__ import annotations

from transformer_lens.hook_points import HookPoint

__all__ = ["HookPoint"]
