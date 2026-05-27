"""REQ_126: Canonical type for the family basis-projection contract.

A ``BasisProjectionSite`` describes one named site where a family-supplied
composer produces a matrix ready for projection onto the family's basis.
The ``weight_basis_projection`` and (future) ``activation_basis_projection``
analyzers iterate over a family's declared sites and apply REQ_109's basis
primitives to each composer's output.

This honors the architectural rule from ``PROJECT.md``: families are context
providers; analyzers are universal instruments. The analyzer does not embed
any per-site composition logic — site composition is the family's
responsibility.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np


class BasisProjectionSite(NamedTuple):
    """A named site where a family supplies a matrix for basis projection.

    Attributes:
        name: Stable identifier for the site (e.g., ``"embedding"``,
            ``"mlp_in"``, ``"attn_qk"``). Used as the per-site key prefix
            in the analyzer's output artifact.
        compose: Pure function ``(source, context) -> np.ndarray`` returning
            the composed matrix for this site. ``source`` is the
            parameter_snapshot artifact dict (weight side) or activation
            cache (activation side); ``context`` is the family's analysis
            context (carries ``params``, etc.). The returned array must
            place period-length axes at the positions named in
            ``period_axes``.
        period_axes: Indices of axes in the composed matrix that carry the
            family basis's period. A tuple of length 1 = 1D projection per
            non-period unit; length 2 = 2D projection per non-period unit.
        description: One-line human-readable description of what the site
            represents (e.g., ``"W_E embedding excluding equals token"``).
    """

    name: str
    compose: Callable[[Any, dict[str, Any]], np.ndarray]
    period_axes: tuple[int, ...]
    description: str
