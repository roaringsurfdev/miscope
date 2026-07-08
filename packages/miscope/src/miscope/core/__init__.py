"""Core types, enums, and constants for miscope.

Contains only types, enums, and constants — no compute, no I/O, no heavy
dependencies. Importable from any layer (renderers, dashboard, notebooks).
"""

from miscope.core import architecture, weights
from miscope.core.pca import PCAResult
from miscope.core.svd import SVDResult

__all__ = ["PCAResult", "SVDResult", "architecture", "weights"]
