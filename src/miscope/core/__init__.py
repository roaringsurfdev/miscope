"""Core types, enums, and constants for miscope.

Contains only types, enums, and constants — no compute, no I/O, no heavy
dependencies. Importable from any layer (renderers, dashboard, notebooks).
"""

from miscope.core.pca import PCAResult

__all__ = ["PCAResult"]
