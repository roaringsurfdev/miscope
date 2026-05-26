"""Guards the public API surface of ``miscope.analysis`` (REQ_125).

``ArtifactLoader`` is an internal storage primitive: it is intentionally not
re-exported from ``miscope.analysis``. Consumers reach a configured loader
through ``variant.artifacts``. These tests fail loudly if a future refactor
accidentally re-exposes the class through the public surface.
"""

from __future__ import annotations

import miscope.analysis as analysis_pkg


def test_artifact_loader_not_in_all() -> None:
    """``ArtifactLoader`` is not listed in ``miscope.analysis.__all__``."""
    assert "ArtifactLoader" not in analysis_pkg.__all__


def test_artifact_loader_not_attribute_on_package() -> None:
    """``ArtifactLoader`` is not bound as an attribute of ``miscope.analysis``.

    The class still lives at ``miscope.analysis.artifact_loader.ArtifactLoader``
    for internal use; this test ensures it is not re-exported at the package
    level, which would signal it as part of the public API.
    """
    assert not hasattr(analysis_pkg, "ArtifactLoader")


def test_artifact_loader_still_importable_via_full_path() -> None:
    """Sanity check: the class remains available at its canonical internal path.

    REQ_125 changed only the public-surface signal; the class itself, its
    methods, and its on-disk contract are unchanged.
    """
    from miscope.analysis.artifact_loader import ArtifactLoader

    assert ArtifactLoader is not None
