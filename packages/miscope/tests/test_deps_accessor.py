"""Tests for REQ_128: the scoped lazy dependency accessor (DepsAccessor).

Covers the four verbs, scope enforcement, layout-mismatch detection, the
required-``fields`` contract, and npz-key field validation. Also exercises the
``fields=`` selective-load additions to ``ArtifactLoader``.
"""

import os
import tempfile

import numpy as np
import pytest

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.deps import (
    ALL,
    ArtifactLayoutError,
    DepsAccessor,
    UndeclaredDependencyError,
)

PER_EPOCH = "upstream_pe"
CROSS_EPOCH = "upstream_ce"
UNDECLARED = "not_declared"
EPOCHS = [0, 100, 200]


@pytest.fixture
def store():
    """A temp artifact store: one per-epoch upstream, one cross-epoch upstream."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pe_dir = os.path.join(tmpdir, PER_EPOCH)
        os.makedirs(pe_dir)
        for i, epoch in enumerate(EPOCHS):
            np.savez_compressed(
                os.path.join(pe_dir, f"epoch_{epoch:05d}.npz"),
                data=np.full(4, float(i), dtype=np.float32),
                extra=np.full(2, float(i) + 0.5, dtype=np.float32),
            )

        ce_dir = os.path.join(tmpdir, CROSS_EPOCH)
        os.makedirs(ce_dir)
        np.savez_compressed(
            os.path.join(ce_dir, "cross_epoch.npz"),
            trajectory=np.arange(6, dtype=np.float32).reshape(3, 2),
            meta=np.array([42.0], dtype=np.float32),
        )

        yield tmpdir


@pytest.fixture
def deps(store):
    loader = ArtifactLoader(store)
    return DepsAccessor(loader, frozenset({PER_EPOCH, CROSS_EPOCH}))


class TestVerbs:
    def test_load_epoch_selective(self, deps):
        d = deps.load_epoch(PER_EPOCH, 100, fields=["data"])
        assert set(d) == {"data"}
        np.testing.assert_array_equal(d["data"], np.full(4, 1.0, dtype=np.float32))

    def test_load_epoch_all(self, deps):
        d = deps.load_epoch(PER_EPOCH, 0, fields=ALL)
        assert set(d) == {"data", "extra"}

    def test_stream_yields_in_order(self, deps):
        seen = [(epoch, d["data"][0]) for epoch, d in deps.stream(PER_EPOCH, fields=["data"])]
        assert [e for e, _ in seen] == EPOCHS
        assert [v for _, v in seen] == [0.0, 1.0, 2.0]

    def test_stream_is_lazy_one_resident(self, deps):
        """Validation happens eagerly; epochs load one at a time on iteration."""
        loaded: list[int] = []
        original = deps._loader.load_epoch

        def spy(name, epoch, fields=None):
            loaded.append(epoch)
            return original(name, epoch, fields=fields)

        deps._loader.load_epoch = spy  # type: ignore[method-assign]
        it = deps.stream(PER_EPOCH, fields=["data"])
        assert loaded == []  # nothing loaded at call time
        next(it)
        assert loaded == [EPOCHS[0]]  # only the first epoch is resident
        next(it)
        assert loaded == EPOCHS[:2]

    def test_load_stack(self, deps):
        d = deps.load_stack(PER_EPOCH, fields=["data"])
        np.testing.assert_array_equal(d["epochs"], EPOCHS)
        assert d["data"].shape == (3, 4)
        assert "extra" not in d

    def test_load_cross_epoch_selective(self, deps):
        d = deps.load_cross_epoch(CROSS_EPOCH, fields=["trajectory"])
        assert set(d) == {"trajectory"}
        assert d["trajectory"].shape == (3, 2)


class TestScopeEnforcement:
    def test_load_epoch_undeclared_raises(self, deps):
        with pytest.raises(UndeclaredDependencyError, match=UNDECLARED):
            deps.load_epoch(UNDECLARED, 0, fields=ALL)

    def test_stream_undeclared_raises_at_call_time(self, deps):
        # stream validates eagerly — the error surfaces on the call, not on iteration.
        with pytest.raises(UndeclaredDependencyError):
            deps.stream(UNDECLARED, fields=ALL)

    def test_load_stack_undeclared_raises(self, deps):
        with pytest.raises(UndeclaredDependencyError):
            deps.load_stack(UNDECLARED, fields=ALL)

    def test_load_cross_epoch_undeclared_raises(self, deps):
        with pytest.raises(UndeclaredDependencyError):
            deps.load_cross_epoch(UNDECLARED, fields=ALL)


class TestLayoutMismatch:
    def test_per_epoch_verb_on_cross_epoch_upstream_raises(self, deps):
        with pytest.raises(ArtifactLayoutError, match="load_cross_epoch"):
            deps.load_stack(CROSS_EPOCH, fields=ALL)

    def test_stream_on_cross_epoch_upstream_raises(self, deps):
        with pytest.raises(ArtifactLayoutError):
            deps.stream(CROSS_EPOCH, fields=ALL)

    def test_load_cross_epoch_on_per_epoch_upstream_raises(self, deps):
        with pytest.raises(ArtifactLayoutError, match="stream"):
            deps.load_cross_epoch(PER_EPOCH, fields=ALL)


class TestFieldsContract:
    def test_fields_none_rejected(self, deps):
        with pytest.raises(ValueError, match="fields is required"):
            deps.load_epoch(PER_EPOCH, 0, fields=None)

    def test_fields_bare_string_rejected(self, deps):
        with pytest.raises(ValueError, match="not the string"):
            deps.load_epoch(PER_EPOCH, 0, fields="data")

    def test_unknown_field_raises_naming_artifact_and_field(self, deps):
        with pytest.raises(ValueError, match="bogus") as exc:
            deps.load_epoch(PER_EPOCH, 0, fields=["bogus"])
        assert PER_EPOCH in str(exc.value)

    def test_unknown_field_load_stack(self, deps):
        with pytest.raises(ValueError, match="bogus"):
            deps.load_stack(PER_EPOCH, fields=["bogus"])

    def test_unknown_field_cross_epoch(self, deps):
        with pytest.raises(ValueError, match="bogus"):
            deps.load_cross_epoch(CROSS_EPOCH, fields=["bogus"])


class TestArtifactLoaderFields:
    """The fields= additions on ArtifactLoader (used under the accessor)."""

    def test_load_epoch_fields_selective(self, store):
        loader = ArtifactLoader(store)
        d = loader.load_epoch(PER_EPOCH, 0, fields=["extra"])
        assert set(d) == {"extra"}

    def test_load_epoch_fields_unknown_raises(self, store):
        loader = ArtifactLoader(store)
        with pytest.raises(ValueError, match="bogus"):
            loader.load_epoch(PER_EPOCH, 0, fields=["bogus"])

    def test_load_cross_epoch_fields_selective(self, store):
        loader = ArtifactLoader(store)
        d = loader.load_cross_epoch(CROSS_EPOCH, fields=["meta"])
        assert set(d) == {"meta"}

    def test_load_epochs_unknown_field_raises_early(self, store):
        loader = ArtifactLoader(store)
        with pytest.raises(ValueError, match="bogus"):
            loader.load_epochs(PER_EPOCH, fields=["bogus"])
