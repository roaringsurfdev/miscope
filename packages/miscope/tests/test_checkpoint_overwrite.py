"""REQ_149: non-destructive checkpoint writes (skip-existing + overwrite_all).

The over-invalidation these tests guard against: an insert/extend run that
re-saves existing checkpoints gives every file a new mtime, which flips the
checkpoint fingerprint and forces a full corpus re-analysis. ``_save_checkpoint``
is the single enforcement point — it skips an existing file unless ``overwrite``
is set, so existing snapshots keep their bytes *and* mtime and stay fresh.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from miscope.families.discovery import discover_families


@pytest.fixture
def family(tmp_path_factory):
    """A modulo_addition_1layer family rooted in a temp data dir."""
    data_root = Path(tempfile.mkdtemp(dir=tmp_path_factory.mktemp("data")))
    family_dir = data_root / "modulo_addition_1layer"
    (family_dir / "variants").mkdir(parents=True)
    family_json = {
        "name": "modulo_addition_1layer",
        "display_name": "Modulo Addition (1 Layer)",
        "description": "Single-layer transformer for modular arithmetic",
        "architecture": {
            "n_layers": 1,
            "n_heads": 4,
            "d_model": 128,
            "d_head": 32,
            "d_mlp": 512,
            "act_fn": "relu",
            "normalization_type": None,
            "n_ctx": 3,
        },
        "domain_parameters": {
            "prime": {"type": "int", "description": "Modulus", "default": 113},
            "seed": {"type": "int", "description": "Random seed", "default": 999},
        },
        "analyzers": [],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "p{prime}_seed{seed}",
    }
    (family_dir / "family.json").write_text(json.dumps(family_json))
    return discover_families(data_root=data_root)["modulo_addition_1layer"]


@pytest.fixture
def trained_variant(family):
    """A variant trained with checkpoints at epochs 0, 25, 49."""
    variant = family.create_variant({"prime": 17, "seed": 42, "data_seed": 598})
    variant.train(num_epochs=50, checkpoint_epochs=[0, 25, 49], device="cpu")
    return variant


def _checkpoint_path(variant, epoch):
    return variant.checkpoints_dir / f"checkpoint_epoch_{epoch:05d}.safetensors"


def _stat(variant, epoch):
    """(mtime_ns, bytes) identity of a checkpoint file."""
    path = _checkpoint_path(variant, epoch)
    return path.stat().st_mtime_ns, path.read_bytes()


class TestSaveCheckpointEnforcementPoint:
    """``_save_checkpoint`` is the single skip-existing / overwrite gate."""

    def test_writes_new_checkpoint_and_reports_written(self, trained_variant):
        sd = {"w": torch.zeros(4)}
        assert trained_variant._save_checkpoint(sd, 99999) is True
        assert _checkpoint_path(trained_variant, 99999).exists()

    def test_skips_existing_without_overwrite(self, trained_variant):
        original = {"w": torch.zeros(4)}
        trained_variant._save_checkpoint(original, 99999)
        before = _checkpoint_path(trained_variant, 99999).read_bytes()

        # A second save of *different* content must be skipped, returning False,
        # and must not touch the file on disk.
        assert trained_variant._save_checkpoint({"w": torch.ones(4)}, 99999) is False
        assert _checkpoint_path(trained_variant, 99999).read_bytes() == before

    def test_overwrite_rewrites_existing(self, trained_variant):
        trained_variant._save_checkpoint({"w": torch.zeros(4)}, 99999)
        before = _checkpoint_path(trained_variant, 99999).read_bytes()

        assert trained_variant._save_checkpoint({"w": torch.ones(4)}, 99999, overwrite=True) is True
        assert _checkpoint_path(trained_variant, 99999).read_bytes() != before


class TestNonDestructiveInsert:
    """Inserting density leaves existing snapshots byte- and mtime-identical."""

    def test_existing_checkpoints_untouched_on_denser_retrain(self, trained_variant):
        existing = {e: _stat(trained_variant, e) for e in (0, 25, 49)}

        # Insert density: a denser schedule over the same horizon, default
        # overwrite_all=False.
        result = trained_variant.train(
            num_epochs=50, checkpoint_epochs=[0, 10, 25, 40, 49], device="cpu"
        )

        # Pre-existing files: bytes AND mtime unchanged (never re-opened for write).
        for epoch, (mtime, payload) in existing.items():
            assert _stat(trained_variant, epoch) == (mtime, payload), (
                f"epoch {epoch} checkpoint was rewritten"
            )

        # Only the genuinely-new epochs were written this run.
        assert sorted(result.checkpoint_epochs) == [10, 40]
        assert _checkpoint_path(trained_variant, 10).exists()
        assert _checkpoint_path(trained_variant, 40).exists()

    def test_metadata_records_full_checkpoint_set(self, trained_variant):
        trained_variant.train(num_epochs=50, checkpoint_epochs=[0, 10, 25, 40, 49], device="cpu")
        # Metadata carries the variant's full set (existing ∪ written), so the
        # loss-curve rug sees every checkpoint — not just this run's writes.
        assert trained_variant.metadata["checkpoint_epochs"] == [0, 10, 25, 40, 49]

    def test_same_schedule_writes_nothing_by_default(self, trained_variant):
        # Re-running the identical schedule is a no-op: every file exists, so every
        # write is skipped — this is the property that keeps artifacts fresh.
        result = trained_variant.train(num_epochs=50, checkpoint_epochs=[0, 25, 49], device="cpu")
        assert result.checkpoint_epochs == []

    def test_overwrite_all_rewrites_whole_schedule(self, trained_variant):
        # The explicit opt-in: every scheduled epoch is written this run.
        result = trained_variant.train(
            num_epochs=50, checkpoint_epochs=[0, 25, 49], device="cpu", overwrite_all=True
        )
        assert sorted(result.checkpoint_epochs) == [0, 25, 49]
