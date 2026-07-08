"""Tests for scripts/migrate_dseed.py (REQ_061; refreshed by REQ_123)."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts/ at the repo root to path so we can import migrate_dseed
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import migrate_dseed  # type: ignore[import]


@pytest.fixture
def family_dir():
    """Temporary data root + variants directory with unmigrated variant dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        variants = root / "modulo_addition_1layer" / "variants"
        variants.mkdir(parents=True)

        for prime, seed in [(113, 999), (97, 485)]:
            variant_dir = variants / f"p{prime}_seed{seed}"
            variant_dir.mkdir()
            config = {"prime": prime, "seed": seed, "data_seed": 598}
            (variant_dir / "config.json").write_text(json.dumps(config))

        yield root, variants


class TestCollectRenames:
    def test_identifies_unmigrated_dirs(self, family_dir):
        _, variants = family_dir
        renames = migrate_dseed.collect_renames(variants, default_data_seed=598)
        assert len(renames) == 2

    def test_new_names_include_dseed(self, family_dir):
        _, variants = family_dir
        renames = migrate_dseed.collect_renames(variants, default_data_seed=598)
        for _, new_path in renames:
            assert "_dseed598" in new_path.name

    def test_reads_data_seed_from_config(self, family_dir):
        _, variants = family_dir
        variant_dir = variants / "p113_seed999"
        config = {"data_seed": 42}
        (variant_dir / "config.json").write_text(json.dumps(config))

        renames = migrate_dseed.collect_renames(variants, default_data_seed=598)
        new_names = {new.name for _, new in renames}
        assert "p113_seed999_dseed42" in new_names

    def test_uses_fallback_when_no_config(self, family_dir):
        _, variants = family_dir
        (variants / "p113_seed999" / "config.json").unlink()

        renames = migrate_dseed.collect_renames(variants, default_data_seed=598)
        new_names = {new.name for _, new in renames}
        assert "p113_seed999_dseed598" in new_names


class TestIdempotency:
    def test_skips_already_migrated_dirs(self, family_dir):
        root, variants = family_dir
        migrate_dseed.run(root, "modulo_addition_1layer", default_data_seed=598, apply=True)

        renames = migrate_dseed.collect_renames(variants, default_data_seed=598)
        assert len(renames) == 0

    def test_does_not_rename_dirs_with_dseed(self, family_dir):
        _, variants = family_dir
        (variants / "p7_seed42_dseed598").mkdir()

        renames = migrate_dseed.collect_renames(variants, default_data_seed=598)
        migrated_names = {old.name for old, _ in renames}
        assert "p7_seed42_dseed598" not in migrated_names


class TestApply:
    def test_renames_directories(self, family_dir):
        root, variants = family_dir
        migrate_dseed.run(root, "modulo_addition_1layer", default_data_seed=598, apply=True)

        dirs = {d.name for d in variants.iterdir() if d.is_dir()}
        assert "p113_seed999_dseed598" in dirs
        assert "p97_seed485_dseed598" in dirs
        assert "p113_seed999" not in dirs
        assert "p97_seed485" not in dirs

    def test_preserves_contents_after_rename(self, family_dir):
        root, variants = family_dir
        migrate_dseed.run(root, "modulo_addition_1layer", default_data_seed=598, apply=True)

        config_path = variants / "p113_seed999_dseed598" / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["prime"] == 113

    def test_dry_run_does_not_rename(self, family_dir):
        root, variants = family_dir
        migrate_dseed.run(root, "modulo_addition_1layer", default_data_seed=598, apply=False)

        dirs = {d.name for d in variants.iterdir() if d.is_dir()}
        assert "p113_seed999" in dirs
        assert "p113_seed999_dseed598" not in dirs
