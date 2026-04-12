"""Tests for application configuration (REQ_036)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from miscope.config import AppConfig, get_config


class TestAppConfig:
    """Tests for AppConfig and get_config()."""

    def test_default_paths_resolve(self):
        """Default paths should resolve to valid project directories."""
        cfg = get_config()

        assert isinstance(cfg, AppConfig)
        assert isinstance(cfg.project_root, Path)
        assert isinstance(cfg.results_dir, Path)
        assert isinstance(cfg.model_families_dir, Path)

    def test_project_root_contains_pyproject(self):
        """Project root should contain pyproject.toml."""
        cfg = get_config()
        assert (cfg.project_root / "pyproject.toml").exists()

    def test_default_results_dir(self):
        """Default results_dir should be project_root/results."""
        cfg = get_config()
        assert cfg.results_dir == cfg.project_root / "results"

    def test_default_model_families_dir(self):
        """Default model_families_dir should be project_root/model_families."""
        cfg = get_config()
        assert cfg.model_families_dir == cfg.project_root / "model_families"

    def test_env_override_results_dir(self):
        """MISCOPE_RESULTS_DIR env var overrides results directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MISCOPE_RESULTS_DIR": tmpdir}):
                cfg = get_config()
                assert cfg.results_dir == Path(tmpdir)

    def test_env_override_model_families_dir(self):
        """MISCOPE_MODEL_FAMILIES_DIR env var overrides model families directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MISCOPE_MODEL_FAMILIES_DIR": tmpdir}):
                cfg = get_config()
                assert cfg.model_families_dir == Path(tmpdir)

    def test_env_override_project_root(self):
        """MISCOPE_PROJECT_ROOT env var overrides project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MISCOPE_PROJECT_ROOT": tmpdir}):
                cfg = get_config()
                assert cfg.project_root == Path(tmpdir).resolve()

    def test_config_is_frozen(self):
        """AppConfig should be immutable."""
        cfg = get_config()
        with pytest.raises(AttributeError):
            cfg.results_dir = Path("/tmp")  # type: ignore[misc]

    def test_env_overrides_independent(self):
        """Each env var override is independent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MISCOPE_RESULTS_DIR": tmpdir}):
                cfg = get_config()
                # results_dir overridden, model_families_dir uses default
                assert cfg.results_dir == Path(tmpdir)
                assert cfg.model_families_dir == cfg.project_root / "model_families"

    def test_legacy_tdw_vars_still_work(self):
        """TDW_* legacy env vars are still accepted as fallbacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TDW_RESULTS_DIR": tmpdir}):
                cfg = get_config()
                assert cfg.results_dir == Path(tmpdir)

    def test_miscope_vars_take_priority_over_tdw(self):
        """MISCOPE_* vars take priority over TDW_* legacy vars."""
        with tempfile.TemporaryDirectory() as tdw_dir:
            with tempfile.TemporaryDirectory() as miscope_dir:
                with patch.dict(
                    os.environ,
                    {"TDW_RESULTS_DIR": tdw_dir, "MISCOPE_RESULTS_DIR": miscope_dir},
                ):
                    cfg = get_config()
                    assert cfg.results_dir == Path(miscope_dir)
