"""Tests for application configuration (REQ_036 / REQ_123)."""

import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from miscope.config import AppConfig, get_config


class TestAppConfig:
    """Tests for AppConfig and get_config()."""

    def test_default_paths_resolve(self):
        cfg = get_config()
        assert isinstance(cfg, AppConfig)
        assert isinstance(cfg.project_root, Path)
        assert isinstance(cfg.data_root, Path)

    def test_project_root_contains_pyproject(self):
        cfg = get_config()
        assert (cfg.project_root / "pyproject.toml").exists()

    def test_default_data_root(self):
        """Default data_root should be ``project_root / data``."""
        cfg = get_config()
        assert cfg.data_root == cfg.project_root / "data"

    def test_env_override_data_root(self):
        """MISCOPE_DATA_ROOT env var overrides the data root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MISCOPE_DATA_ROOT": tmpdir}, clear=False):
                cfg = get_config()
                assert cfg.data_root == Path(tmpdir)

    def test_env_override_project_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MISCOPE_PROJECT_ROOT": tmpdir}, clear=False):
                cfg = get_config()
                assert cfg.project_root == Path(tmpdir).resolve()

    def test_config_is_frozen(self):
        cfg = get_config()
        with pytest.raises(AttributeError):
            cfg.data_root = Path("/tmp")  # type: ignore[misc]

    def test_legacy_tdw_data_root_accepted(self):
        """TDW_DATA_ROOT is accepted as a legacy alias for MISCOPE_DATA_ROOT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"TDW_DATA_ROOT": tmpdir}
            with patch.dict(os.environ, env, clear=False):
                # Ensure MISCOPE_DATA_ROOT is unset so the legacy alias takes over.
                os.environ.pop("MISCOPE_DATA_ROOT", None)
                cfg = get_config()
                assert cfg.data_root == Path(tmpdir)

    def test_miscope_data_root_takes_priority_over_tdw(self):
        with tempfile.TemporaryDirectory() as tdw_dir:
            with tempfile.TemporaryDirectory() as miscope_dir:
                env = {"TDW_DATA_ROOT": tdw_dir, "MISCOPE_DATA_ROOT": miscope_dir}
                with patch.dict(os.environ, env, clear=False):
                    cfg = get_config()
                    assert cfg.data_root == Path(miscope_dir)

    def test_deprecated_results_dir_emits_warning(self):
        """Setting the retired MISCOPE_RESULTS_DIR alone emits a DeprecationWarning."""
        env_clean = dict(os.environ)
        env_clean.pop("MISCOPE_DATA_ROOT", None)
        env_clean.pop("TDW_DATA_ROOT", None)
        env_clean["MISCOPE_RESULTS_DIR"] = "/tmp/legacy-results"
        with patch.dict(os.environ, env_clean, clear=True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg = get_config()
                assert any(issubclass(w.category, DeprecationWarning) for w in caught)
                # Default data_root still resolves under the project root.
                assert cfg.data_root == cfg.project_root / "data"

    def test_data_root_set_silences_deprecation(self):
        """If MISCOPE_DATA_ROOT is set, deprecated vars do not trigger warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "MISCOPE_DATA_ROOT": tmpdir,
                "MISCOPE_RESULTS_DIR": "/tmp/legacy",
            }
            with patch.dict(os.environ, env, clear=False):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    cfg = get_config()
                    assert cfg.data_root == Path(tmpdir)
                    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
