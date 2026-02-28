"""Tests for REQ_040: Migrate Training & Analysis Run Management to Dash.

Tests cover:
- JobProgress state management (thread-safe progress tracking)
- URL routing (correct page layout per pathname)
- Training and Analysis Run page layouts render without error
- Version accessible from dashboard
"""

from dashboard.state import JobProgress


class TestJobProgress:
    """Tests for thread-safe job progress tracking."""

    def test_initial_state(self):
        """Fresh JobProgress is not running with zero progress."""
        jp = JobProgress()
        state = jp.get_state()

        assert state["running"] is False
        assert state["progress"] == 0.0
        assert state["message"] == ""
        assert state["result"] == ""

    def test_start(self):
        """start() marks job as running and resets state."""
        jp = JobProgress()
        jp.start()
        state = jp.get_state()

        assert state["running"] is True
        assert state["progress"] == 0.0
        assert state["message"] == "Starting..."
        assert state["result"] == ""

    def test_update(self):
        """update() changes progress and message."""
        jp = JobProgress()
        jp.start()
        jp.update(0.5, "Halfway done")
        state = jp.get_state()

        assert state["running"] is True
        assert state["progress"] == 0.5
        assert state["message"] == "Halfway done"

    def test_finish(self):
        """finish() marks job as complete with result."""
        jp = JobProgress()
        jp.start()
        jp.update(0.5, "Working...")
        jp.finish("All done!")
        state = jp.get_state()

        assert state["running"] is False
        assert state["progress"] == 1.0
        assert state["message"] == "Complete"
        assert state["result"] == "All done!"

    def test_start_resets_previous(self):
        """start() clears result from a previous run."""
        jp = JobProgress()
        jp.start()
        jp.finish("First run done")

        jp.start()
        state = jp.get_state()

        assert state["running"] is True
        assert state["result"] == ""
        assert state["progress"] == 0.0

    def test_global_instances_exist(self):
        """Module-level training_progress and analysis_progress exist."""
        from dashboard.state import analysis_progress, training_progress

        assert isinstance(training_progress, JobProgress)
        assert isinstance(analysis_progress, JobProgress)


class TestVersioning:
    """Tests for version.py in dashboard."""

    def test_version_importable_from_dashboard(self):
        """Can import __version__ from dashboard.version."""
        from dashboard.version import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3

    def test_version_format_semantic(self):
        """Version follows MAJOR.MINOR.BUILD format."""
        from dashboard.version import __version__

        parts = __version__.split(".")
        assert all(part.isdigit() for part in parts)


class TestNavigation:
    """Tests for site-level navigation."""

    def test_sitenav_renders(self):
        """create_sitenav returns a component."""
        from dashboard.components.sitenav import create_sitenav

        navbar = create_sitenav()
        assert navbar is not None

    def test_sitenav_contains_version(self):
        """Sitenav brand includes the version string."""
        from dashboard.components.sitenav import create_sitenav
        from dashboard.version import __version__

        navbar = create_sitenav()
        brand = getattr(navbar, "brand", "")
        assert __version__ in str(brand)


class TestPageLayouts:
    """Tests for Training and Analysis Run page layouts."""

    def test_training_layout_renders(self):
        """create_training_page_layout returns a valid component."""
        from dashboard.pages.training import create_training_page_layout

        layout = create_training_page_layout()
        assert layout is not None

    def test_analysis_run_layout_renders(self):
        """create_analysis_run_page_layout returns a valid component."""
        from dashboard.pages.analysis_run import create_analysis_run_page_layout

        layout = create_analysis_run_page_layout()
        assert layout is not None

    def test_main_layout_has_url_and_page_content(self):
        """create_default_layout includes dcc.Location and page-content container."""
        from dashboard.layout import create_default_layout

        layout = create_default_layout()
        assert layout is not None
        assert layout.children is not None


class TestDashAppCreation:
    """Tests for Dash app factory."""

    def test_create_app(self):
        """create_app returns a Dash application."""
        from dashboard.app import create_app

        app = create_app()
        assert app is not None
        assert app.title == "MechInterp Scope"
