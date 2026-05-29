"""Pytest configuration for the miscope test suite.

Puts the tests directory on ``sys.path`` so test modules can import shared
helpers (e.g. ``_deps_fakes``) by plain module name under
``--import-mode=importlib``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
