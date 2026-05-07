# Integration tests

Cross-cutting tests that exercise the seam between workspace members — for
example, the dashboard rendering against real miscope artifacts. Tests that
test a single workspace member's internals belong in that member's `tests/`
directory (`packages/miscope/tests/`, `apps/dashboard/tests/`).

This directory is intentionally empty as of REQ_115; it exists so that
future cross-cutting tests have a clear home.
