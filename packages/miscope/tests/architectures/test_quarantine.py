"""TransformerLens quarantine smoke test (REQ_112).

Asserts that TransformerLens imports outside the canonical-name surface
(``architectures/hooked_transformer.py`` and ``architectures/hooks.py``,
which aliases ``HookPoint``) do not grow beyond a known allow-list.

The allow-list is the set of files that imported TransformerLens
**before** REQ_112 landed. Each entry is scope owned by REQ_114 (analyzer
migration) or REQ_113 (MLP families). When those REQs migrate a file
off TransformerLens, its entry is removed from this list.

The strict version of this test (zero entries in the allow-list, hits
only in ``hooked_transformer.py``) becomes load-bearing under REQ_114.
For REQ_112, we enforce *no growth* — adding a TL import to a
not-allowed file is a regression.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
SRC = PACKAGE_ROOT / "src" / "miscope"

# Modules that are allowed to import from ``transformer_lens`` through
# the end of REQ_112. Each entry has an owning REQ that retires it.
#
# Path components are POSIX-style (forward slashes) and relative to
# ``src/miscope/``. The set is verified to match exactly — adding *or*
# removing a file silently is a regression that must be reflected here.
LEGACY_TL_IMPORTERS: frozenset[str] = frozenset(
    # REQ_114 cleared every entry on this list. The set is intentionally
    # empty; new entries imply a regression. Outside the canonical
    # surface, no module under ``src/miscope/`` should import from
    # ``transformer_lens``.
)

# Modules that are *expected* to import from ``transformer_lens`` —
# they constitute the canonical-name surface and exist by design.
CANONICAL_TL_IMPORTERS: frozenset[str] = frozenset(
    {
        "architectures/hooked_transformer.py",  # the quarantine module
        "architectures/hooks.py",  # HookPoint alias (REQ_105 carve-out)
    }
)


def _scan_tl_importers() -> set[str]:
    """Return the set of POSIX-relative paths under src/miscope that import TL.

    Uses ``grep -l`` to locate any file whose source contains the
    ``transformer_lens`` token outside of pure docstring prose. We refine
    the grep result by re-reading each match and confirming there is at
    least one *import* line referencing the package — docstring-only
    references (allowed for explanatory comments) are filtered out.
    """
    result = subprocess.run(
        [
            "grep",
            "-rln",
            "--include=*.py",
            "transformer_lens",
            str(SRC),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    candidates = [Path(line) for line in result.stdout.splitlines() if line]

    real_importers: set[str] = set()
    import_re = re.compile(
        r"^\s*(?:from\s+transformer_lens|import\s+transformer_lens)\b",
        re.MULTILINE,
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        if import_re.search(text):
            rel = path.relative_to(SRC).as_posix()
            real_importers.add(rel)
    return real_importers


def test_no_unexpected_tl_imports():
    """Every TL importer must be either canonical or in the legacy allow-list."""
    importers = _scan_tl_importers()
    expected = CANONICAL_TL_IMPORTERS | LEGACY_TL_IMPORTERS
    unexpected = importers - expected
    assert not unexpected, (
        f"New TransformerLens import(s) found in: {sorted(unexpected)}.\n"
        f"REQ_112 prohibits adding TL imports outside "
        f"src/miscope/architectures/hooked_transformer.py. If a legacy "
        f"file genuinely needs TL access (under REQ_113 or REQ_114), "
        f"add it to LEGACY_TL_IMPORTERS in this test with the owning "
        f"REQ as a comment."
    )


def test_legacy_allow_list_does_not_drift_silently():
    """Files in the allow-list must still exist and still import TL.

    If a file in ``LEGACY_TL_IMPORTERS`` no longer imports TL, the
    REQ_113 / REQ_114 migration that owns it has shipped — update the
    allow-list to reflect the win.
    """
    importers = _scan_tl_importers()
    stale = LEGACY_TL_IMPORTERS - importers
    assert not stale, (
        f"Legacy allow-list contains stale entries: {sorted(stale)}.\n"
        f"These files no longer import transformer_lens — remove them "
        f"from LEGACY_TL_IMPORTERS to lock in the migration."
    )


def test_canonical_modules_actually_import_tl():
    """Sanity check: the canonical-name surface really is the only entry point."""
    importers = _scan_tl_importers()
    assert CANONICAL_TL_IMPORTERS <= importers, (
        f"Canonical TL importers missing from src/miscope: "
        f"{sorted(CANONICAL_TL_IMPORTERS - importers)}. The boundary "
        f"module(s) are how miscope reaches TransformerLens."
    )


def test_hooked_model_and_cache_are_tl_free():
    """The architecture-agnostic core must not import TL — REQ_105 invariant."""
    forbidden = (
        SRC / "architectures" / "hooked_model.py",
        SRC / "architectures" / "activation_cache.py",
    )
    import_re = re.compile(
        r"^\s*(?:from\s+transformer_lens|import\s+transformer_lens)\b",
        re.MULTILINE,
    )
    for path in forbidden:
        assert path.exists(), f"{path} should exist"
        text = path.read_text(encoding="utf-8")
        assert not import_re.search(text), (
            f"{path.relative_to(PROJECT_ROOT)} imports transformer_lens. "
            f"The HookedModel base class and ActivationCache must remain "
            f"TL-free (REQ_105 invariant)."
        )
