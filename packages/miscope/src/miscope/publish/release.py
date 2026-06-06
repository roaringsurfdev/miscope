"""Release publication — attach a built bundle to a ``data-*`` GitHub Release (REQ_110E).

The warehouse data is local-only (gitignored, never in CI), so a bundle is *built*
on the machine that has the data and *published* from there via ``gh``. This module
turns a built :class:`BuildResult` into a ``gh release create`` invocation: the
Parquet files and ``manifest.json`` become Release assets at stable URLs, tagged
``data-{name}-{version}``. The tag namespace is deliberately distinct from the
package's ``vX.Y.Z`` tags.

Release notes are rendered from the manifest so the Release page itself carries the
citable record — version, mint date, ``miscope`` version, and each table's row
count + content hash. ``release_argv`` is pure (returns the command) so it is
testable without spawning a process; ``create_release`` runs it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from miscope.publish.build import BuildResult
from miscope.publish.manifest import BundleManifest


def render_notes(manifest: BundleManifest) -> str:
    """Markdown Release notes from a manifest — the human-facing citable record."""
    lines = [
        manifest.description.strip() or f"Data bundle `{manifest.bundle_name}`.",
        "",
        f"- **Bundle version:** {manifest.bundle_version}",
        f"- **Minted:** {manifest.mint_date}",
        f"- **miscope version:** {manifest.miscope_version}",
    ]
    if manifest.articles:
        lines.append(f"- **Article(s):** {', '.join(manifest.articles)}")
    lines += ["", "| Table | Rows | Content hash |", "|---|---:|---|"]
    lines += [f"| `{t.name}` | {t.rows:,} | `{t.content_hash}` |" for t in manifest.tables]
    return "\n".join(lines) + "\n"


def release_argv(
    result: BuildResult, *, draft: bool = False, notes_path: Path | None = None
) -> list[str]:
    """The ``gh release create`` command publishing this bundle's assets under its tag.

    ``notes_path`` points ``gh`` at a rendered notes file (``--notes-file``); when
    omitted the notes are passed inline via ``--notes``.
    """
    manifest = result.manifest
    argv = [
        "gh",
        "release",
        "create",
        manifest.tag,
        "--title",
        f"{manifest.bundle_name} {manifest.bundle_version}",
    ]
    if notes_path is not None:
        argv += ["--notes-file", str(notes_path)]
    else:
        argv += ["--notes", render_notes(manifest)]
    if draft:
        argv.append("--draft")
    assets = [str(result.manifest_path), *(str(p) for p in result.parquet_paths)]
    return argv + assets


def create_release(result: BuildResult, *, draft: bool = False, dry_run: bool = False) -> list[str]:
    """Publish the bundle as a ``data-*`` Release via ``gh`` (returns the command run).

    Writes the rendered notes next to the assets and points ``gh`` at it. With
    ``dry_run`` the command is assembled and returned but not executed — the safe
    way to inspect exactly what would be published.
    """
    notes_path = result.out_dir / "NOTES.md"
    notes_path.write_text(render_notes(result.manifest), encoding="utf-8")
    argv = release_argv(result, draft=draft, notes_path=notes_path)
    if not dry_run:
        subprocess.run(argv, check=True)
    return argv
