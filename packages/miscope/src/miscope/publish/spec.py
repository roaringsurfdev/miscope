"""Bundle spec — the declarative description of *what* a publication bundle contains (REQ_110E).

A spec is a small TOML file co-located with the article it backs
(``apps/fieldnotes/bundles/{name}.toml``). It names the source family, the bundle
version, the referencing article(s), and the curated set of tables — each either a
whole warehouse view or a subsetting SQL query. The spec is the *intent*; the build
script turns it into frozen Parquet + a manifest. Keeping the selection declarative
(not a pile of CLI flags) makes a bundle's contents reviewable in a diff and keeps
per-article curation — the rejection of kitchen-sink bundles — explicit.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BundleTable:
    """One table to publish: a warehouse view name, optionally subset by a query.

    ``name`` is both the source view (default) and the published file stem
    (``{name}.parquet``), so a curated table can carry a publication-facing name
    while ``query`` pulls/filters/joins whatever warehouse views it needs.
    """

    name: str
    query: str | None = None

    def sql(self) -> str:
        """The SELECT that materializes this table (defaults to the whole view)."""
        return self.query or f'SELECT * FROM "{self.name}"'


@dataclass(frozen=True)
class BundleSpec:
    """A publication bundle's declarative contents (loaded from a TOML spec file)."""

    name: str
    version: str
    family: str
    description: str
    articles: tuple[str, ...]
    tables: tuple[BundleTable, ...]

    @property
    def tag(self) -> str:
        """The ``data-*`` Release tag this bundle publishes under."""
        return f"data-{self.name}-{self.version}"

    @classmethod
    def from_dict(cls, data: dict) -> BundleSpec:
        tables = tuple(
            BundleTable(name=t["name"], query=t.get("query")) for t in data.get("tables", ())
        )
        if not tables:
            raise ValueError("A bundle spec must declare at least one table.")
        return cls(
            name=data["name"],
            version=data["version"],
            family=data["family"],
            description=data.get("description", ""),
            articles=tuple(data.get("articles", ())),
            tables=tables,
        )

    @classmethod
    def from_toml(cls, path: Path | str) -> BundleSpec:
        """Load a bundle spec from a TOML file."""
        return cls.from_dict(tomllib.loads(Path(path).read_text(encoding="utf-8")))
