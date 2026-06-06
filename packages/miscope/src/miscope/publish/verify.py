"""Bundle URL verification — prove a published bundle is queryable by range request (REQ_110E).

A published bundle is only useful if a reader can query it *remotely* without
downloading the whole file. This module opens a Release asset base URL through the
same :func:`miscope.query.open` bundle mode a reader would use and runs a trivial
probe per table. A success proves the two server-side preconditions the publication
workflow depends on: HTTP range requests (DuckDB ``httpfs`` fetches only the bytes a
query touches) and permissive CORS/read access on the GitHub Release CDN.

This is the Python-side half of the REQ_110E range-request/CORS check; the browser
(DuckDB-WASM) half lands in 110-F. It needs a live URL, so it is an opt-in
verification, not a CI unit test.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import miscope.query as query


@dataclass(frozen=True)
class VerifyResult:
    """Per-table row counts fetched from a remote bundle (proof the URL is queryable)."""

    base_url: str
    row_counts: dict[str, int]

    @property
    def ok(self) -> bool:
        """Whether every probed table returned (a count, even zero, is a successful fetch)."""
        return bool(self.row_counts)


def verify_bundle_url(base_url: str, tables: Iterable[str]) -> VerifyResult:
    """Open ``base_url`` in bundle mode and probe each table with a ``COUNT(*)``.

    Raises whatever DuckDB raises on a failed remote read (a missing asset, a CORS
    rejection, a server without range support), so a failure is loud and specific.
    """
    names = list(tables)
    counts: dict[str, int] = {}
    with query.open(root=base_url, tables=names) as con:
        for table in names:
            counts[table] = int(con.df(f'SELECT COUNT(*) AS n FROM "{table}"')["n"].iloc[0])
    return VerifyResult(base_url=base_url, row_counts=counts)
