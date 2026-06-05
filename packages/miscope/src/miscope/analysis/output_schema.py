"""Analyzer output schema primitives (REQ_107).

These types make an analyzer's *output* an explicit, registered declaration
rather than something that emerges implicitly from the keys its ``analyze()``
happens to return. Each :class:`OutputField` states what a field is, what
:class:`FieldKind` it is (``columnar`` vs ``tensor``), and which
:class:`Coord` keys it.

That single declaration is load-bearing three ways (REQ_107 keystone):

- **Write routing (REQ_110).** ``kind`` decides Parquet-row vs. tensor-blob:
  ``columnar`` fields flatten to a long-format Parquet row; ``tensor`` fields
  stay as ``.npz``/``.npy`` blobs referenced by a coordinate-keyed descriptor.
  The per-analyzer storage split becomes *derived*, not hand-authored.
- **Join keys (REQ_110).** ``coords`` are the columns a field is keyed by, so a
  cross-analyzer join is both possible (shared keys exist) and discoverable
  (``registry.field(name)`` reports what a field is keyed by).
- **Discoverability (REQ_107).** The registry enumerates every field with its
  kind, coords, and one-line description.

The declaration lives on :class:`~miscope.analysis.spec.AnalyzerSpec` (next to
the analyzer it describes) — never in a parallel manifest. Drift between code
and declaration is therefore impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FieldKind(str, Enum):
    """Storage nature of an output field — the REQ_110 write-routing discriminator.

    Mirrors ``DataViewField.field_type`` (``"dataframe"``/``"ndarray"``) and the
    catalog sketch's ``PayloadKind``.
    """

    COLUMNAR = "columnar"
    """Scalar or series that flattens to a Parquet row — queryable, joinable."""

    TENSOR = "tensor"
    """Dense array kept as a blob and referenced by descriptor — retrieved for
    linear algebra, not queried."""


class Coord(str, Enum):
    """Canonical coordinate vocabulary that keys output fields (REQ_107).

    A field's ``coords`` are what make it a join target instead of something each
    consumer re-derives. ``VARIANT`` is special: it expands, per family, to the
    family's declared ``domain_parameters`` columns plus the opaque composed
    ``variant_id`` handle (so cross-family joins use ``variant_id`` while
    in-family filters use the param columns, e.g. ``WHERE prime > 100``). The
    registry records that expansion from ``family.domain_parameters`` rather than
    hardcoding a key.
    """

    VARIANT = "variant"
    """Variant identity — expands to ``variant_id`` + the family's param columns."""

    EPOCH = "epoch"
    """Checkpoint epoch. Present for per-epoch and trajectory fields."""

    NEURON = "neuron"
    """MLP neuron index (``d_mlp`` axis)."""

    ROW_ID = "row_id"
    """Generic sample identifier (e.g. centroid label, PCA sample row)."""

    SITE = "site"
    """Weight matrix or activation site (e.g. ``W_in``, ``mlp_out``)."""

    GROUP = "group"
    """Group identifier; its semantics are given by the accompanying ``GROUP_TYPE``."""

    GROUP_TYPE = "group_type"
    """Discriminator for what ``GROUP`` refers to (weight_matrix, activation_site, …)."""

    FREQUENCY = "frequency"
    """Fourier frequency index."""


@dataclass(frozen=True)
class OutputField:
    """One declared output of an analyzer.

    Declarations are **logical**: ``name`` is the un-prefixed field
    (e.g. ``circularity``, ``power``), and the per-analyzer composition of
    coordinate values into the on-disk key (``mlp_out_circularity``,
    ``group_11__W_in__...``) is the long-format writer's concern (REQ_110-A),
    not part of this declaration. REQ_107 is the *contract*; it carries no
    key-composition machinery (constraint: no heavy schema language).

    Construct with the :meth:`columnar` / :meth:`tensor` helpers for brevity;
    ``coords`` and ``kind`` accept plain strings (validated against the
    canonical vocabularies).

    Attributes:
        name: Logical field name — the un-prefixed key produced by ``analyze()``.
        dtype: Element dtype as a string (e.g. ``"float32"``, ``"int64"``,
            ``"bool"``, ``"complex128"``). For ``tensor`` fields this is the
            array element dtype.
        kind: Whether the field routes to a Parquet row (``COLUMNAR``) or a
            tensor blob + descriptor (``TENSOR``).
        coords: The coordinates that key this field, from :class:`Coord`.
        description: One-line semantic description. Long-form explanation belongs
            in the analyzer docstring — the registry enumerates, it does not explain.
    """

    name: str
    dtype: str
    kind: FieldKind
    coords: tuple[Coord, ...]
    description: str

    def __post_init__(self) -> None:
        # Defensive normalization: a direct construction may pass enum members or
        # their string values; coerce both to the canonical enums (loud on a bad
        # value). The ``columnar`` / ``tensor`` helpers already coerce ``coords``.
        object.__setattr__(self, "kind", FieldKind(self.kind))
        object.__setattr__(self, "coords", tuple(Coord(c) for c in self.coords))

    @classmethod
    def columnar(
        cls, name: str, dtype: str, coords: tuple[str | Coord, ...], description: str
    ) -> OutputField:
        """A scalar/series field that flattens to a Parquet row."""
        return cls(name, dtype, FieldKind.COLUMNAR, _as_coords(coords), description)

    @classmethod
    def tensor(
        cls, name: str, dtype: str, coords: tuple[str | Coord, ...], description: str
    ) -> OutputField:
        """A dense-array field kept as a blob and referenced by descriptor."""
        return cls(name, dtype, FieldKind.TENSOR, _as_coords(coords), description)

    @property
    def coord_names(self) -> tuple[str, ...]:
        """Coordinate keys as plain strings (for tabular rendering / routing)."""
        return tuple(c.value for c in self.coords)


def _as_coords(values: tuple[str | Coord, ...]) -> tuple[Coord, ...]:
    """Coerce a tuple of coord strings/enums to canonical :class:`Coord` enums."""
    return tuple(Coord(v) for v in values)
