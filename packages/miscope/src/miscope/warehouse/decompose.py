"""Key decomposition: npz key -> (logical field, prefix-coord values) — REQ_110A.

REQ_107 declares *logical* output fields (un-prefixed names + keying coords). On
disk, an analyzer composes coordinate values into the npz key
(``mlp_out_radii``, ``group_8__W_in__n_components``). This module owns the
inverse map — the decomposition REQ_107 deliberately carries no machinery for.
The rules are the hard-won ones captured in
``docs/notes/req110a_decomposition_map.md``.

A *prefix coord* (``site`` and/or ``group``) is carried in the npz key; every
other coord (``neuron``, ``row_id``, ``frequency``, and — for cross-epoch
analyzers — ``epoch``) is an *array axis*, flattened by :mod:`.flatten`. Whether
``site``/``group`` is prefix or axis is per-analyzer, so this module assigns each
analyzer one :class:`KeyStyle` plus the coords its key carries.

Parsing is done *relative to the known field name* (declared in the Spec), so the
site/group vocabulary is not needed to split: strip the style's separator and
whatever remains is the coord token. Each npz key is assigned to exactly one
field — the longest-name match — so a shorter field name can never steal a key
that belongs to a more specific one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import Enum

from miscope.analysis.output_schema import Coord, OutputField


class KeyStyle(str, Enum):
    """How an analyzer composes coord tokens with the logical field name."""

    FLAT = "flat"
    """Key is the field name verbatim (no prefix coords)."""

    PREFIX_US = "prefix_us"
    """``{token}_{name}`` — single-underscore prefix (token may contain ``_``)."""

    PREFIX_DUS = "prefix_dus"
    """``{token}__{name}`` — double-underscore prefix."""

    REVERSED_US = "reversed_us"
    """``{name}_{token}`` — field first, then the coord token (e.g. ``sv_W_E``)."""

    DMD = "dmd"
    """``{group}__{site}__{name}`` — two tokens, gated on a ``group_`` first part."""


@dataclass(frozen=True)
class AnalyzerDecomp:
    """Per-analyzer decomposition rule.

    Attributes:
        style: How the npz key composes the field name with prefix-coord tokens.
        key_coords: Coords carried in the key (subset of ``{SITE, GROUP}``). A
            field uses only the subset of these that appear in its own coords;
            the rest of its coords are array axes.
        axis_labels: Maps an array-axis coord to the field whose values label that
            axis (e.g. ``GROUP -> "group_freqs"``). Absent axes label positionally.
    """

    style: KeyStyle = KeyStyle.FLAT
    key_coords: tuple[Coord, ...] = ()
    axis_labels: dict[Coord, str] = dc_field(default_factory=dict)


# Per-analyzer rules. Anything not listed defaults to FLAT with positional axes
# (correct for the flat, self-labelling analyzers). Drawn verbatim from the
# decomposition map note.
_DECOMP: dict[str, AnalyzerDecomp] = {
    # --- per-epoch, site in the key ---
    "activation_frequency_norm": AnalyzerDecomp(KeyStyle.PREFIX_US, (Coord.SITE,)),
    "repr_geometry": AnalyzerDecomp(KeyStyle.PREFIX_US, (Coord.SITE,)),
    "centroid_fourier_alignment": AnalyzerDecomp(KeyStyle.PREFIX_US, (Coord.SITE,)),
    "weight_basis_projection": AnalyzerDecomp(KeyStyle.PREFIX_US, (Coord.SITE,)),
    "full_ov_circuit": AnalyzerDecomp(KeyStyle.PREFIX_US, (Coord.SITE,)),
    "weight_spectra": AnalyzerDecomp(KeyStyle.REVERSED_US, (Coord.SITE,)),
    # --- cross-epoch, site in the key (double underscore) ---
    "global_centroid_pca": AnalyzerDecomp(KeyStyle.PREFIX_DUS, (Coord.SITE,)),
    "activation_dmd": AnalyzerDecomp(KeyStyle.PREFIX_DUS, (Coord.SITE,)),
    # --- cross-epoch, group in the key (double underscore) ---
    "parameter_trajectory": AnalyzerDecomp(KeyStyle.PREFIX_DUS, (Coord.GROUP,)),
    # --- cross-epoch, site in the key (single underscore, site = Win/Wout) ---
    "freq_group_weight_geometry": AnalyzerDecomp(
        KeyStyle.PREFIX_US, (Coord.SITE,), {Coord.GROUP: "group_freqs"}
    ),
    # --- cross-epoch, group + site both in the key (DMD units) ---
    "parameter_dmd": AnalyzerDecomp(
        KeyStyle.DMD, (Coord.GROUP, Coord.SITE), {Coord.GROUP: "populated_groups"}
    ),
    # --- flat analyzers with data-dependent axis labels ---
    "intragroup_manifold": AnalyzerDecomp(axis_labels={Coord.GROUP: "group_freqs"}),
    "neuron_group_pca": AnalyzerDecomp(axis_labels={Coord.GROUP: "group_freqs"}),
    "transient_frequency": AnalyzerDecomp(axis_labels={Coord.FREQUENCY: "ever_qualified_freqs"}),
    # --- the messiest: field-first, token is site OR pair-as-group ---
    "gradient_site": AnalyzerDecomp(
        KeyStyle.REVERSED_US,
        (Coord.SITE, Coord.GROUP),
        {Coord.FREQUENCY: "key_frequencies"},
    ),
}

_DEFAULT = AnalyzerDecomp()


def get_decomp(analyzer_name: str) -> AnalyzerDecomp:
    """The decomposition rule for an analyzer (FLAT default)."""
    return _DECOMP.get(analyzer_name, _DEFAULT)


@dataclass(frozen=True)
class KeyMatch:
    """One on-disk npz key resolved to a field and its prefix-coord values.

    ``prefix_coords`` carries only the tokens the field *declares* as coords; a
    ``loose`` match stripped a real on-disk prefix the field does not key by
    (a field written per-site though declared site-independent, e.g.
    ``frequencies``) — :mod:`.flatten` ignores the stripped token and the writer
    dedups the now-identical replicated rows.
    """

    npz_key: str
    field: OutputField
    prefix_coords: dict[str, str]  # Coord.value -> token (only field-declared coords)
    loose: bool = False


def assign_keys(
    analyzer_name: str,
    fields: tuple[OutputField, ...],
    npz_keys: tuple[str, ...],
) -> list[KeyMatch]:
    """Assign each npz key to its declared field, parsing prefix-coord tokens.

    ``fields`` must be *all* declared fields (columnar and tensor) so a key that
    belongs to a tensor field is not mis-assigned to a columnar one. Strict
    (coord-backed) matches win; a loose pass then rescues keys that carry an
    on-disk prefix the field does not declare. Keys matching no field are dropped.
    """
    decomp = get_decomp(analyzer_name)
    matches: list[KeyMatch] = []
    unmatched: list[str] = []
    for key in npz_keys:
        best = _best_field(fields, lambda f: _try_match(decomp, f, key))
        if best is not None:
            matches.append(KeyMatch(npz_key=key, field=best[1], prefix_coords=best[2]))
        else:
            unmatched.append(key)

    for key in unmatched:
        best = _best_field(fields, lambda f: _try_match_loose(decomp, f, key))
        if best is not None:
            matches.append(KeyMatch(npz_key=key, field=best[1], prefix_coords=best[2], loose=True))
    return matches


def _best_field(
    fields: tuple[OutputField, ...],
    parse: Callable[[OutputField], dict[str, str] | None],
) -> tuple[int, OutputField, dict[str, str]] | None:
    """Pick the longest-name field whose ``parse`` succeeds (most specific wins)."""
    best: tuple[int, OutputField, dict[str, str]] | None = None
    for f in fields:
        parsed = parse(f)
        if parsed is None:
            continue
        score = len(f.name)
        if best is None or score > best[0]:
            best = (score, f, parsed)
    return best


def _try_match_loose(decomp: AnalyzerDecomp, field: OutputField, key: str) -> dict[str, str] | None:
    """Strip the analyzer's prefix affix even when the field doesn't key by it.

    Rescues fields replicated on disk under a coord they're declared independent
    of. Returns ``{}`` (no field-declared coord) so flatten emits the declared
    shape and the writer dedups the replicas.
    """
    if not decomp.key_coords or decomp.style is KeyStyle.DMD:
        return None
    name = field.name
    sep = "__" if decomp.style is KeyStyle.PREFIX_DUS else "_"
    if decomp.style is KeyStyle.REVERSED_US:
        return {} if key.startswith(f"{name}{sep}") else None
    return {} if key.endswith(f"{sep}{name}") and key != name else None


def _try_match(decomp: AnalyzerDecomp, field: OutputField, key: str) -> dict[str, str] | None:
    """Return parsed prefix coords if ``key`` decomposes to ``field``, else None."""
    field_key_coords = tuple(c for c in decomp.key_coords if c in field.coords)
    name = field.name

    # A DMD unit requires BOTH group and site in the key; a field carrying only
    # one of them (e.g. ``populated_groups`` keyed by group alone) is flat on disk.
    if decomp.style is KeyStyle.DMD and not (
        Coord.GROUP in field_key_coords and Coord.SITE in field_key_coords
    ):
        field_key_coords = ()

    # No prefix coords for this field -> the key is the bare name.
    if not field_key_coords:
        return {} if key == name else None

    if decomp.style is KeyStyle.PREFIX_US:
        return _single_token(key, name, sep="_", suffix=True, coord=field_key_coords[0])
    if decomp.style is KeyStyle.PREFIX_DUS:
        return _single_token(key, name, sep="__", suffix=True, coord=field_key_coords[0])
    if decomp.style is KeyStyle.REVERSED_US:
        return _single_token(key, name, sep="_", suffix=False, coord=field_key_coords[0])
    if decomp.style is KeyStyle.DMD:
        return _match_dmd(key, name, field_key_coords)
    return None


def _single_token(
    key: str, name: str, *, sep: str, suffix: bool, coord: Coord
) -> dict[str, str] | None:
    """Strip one coord token from a ``{token}{sep}{name}`` / ``{name}{sep}{token}`` key."""
    affix = f"{sep}{name}" if suffix else f"{name}{sep}"
    if suffix:
        if not key.endswith(affix):
            return None
        token = key[: -len(affix)]
    else:
        if not key.startswith(affix):
            return None
        token = key[len(affix) :]
    return {coord.value: token} if token else None


def _match_dmd(key: str, name: str, field_key_coords: tuple[Coord, ...]) -> dict[str, str] | None:
    """Parse ``{group}__{site}__{name}`` (group + site both required in coords)."""
    if Coord.GROUP not in field_key_coords or Coord.SITE not in field_key_coords:
        return None
    parts = key.split("__")
    if len(parts) < 3 or not parts[0].startswith("group_"):
        return None
    if "__".join(parts[2:]) != name:
        return None
    return {Coord.GROUP.value: parts[0], Coord.SITE.value: parts[1]}
