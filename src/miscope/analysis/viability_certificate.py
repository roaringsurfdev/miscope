"""Viability Certificate — geometric analysis of frequency sets (REQ_086).

Computes analytic metrics that characterize whether a chosen frequency set
produces viable geometry for modular arithmetic generalization.

All computation is purely analytical — no model weights required.
The only empirical input is the observed W_E participation ratio at the
effective-dimensionality crossover epoch.

Calibrated thresholds (from notebooks/viability_certificate_calibration.py):
  max_alias > 0.80  →  aliasing failure risk
  gap_pct   > 30%   →  coverage concern
  2|F| > W_E_PR     →  compression constraint is binding (rare in corpus)
"""

from __future__ import annotations

import json as _json
from itertools import combinations
from pathlib import Path as _Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Module-level ideal-set cache (per prime × size)
# Populated at import time from pre-computed JSON; falls back to exhaustive
# search for any (prime, size) pair not in the file.
# ---------------------------------------------------------------------------

_IDEAL_SETS_PATH = _Path(__file__).parent.parent.parent.parent / (
    "model_families/modulo_addition_1layer/ideal_frequency_sets.json"
)

_ideal_cache: dict[tuple[int, int], tuple[list[int], float]] = {}


def _load_ideal_sets_from_disk() -> None:
    if not _IDEAL_SETS_PATH.exists():
        return
    try:
        with open(_IDEAL_SETS_PATH) as f:
            data = _json.load(f)
        for key_str, entry in data.items():
            prime, size = (int(x) for x in key_str.split(":"))
            _ideal_cache[(prime, size)] = (entry["ideal_set"], entry["ideal_dist"])
    except Exception:
        pass


_load_ideal_sets_from_disk()

# ---------------------------------------------------------------------------
# Thresholds derived from calibration notebook
# ---------------------------------------------------------------------------

ALIAS_FAILURE_THRESHOLD = 0.80  # max aliasing risk → aliasing failure
ALIAS_WARNING_THRESHOLD = 0.72  # max aliasing risk → elevated warning
COVERAGE_GAP_THRESHOLD = 30.0  # gap-from-ideal % → coverage concern


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------


def build_centroid_matrix(prime: int, freqs: list[int]) -> np.ndarray:
    """Idealized Fourier centroid matrix, shape (prime, 2*|freqs|).

    Row r: [cos(2πkr/p), sin(2πkr/p) for k in freqs]
    """
    r = np.arange(prime, dtype=float)
    cols = []
    for k in freqs:
        theta = 2 * np.pi * k * r / prime
        cols.append(np.cos(theta))
        cols.append(np.sin(theta))
    return np.stack(cols, axis=1)


def min_pairwise_distance(C: np.ndarray) -> float:
    """Minimum L2 distance between any two rows of C (vectorised)."""
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]  # (p, p, d)
    dists = np.linalg.norm(diff, axis=2)  # (p, p)
    np.fill_diagonal(dists, np.inf)
    return float(dists.min())


def separation_profile(
    prime: int,
    freqs: list[int],
    d_model: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum pairwise centroid distance as dimensions are progressively removed.

    Embeds the 2|F|-dim centroid cloud in d_model space (zero-padded),
    decomposes via SVD, then sweeps retained dims from d_model down to 1.

    Returns (dims, min_distances) — both length d_model.
    The 'cliff' where distance drops rapidly sits at d ≈ 2|F|.
    For the calibration corpus, W_E_PR >> 2|F|, so the crossover marker
    sits in the flat (unaffected) region.
    """
    C = build_centroid_matrix(prime, freqs)
    C_centred = C - C.mean(axis=0)
    C_full = np.zeros((prime, d_model))
    C_full[:, : C_centred.shape[1]] = C_centred

    _, _, Vt = np.linalg.svd(C_full, full_matrices=False)
    # Project onto all components once; then slice for each dim
    C_proj_all = C_full @ Vt.T  # (prime, d_model)

    dims = np.arange(1, d_model + 1)
    min_dists = np.array(
        [min_pairwise_distance(C_proj_all[:, :d]) for d in dims],
        dtype=float,
    )
    return dims, min_dists


def aliasing_risk(prime: int, freqs: list[int]) -> dict[int, float]:
    """Per-frequency aliasing risk: k / ((p-1)/2).

    Risk = 1.0 at the Nyquist limit. Values > 0.80 are the calibrated
    failure threshold. High risk means small perturbations collapse
    geometric separation — even when current min distance looks fine.
    """
    nyquist = (prime - 1) / 2.0
    return {k: k / nyquist for k in freqs}


def predicted_hard_pairs(
    prime: int, freqs: list[int], n_pairs: int = 5
) -> dict[int, list[tuple[int, int]]]:
    """For each frequency k, the n_pairs hardest residue class pairs to separate.

    Pairs whose separation = floor(p/k) steps are closest in the k-th Fourier
    direction and therefore most aliasing-prone.
    """
    result: dict[int, list[tuple[int, int]]] = {}
    for k in freqs:
        period = max(1, round(prime / k))
        pairs = [(r, (r + period) % prime) for r in range(min(n_pairs, prime // 2))]
        result[k] = pairs
    return result


def find_ideal_set(prime: int, size: int) -> tuple[list[int], float]:
    """Minimum-cardinality subset of {1,…,(p-1)/2} maximising min pairwise distance.

    Results are cached per (prime, size).  Exhaustive search — tractable for
    p ≤ 127 and size ≤ 5 (worst case ~3.8M subsets for p=127, size=5).
    """
    key = (prime, size)
    if key in _ideal_cache:
        return _ideal_cache[key]

    best_set: list[int] = []
    best_dist = -1.0
    for subset in combinations(range(1, prime // 2 + 1), size):
        d = min_pairwise_distance(build_centroid_matrix(prime, list(subset)))
        if d > best_dist:
            best_dist = d
            best_set = list(subset)

    _ideal_cache[key] = (best_set, best_dist)
    return best_set, best_dist


def _pr_to_dist(pr: float, dims: np.ndarray, min_dists: np.ndarray) -> float:
    """Interpolate min_distance at a non-integer dimensionality target."""
    idx = int(np.clip(np.searchsorted(dims, pr), 0, len(dims) - 1))
    return float(min_dists[idx])


# ---------------------------------------------------------------------------
# Certificate computation
# ---------------------------------------------------------------------------


def compute_certificate(
    prime: int,
    freqs: list[int],
    W_E_PR: float,
    d_model: int = 128,
) -> dict[str, Any]:
    """Compute all viability metrics for a (prime, frequency_set, W_E_PR) triple.

    Returns a dict suitable for driving all dashboard visualizations.
    The 'regime' key gives the top-level classification:
      'viable'            — passes all thresholds
      'aliasing_failure'  — max_alias > ALIAS_FAILURE_THRESHOLD
      'coverage_concern'  — gap from ideal > COVERAGE_GAP_THRESHOLD
      'compression_risk'  — 2|F| exceeds W_E_PR (rare)
    """
    if not freqs:
        return {"error": "No frequencies provided"}

    dims, min_dists = separation_profile(prime, freqs, d_model)
    ambient_min_dist = float(min_dists[2 * len(freqs) - 1])  # at 2|F| dims
    compressed_min_dist = _pr_to_dist(W_E_PR, dims, min_dists)

    alias = aliasing_risk(prime, freqs)
    max_alias = max(alias.values())
    mean_alias = float(np.mean(list(alias.values())))

    n = len(freqs)
    ideal_set, ideal_dist = find_ideal_set(prime, n)
    gap_pct = 100.0 * (ideal_dist - ambient_min_dist) / ideal_dist if ideal_dist > 0 else 0.0

    subspace_dims = 2 * n
    compression_margin = W_E_PR - subspace_dims

    hard_pairs = predicted_hard_pairs(prime, freqs)

    if max_alias > ALIAS_FAILURE_THRESHOLD:
        regime = "aliasing_failure"
    elif compression_margin < 0:
        regime = "compression_risk"
    elif gap_pct > COVERAGE_GAP_THRESHOLD:
        regime = "coverage_concern"
    else:
        regime = "viable"

    return {
        "prime": prime,
        "freqs": freqs,
        "W_E_PR": W_E_PR,
        "d_model": d_model,
        "dims": dims,
        "min_dists": min_dists,
        "ambient_min_dist": ambient_min_dist,
        "compressed_min_dist": compressed_min_dist,
        "alias_per_freq": alias,
        "max_alias": max_alias,
        "mean_alias": mean_alias,
        "ideal_set": ideal_set,
        "ideal_dist": ideal_dist,
        "gap_pct": gap_pct,
        "subspace_dims": subspace_dims,
        "compression_margin": compression_margin,
        "hard_pairs": hard_pairs,
        "regime": regime,
    }
