"""Pre-compute ideal frequency sets for all corpus (prime, size) combinations.

Saves results to model_families/modulo_addition_1layer/ideal_frequency_sets.json.
This file is loaded at startup by viability_certificate.py to avoid repeating
exhaustive search after every app restart.

Run once from the project root:
    python scripts/precompute_ideal_sets.py

Timing (approximate, depends on hardware):
    sizes 2-4:  < 30 seconds total
    size 5:     ~2-3 minutes for p=101, p=103; ~5-10 min for p=127
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_PATH = Path("model_families/modulo_addition_1layer/ideal_frequency_sets.json")

# Primes in the corpus
CORPUS_PRIMES = [59, 89, 97, 101, 103, 107, 109, 113, 127]

# Observed max frequency set size per prime (from variant_registry.json).
# We pre-compute up to max_size + 1 to cover the "what if we added one more?" question.
MAX_SIZE_BY_PRIME = {
    59: 3,
    89: 4,
    97: 4,
    101: 5,
    103: 5,
    107: 4,
    109: 4,
    113: 4,
    127: 5,
}


def build_centroid_matrix(prime: int, freqs: list[int]) -> np.ndarray:
    r = np.arange(prime, dtype=float)
    cols = []
    for k in freqs:
        theta = 2 * np.pi * k * r / prime
        cols.append(np.cos(theta))
        cols.append(np.sin(theta))
    return np.stack(cols, axis=1)


def min_pairwise_distance(C: np.ndarray) -> float:
    """Use scipy.pdist for speed in tight search loops."""
    return float(pdist(C).min())


def find_ideal_set(prime: int, size: int) -> tuple[list[int], float]:
    candidates = list(range(1, prime // 2 + 1))
    best_set: list[int] = []
    best_dist = -1.0
    for subset in combinations(candidates, size):
        C = build_centroid_matrix(prime, list(subset))
        d = min_pairwise_distance(C)
        if d > best_dist:
            best_dist = d
            best_set = list(subset)
    return best_set, best_dist


def main() -> None:
    # Load existing results to allow incremental runs
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            results: dict = json.load(f)
        print(f"Loaded {len(results)} existing entries from {OUTPUT_PATH}")
    else:
        results = {}

    for prime in CORPUS_PRIMES:
        max_size = MAX_SIZE_BY_PRIME[prime]
        for size in range(2, max_size + 1):
            key = f"{prime}:{size}"
            if key in results:
                entry = results[key]
                print(f"  p={prime} size={size}  [cached] ideal={entry['ideal_set']} dist={entry['ideal_dist']:.4f}")
                continue

            n_subsets = 1
            n = prime // 2
            for i in range(size):
                n_subsets = n_subsets * (n - i) // (i + 1)
            print(f"  p={prime} size={size}  searching {n_subsets:,} subsets...", end=" ", flush=True)
            t0 = time.time()
            ideal_set, ideal_dist = find_ideal_set(prime, size)
            elapsed = time.time() - t0
            print(f"done in {elapsed:.1f}s  →  ideal={ideal_set}  dist={ideal_dist:.4f}")

            results[key] = {"ideal_set": ideal_set, "ideal_dist": ideal_dist}

            # Save incrementally so a long run is recoverable
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
