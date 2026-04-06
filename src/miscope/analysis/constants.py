"""Platform-level constants for analysis infrastructure.

These are internal implementation constants — not domain parameters and not
user-configurable. They govern numerical algorithms where determinism is
required for regression testing and reproducible artifacts.

If a value here ever needs to become user-facing (e.g., because it turns out
to affect a scientifically meaningful result), move it to config.py and give
it an environment variable override.
"""

# Fixes sklearn's randomized SVD (used by PCA when the data matrix is large).
# Without a fixed seed, parameter_trajectory artifacts are non-deterministic
# across runs, breaking regression checksums.
SVD_RANDOM_STATE = 42
