"""Velocity and acceleration primitives.

Finite-difference derivatives of trajectories along a time axis. These
primitives replace inline ``np.diff`` patterns scattered across analyzers
and renderers. Pure ``np.ndarray`` in / ``np.ndarray`` out — no knowledge
of ``Variant``, ``Epoch``, or ``Site``.

For non-uniform time spacing, normalize by the time gap downstream:

    v = compute_velocity(traj) / np.diff(times)[:, None]
"""

import numpy as np


def compute_velocity(trajectory: np.ndarray, time_axis: int = 0) -> np.ndarray:
    """First derivative via finite difference along ``time_axis``.

    Args:
        trajectory: Array with samples ordered along ``time_axis``.
        time_axis: Axis along which to take differences. Default 0.

    Returns:
        Array with one fewer entry along ``time_axis``.
        ``out[i] = trajectory[i+1] - trajectory[i]`` along the time axis.
    """
    if trajectory.shape[time_axis] < 2:
        raise ValueError(
            f"trajectory needs at least 2 samples along axis {time_axis}; "
            f"got shape {trajectory.shape}"
        )
    return np.diff(trajectory, axis=time_axis)


def compute_acceleration(trajectory: np.ndarray, time_axis: int = 0) -> np.ndarray:
    """Second derivative via twice-applied finite difference along ``time_axis``.

    Args:
        trajectory: Array with samples ordered along ``time_axis``.
        time_axis: Axis along which to take differences. Default 0.

    Returns:
        Array with two fewer entries along ``time_axis``.
        ``out[i] = trajectory[i+2] - 2*trajectory[i+1] + trajectory[i]``.
    """
    if trajectory.shape[time_axis] < 3:
        raise ValueError(
            f"trajectory needs at least 3 samples along axis {time_axis}; "
            f"got shape {trajectory.shape}"
        )
    return np.diff(trajectory, n=2, axis=time_axis)
