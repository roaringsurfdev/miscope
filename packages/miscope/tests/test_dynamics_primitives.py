"""Unit tests for velocity / acceleration primitives (REQ_109 phase 1a)."""

import numpy as np
import pytest

from miscope.analysis.library.dynamics import compute_acceleration, compute_velocity


class TestComputeVelocity:
    def test_constant_velocity_for_linear_trajectory(self):
        # Linear: position = a + b*t. Velocity is constant b per step.
        t = np.arange(5)
        traj = np.column_stack([2.0 + 3.0 * t, 1.0 - 0.5 * t])
        v = compute_velocity(traj)
        assert v.shape == (4, 2)
        np.testing.assert_allclose(v, np.tile([3.0, -0.5], (4, 1)))

    def test_one_dim_input(self):
        traj = np.array([0.0, 1.0, 4.0, 9.0])  # squares
        v = compute_velocity(traj)
        np.testing.assert_allclose(v, [1.0, 3.0, 5.0])

    def test_time_axis_parameter(self):
        # Time on axis 1
        traj = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [10.0, 9.0, 8.0, 7.0],
            ]
        )
        v = compute_velocity(traj, time_axis=1)
        assert v.shape == (2, 3)
        np.testing.assert_allclose(v[0], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(v[1], [-1.0, -1.0, -1.0])

    def test_rejects_singleton_trajectory(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_velocity(np.array([[1.0, 2.0]]))


class TestComputeAcceleration:
    def test_zero_for_linear_trajectory(self):
        t = np.arange(5)
        traj = np.column_stack([2.0 + 3.0 * t, 1.0 - 0.5 * t])
        a = compute_acceleration(traj)
        assert a.shape == (3, 2)
        np.testing.assert_allclose(a, np.zeros((3, 2)), atol=1e-12)

    def test_constant_for_quadratic_trajectory(self):
        # position = t^2 → first diff is (1, 3, 5, 7), second diff is (2, 2, 2)
        t = np.arange(5)
        traj = (t**2).astype(float)
        a = compute_acceleration(traj)
        np.testing.assert_allclose(a, [2.0, 2.0, 2.0])

    def test_time_axis_parameter(self):
        # Quadratic along axis 1
        traj = np.array([[(i**2) for i in range(5)], [(2 * i**2) for i in range(5)]]).astype(float)
        a = compute_acceleration(traj, time_axis=1)
        assert a.shape == (2, 3)
        np.testing.assert_allclose(a[0], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(a[1], [4.0, 4.0, 4.0])

    def test_rejects_short_trajectory(self):
        with pytest.raises(ValueError, match="at least 3"):
            compute_acceleration(np.array([[1.0], [2.0]]))
