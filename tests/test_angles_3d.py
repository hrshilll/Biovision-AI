"""Unit tests for 3D angle calculations."""

import numpy as np

from src.biomechanics.angles_3d import (
    Point3D,
    angle_between_vectors,
    calculate_angle_2d,
    joint_angle_3d,
)


def test_right_angle_3d():
    a = Point3D(1, 0, 0)
    b = Point3D(0, 0, 0)
    c = Point3D(0, 1, 0)
    assert abs(joint_angle_3d(a, b, c) - 90.0) < 0.1


def test_straight_angle_3d():
    a = Point3D(-1, 0, 0)
    b = Point3D(0, 0, 0)
    c = Point3D(1, 0, 0)
    assert abs(joint_angle_3d(a, b, c) - 180.0) < 0.1


def test_angle_between_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert abs(angle_between_vectors(v1, v2) - 90.0) < 0.1


def test_legacy_2d_angle():
    a = [0, 0]
    b = [0, 1]
    c = [1, 1]
    angle = calculate_angle_2d(a, b, c)
    assert 80 < angle < 100
