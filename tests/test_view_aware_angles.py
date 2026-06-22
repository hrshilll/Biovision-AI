"""Unit tests for view-aware angle calculations."""

import math

import mediapipe as mp

from src.biomechanics.view_aware_angles import (
    apply_view_aware_angles,
    bilateral_upper_arm_deviation,
    detect_camera_view,
    torso_stability_angle,
    upper_arm_deviation_image,
)

mp_pose = mp.solutions.pose


class _FakeLm:
    def __init__(self, x, y, z=0.0, visibility=0.95):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def _make_landmarks(spec: dict) -> list:
    lms = [_FakeLm(0, 0, 0, 0.0) for _ in range(33)]
    for name, coords in spec.items():
        idx = mp_pose.PoseLandmark[name].value
        lms[idx] = _FakeLm(*coords)
    return lms


def test_detect_front_view():
    image = _make_landmarks(
        {
            "LEFT_SHOULDER": (0.35, 0.35, 0.0, 0.95),
            "RIGHT_SHOULDER": (0.65, 0.35, 0.0, 0.95),
        }
    )
    world = _make_landmarks(
        {
            "LEFT_SHOULDER": (-0.2, 0.0, 0.0, 0.95),
            "RIGHT_SHOULDER": (0.2, 0.0, 0.0, 0.95),
        }
    )
    assert detect_camera_view(image, world) == "front"


def test_upper_arm_vertical_is_low_deviation():
    image = _make_landmarks(
        {
            "LEFT_SHOULDER": (0.4, 0.3, 0.0, 0.95),
            "LEFT_ELBOW": (0.4, 0.5, 0.0, 0.95),
        }
    )
    dev = upper_arm_deviation_image(image, "LEFT")
    assert dev is not None
    assert dev < 15


def test_upper_arm_swing_is_high_deviation():
    image = _make_landmarks(
        {
            "LEFT_SHOULDER": (0.4, 0.3, 0.0, 0.95),
            "LEFT_ELBOW": (0.55, 0.45, 0.0, 0.95),
        }
    )
    dev = upper_arm_deviation_image(image, "LEFT")
    assert dev is not None
    assert dev > 25


def test_torso_stability_mapping():
    assert torso_stability_angle(10) == 170
    assert torso_stability_angle(0) == 180


def test_apply_view_aware_overwrites_shoulder_for_front():
    image = _make_landmarks(
        {
            "LEFT_SHOULDER": (0.35, 0.3, 0.0, 0.95),
            "RIGHT_SHOULDER": (0.65, 0.3, 0.0, 0.95),
            "LEFT_ELBOW": (0.35, 0.5, 0.0, 0.95),
            "RIGHT_ELBOW": (0.65, 0.5, 0.0, 0.95),
            "LEFT_WRIST": (0.35, 0.65, 0.0, 0.95),
            "RIGHT_WRIST": (0.65, 0.65, 0.0, 0.95),
            "LEFT_HIP": (0.4, 0.55, 0.0, 0.95),
            "RIGHT_HIP": (0.6, 0.55, 0.0, 0.95),
            "LEFT_KNEE": (0.4, 0.75, 0.0, 0.95),
            "RIGHT_KNEE": (0.6, 0.75, 0.0, 0.95),
            "LEFT_ANKLE": (0.4, 0.95, 0.0, 0.95),
            "RIGHT_ANKLE": (0.6, 0.95, 0.0, 0.95),
        }
    )
    world = image
    base = {
        "Shoulder_Angle": 120.0,
        "Hip_Angle": 90.0,
        "Trunk_Inclination": 8.0,
        "Elbow_Angle": 140.0,
        "Knee_Angle": 170.0,
    }
    out = apply_view_aware_angles(base, world, image, "RIGHT", 0.1)
    assert out["Camera_View"] == "front"
    assert out["Shoulder_Angle"] < 30
    assert out["Hip_Angle"] > 150
