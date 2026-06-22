"""Posture and alignment metrics derived from 3D pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from src.biomechanics.angles_3d import (
    Point3D,
    angle_between_vectors,
    bilateral_angles_3d,
    extract_angles_3d,
    get_landmark_point,
    joint_angle_3d,
    midpoint,
    normalize,
    vector,
)


@dataclass
class PostureMetrics:
    """Posture-related biomechanical metrics for a single frame."""

    trunk_inclination: float
    spine_alignment: float
    pelvic_tilt: float
    shoulder_levelness: float
    hip_levelness: float
    knee_valgus_index: float
    forward_head_angle: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Trunk_Inclination": self.trunk_inclination,
            "Spine_Alignment": self.spine_alignment,
            "Pelvic_Tilt": self.pelvic_tilt,
            "Shoulder_Levelness": self.shoulder_levelness,
            "Hip_Levelness": self.hip_levelness,
            "Knee_ValGus_Index": self.knee_valgus_index,
            "Forward_Head_Angle": self.forward_head_angle,
        }


def compute_posture_metrics(
    landmarks,
    min_visibility: float = 0.30,
) -> Optional[PostureMetrics]:
    """Compute frame-level posture metrics from pose landmarks."""
    try:
        left_shoulder = get_landmark_point(landmarks, "LEFT_SHOULDER", min_visibility)
        right_shoulder = get_landmark_point(landmarks, "RIGHT_SHOULDER", min_visibility)
        left_hip = get_landmark_point(landmarks, "LEFT_HIP", min_visibility)
        right_hip = get_landmark_point(landmarks, "RIGHT_HIP", min_visibility)
        nose = get_landmark_point(landmarks, "NOSE", min_visibility)
        left_knee = get_landmark_point(landmarks, "LEFT_KNEE", min_visibility)
        right_knee = get_landmark_point(landmarks, "RIGHT_KNEE", min_visibility)
        left_ankle = get_landmark_point(landmarks, "LEFT_ANKLE", min_visibility)
        right_ankle = get_landmark_point(landmarks, "RIGHT_ANKLE", min_visibility)
    except ValueError:
        return None

    pelvis = midpoint(left_hip, right_hip)
    shoulders = midpoint(left_shoulder, right_shoulder)

    vertical = np.array([0.0, -1.0, 0.0], dtype=float)
    trunk_vec = vector(pelvis, shoulders)
    trunk_inclination = round(angle_between_vectors(trunk_vec, vertical), 2)

    mid_spine = midpoint(pelvis, shoulders)
    upper_spine = vector(pelvis, mid_spine)
    lower_spine = vector(mid_spine, shoulders)
    spine_alignment = round(angle_between_vectors(upper_spine, lower_spine), 2)

    hip_line = vector(left_hip, right_hip)
    horizontal = np.array([1.0, 0.0, 0.0], dtype=float)
    pelvic_tilt = round(angle_between_vectors(hip_line, horizontal), 2)

    shoulder_delta_y = abs(left_shoulder.y - right_shoulder.y)
    shoulder_levelness = round(max(0.0, 100.0 - shoulder_delta_y * 500.0), 1)

    hip_delta_y = abs(left_hip.y - right_hip.y)
    hip_levelness = round(max(0.0, 100.0 - hip_delta_y * 500.0), 1)

    left_knee_track = joint_angle_3d(left_hip, left_knee, left_ankle)
    right_knee_track = joint_angle_3d(right_hip, right_knee, right_ankle)
    knee_valgus_index = round(abs(left_knee_track - right_knee_track), 2)

    head_vec = vector(shoulders, nose)
    forward_head_angle = round(angle_between_vectors(head_vec, vertical), 2)

    return PostureMetrics(
        trunk_inclination=trunk_inclination,
        spine_alignment=spine_alignment,
        pelvic_tilt=pelvic_tilt,
        shoulder_levelness=shoulder_levelness,
        hip_levelness=hip_levelness,
        knee_valgus_index=knee_valgus_index,
        forward_head_angle=forward_head_angle,
    )


def posture_score_from_metrics(metrics: PostureMetrics) -> float:
    """Convert posture metrics into a 0-100 score."""
    penalties = [
        min(metrics.trunk_inclination * 0.8, 40.0),
        min(abs(180.0 - metrics.spine_alignment) * 1.5, 30.0),
        min(metrics.pelvic_tilt * 0.5, 20.0),
        min(metrics.forward_head_angle * 0.4, 20.0),
        min((100.0 - metrics.shoulder_levelness) * 0.3, 15.0),
        min((100.0 - metrics.hip_levelness) * 0.3, 15.0),
    ]
    return round(max(0.0, 100.0 - float(np.mean(penalties))), 1)


def merge_posture_into_angles(
    landmarks,
    angles: Mapping[str, float],
    min_visibility: float = 0.30,
) -> Dict[str, float]:
    """Augment angle dictionary with posture metrics and bilateral data."""
    merged = dict(angles)
    posture = compute_posture_metrics(landmarks, min_visibility)
    if posture:
        merged.update(posture.to_dict())
    merged.update(bilateral_angles_3d(landmarks, min_visibility))
    return merged
