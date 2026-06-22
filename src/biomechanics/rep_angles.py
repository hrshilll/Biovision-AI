"""Rep-counting angles optimized for live webcam (2D image landmarks)."""

from __future__ import annotations

from typing import Mapping, Optional

import mediapipe as mp

from src.biomechanics.angles_3d import calculate_angle_2d, get_landmark_xy

mp_pose = mp.solutions.pose

_ANGLE_JOINTS = {
    "elbow_flexion": ("SHOULDER", "ELBOW", "WRIST"),
    "knee_flexion": ("HIP", "KNEE", "ANKLE"),
    "hip_flexion": ("SHOULDER", "HIP", "KNEE"),
    "shoulder_flexion": ("HIP", "SHOULDER", "ELBOW"),
    "shoulder_abduction": ("SHOULDER", "ELBOW", "WRIST"),
}


def _side_score(landmarks, side: str, joints: tuple[str, str, str], min_visibility: float) -> float:
    s = side.upper()
    score = 0.0
    for joint in joints:
        lm = landmarks[mp_pose.PoseLandmark[f"{s}_{joint}"].value]
        if lm.visibility >= min_visibility:
            score += lm.visibility
    return score


def compute_rep_angle_2d(
    image_landmarks,
    angle_key: str,
    side_used: Optional[str] = None,
    min_visibility: float = 0.2,
) -> tuple[Optional[float], Optional[str]]:
    """Compute a 2D joint angle from image landmarks for rep counting."""
    if image_landmarks is None:
        return None, None

    joints = _ANGLE_JOINTS.get(angle_key)
    if joints is None:
        return None, None

    sides = [side_used.upper()] if side_used else ["RIGHT", "LEFT"]

    best_val: Optional[float] = None
    best_side: Optional[str] = None
    best_score = -1.0

    for side in sides:
        if side not in ("LEFT", "RIGHT"):
            continue
        score = _side_score(image_landmarks, side, joints, min_visibility)
        if score < min_visibility * len(joints):
            continue
        try:
            s = side.upper()
            a = get_landmark_xy(image_landmarks, f"{s}_{joints[0]}", min_visibility)
            b = get_landmark_xy(image_landmarks, f"{s}_{joints[1]}", min_visibility)
            c = get_landmark_xy(image_landmarks, f"{s}_{joints[2]}", min_visibility)
            val = calculate_angle_2d(a, b, c)
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_val = val
            best_side = side

    return best_val, best_side


def resolve_rep_angle(
    angles: Mapping[str, float],
    image_landmarks,
    angle_key: str,
    side_used: Optional[str],
    registry,
    min_visibility: float = 0.2,
) -> tuple[Optional[float], Optional[str]]:
    """Prefer view-corrected angle dict; fall back to live 2D computation."""
    if angles:
        val = registry.resolve_angle_value(angles, angle_key)
        if val > 0:
            return val, side_used

    val_2d, side_2d = compute_rep_angle_2d(
        image_landmarks, angle_key, side_used=side_used, min_visibility=min_visibility
    )
    if val_2d is not None and val_2d > 0:
        return val_2d, side_2d

    return None, None
