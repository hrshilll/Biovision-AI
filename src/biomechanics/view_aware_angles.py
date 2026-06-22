"""
View-aware biomechanical angles for live webcam inference.

Front-facing camera views distort legacy 2D / mis-mapped 3D metrics (especially
shoulder_flexion and hip_flexion). This module detects camera orientation and
recomputes legacy angle keys using geometry appropriate to the view.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

import mediapipe as mp
import numpy as np

from src.biomechanics.angles_3d import (
    Point3D,
    angle_between_vectors,
    calculate_angle_2d,
    get_landmark_point,
    get_landmark_xy,
    joint_angle_3d,
    midpoint,
    vector,
)

mp_pose = mp.solutions.pose

_VIEW_FRONT = "front"
_VIEW_SIDE = "side"
_VIEW_OBLIQUE = "oblique"


def detect_camera_view(image_landmarks, world_landmarks=None) -> str:
    """Classify camera as front, side, or oblique from shoulder geometry."""
    if image_landmarks is None:
        return _VIEW_OBLIQUE

    l_sh = image_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = image_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    span_x = abs(l_sh.x - r_sh.x)

    if world_landmarks is not None:
        wl = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        wr = world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        depth_sep = abs(wl.z - wr.z)
        if span_x > 0.11 and depth_sep < span_x * 0.45:
            return _VIEW_FRONT
        if span_x < 0.07 or (span_x > 0 and depth_sep / span_x > 0.75):
            return _VIEW_SIDE
        return _VIEW_OBLIQUE

    if span_x > 0.11:
        return _VIEW_FRONT
    if span_x < 0.06:
        return _VIEW_SIDE
    return _VIEW_OBLIQUE


def _side_visibility(landmarks, side: str, joints: tuple[str, ...], min_vis: float) -> float:
    score = 0.0
    for joint in joints:
        lm = landmarks[mp_pose.PoseLandmark[f"{side}_{joint}"].value]
        if lm.visibility >= min_vis:
            score += lm.visibility
    return score


def _bilateral_mean_3d(
    landmarks,
    joint_triplets: tuple[tuple[str, str, str], ...],
    min_visibility: float,
) -> Optional[float]:
    """Average a 3D joint angle across left/right when both sides are visible."""
    values: list[float] = []
    for side in ("LEFT", "RIGHT"):
        for a, b, c in joint_triplets:
            try:
                pa = get_landmark_point(landmarks, f"{side}_{a}", min_visibility)
                pb = get_landmark_point(landmarks, f"{side}_{b}", min_visibility)
                pc = get_landmark_point(landmarks, f"{side}_{c}", min_visibility)
                values.append(joint_angle_3d(pa, pb, pc))
            except ValueError:
                continue
    if not values:
        return None
    return round(float(np.mean(values)), 2)


def upper_arm_deviation_image(
    image_landmarks,
    side: str,
    min_visibility: float = 0.1,
) -> Optional[float]:
    """
    Degrees the upper arm tilts away from vertical in the frontal (image) plane.
    Low values = elbows pinned; high values = arm swinging forward/out.
    """
    try:
        shoulder = get_landmark_xy(image_landmarks, f"{side}_SHOULDER", min_visibility)
        elbow = get_landmark_xy(image_landmarks, f"{side}_ELBOW", min_visibility)
    except ValueError:
        return None

    dx = elbow[0] - shoulder[0]
    dy = elbow[1] - shoulder[1]
    mag = math.hypot(dx, dy)
    if mag < 1e-5:
        return 0.0
    if dy <= 0:
        return round(min(90.0, math.degrees(math.atan2(mag, max(1e-5, abs(dy))))), 2)
    return round(min(90.0, math.degrees(math.atan2(abs(dx), dy))), 2)


def bilateral_upper_arm_deviation(
    image_landmarks,
    min_visibility: float = 0.1,
) -> Optional[float]:
    """Average upper-arm deviation for both arms (front-view shoulder stability)."""
    vals = [
        upper_arm_deviation_image(image_landmarks, side, min_visibility)
        for side in ("LEFT", "RIGHT")
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 2)


def shoulder_abduction_front(
    image_landmarks,
    side: str,
    min_visibility: float = 0.1,
) -> Optional[float]:
    """
    Frontal-plane shoulder abduction: angle of upper arm away from vertical.
    Used for lateral/front raises when facing the camera.
    """
    return upper_arm_deviation_image(image_landmarks, side, min_visibility)


def bilateral_shoulder_abduction_front(
    image_landmarks,
    min_visibility: float = 0.1,
) -> Optional[float]:
    vals = [
        shoulder_abduction_front(image_landmarks, side, min_visibility)
        for side in ("LEFT", "RIGHT")
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 2)


def torso_stability_angle(trunk_inclination: float) -> float:
    """Map trunk inclination to legacy Hip_Angle scale (180 = upright)."""
    return round(max(0.0, min(180.0, 180.0 - trunk_inclination)), 2)


def plank_body_line_angle(landmarks, min_visibility: float = 0.15) -> Optional[float]:
    """Hip angle along shoulder–hip–ankle line (180 = straight plank)."""
    try:
        l_sh = get_landmark_point(landmarks, "LEFT_SHOULDER", min_visibility)
        r_sh = get_landmark_point(landmarks, "RIGHT_SHOULDER", min_visibility)
        l_hip = get_landmark_point(landmarks, "LEFT_HIP", min_visibility)
        r_hip = get_landmark_point(landmarks, "RIGHT_HIP", min_visibility)
        l_ank = get_landmark_point(landmarks, "LEFT_ANKLE", min_visibility)
        r_ank = get_landmark_point(landmarks, "RIGHT_ANKLE", min_visibility)
        shoulder = midpoint(l_sh, r_sh)
        hip = midpoint(l_hip, r_hip)
        ankle = midpoint(l_ank, r_ank)
        return joint_angle_3d(shoulder, hip, ankle)
    except ValueError:
        return None


def elbow_flare_front(image_landmarks, min_visibility: float = 0.1) -> Optional[float]:
    """Front-view elbow flare: max upper-arm deviation across both arms."""
    vals = [
        upper_arm_deviation_image(image_landmarks, side, min_visibility)
        for side in ("LEFT", "RIGHT")
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(max(vals), 2)


def sagittal_upper_arm_angle(landmarks, side: str, min_visibility: float) -> Optional[float]:
    """Side-view upper-arm angle relative to torso (shoulder stability)."""
    try:
        s = side.upper()
        shoulder = get_landmark_point(landmarks, f"{s}_SHOULDER", min_visibility)
        elbow = get_landmark_point(landmarks, f"{s}_ELBOW", min_visibility)
        hip = get_landmark_point(landmarks, f"{s}_HIP", min_visibility)
        upper_arm = vector(shoulder, elbow)
        torso = vector(hip, shoulder)
        return round(angle_between_vectors(upper_arm, torso), 2)
    except ValueError:
        return None


def best_visible_side(landmarks, min_visibility: float) -> Optional[str]:
    """Pick LEFT or RIGHT with better visibility for arm/leg chains."""
    joints = ("SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE")
    scores = {
        side: _side_visibility(landmarks, side, joints, min_visibility)
        for side in ("LEFT", "RIGHT")
    }
    if scores["LEFT"] <= 0 and scores["RIGHT"] <= 0:
        return None
    return "LEFT" if scores["LEFT"] >= scores["RIGHT"] else "RIGHT"


def apply_view_aware_angles(
    angles: Mapping[str, float],
    world_landmarks,
    image_landmarks,
    side_used: str,
    min_visibility: float = 0.15,
) -> dict[str, float]:
    """
    Overwrite legacy angle keys with view-appropriate measurements.
    Keeps YAML thresholds and rule engines working without per-exercise hacks.
    """
    result = dict(angles)
    view = detect_camera_view(image_landmarks, world_landmarks)
    result["Camera_View"] = view

    world = world_landmarks if world_landmarks is not None else image_landmarks
    min_vis = max(0.1, min_visibility)

    trunk = result.get("Trunk_Inclination") or result.get("Trunk_Inclination_3D")

    if view in (_VIEW_FRONT, _VIEW_OBLIQUE):
        elbow = _bilateral_mean_3d(
            world,
            (("SHOULDER", "ELBOW", "WRIST"),),
            min_vis,
        )
        if elbow is not None:
            result["Elbow_Angle"] = elbow
            result["Elbow_Flexion_3D"] = elbow

        shoulder_dev = bilateral_upper_arm_deviation(image_landmarks, min_vis)
        if shoulder_dev is not None:
            result["Shoulder_Angle"] = shoulder_dev
            result["Shoulder_Flexion_3D"] = shoulder_dev

        abduction = bilateral_shoulder_abduction_front(image_landmarks, min_vis)
        if abduction is not None:
            result["Shoulder_Abduction_3D"] = abduction

        if trunk is not None:
            hip_equiv = torso_stability_angle(float(trunk))
            result["Hip_Angle"] = hip_equiv
            result["Hip_Flexion_3D"] = hip_equiv

        knee = _bilateral_mean_3d(world, (("HIP", "KNEE", "ANKLE"),), min_vis)
        if knee is not None:
            result["Knee_Angle"] = knee
            result["Knee_Flexion_3D"] = knee

        plank = plank_body_line_angle(world, min_vis)
        if plank is not None:
            result["Plank_Body_Angle"] = plank

        flare = elbow_flare_front(image_landmarks, min_vis)
        if flare is not None:
            result["Elbow_Flare_Front"] = flare

    else:
        side = side_used or best_visible_side(world, min_vis) or "RIGHT"
        sagittal = sagittal_upper_arm_angle(world, side, min_vis)
        if sagittal is not None:
            result["Shoulder_Angle"] = sagittal
            result["Shoulder_Flexion_3D"] = sagittal

        plank = plank_body_line_angle(world, min_vis)
        if plank is not None:
            result["Plank_Body_Angle"] = plank
            result["Hip_Angle"] = plank
            result["Hip_Flexion_3D"] = plank

    return result
