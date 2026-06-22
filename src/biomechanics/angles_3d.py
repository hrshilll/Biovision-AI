"""3D biomechanical angle calculations using vector geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


@dataclass(frozen=True)
class Point3D:
    """A 3D point in MediaPipe world coordinates (meters)."""

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_landmark(cls, landmark) -> "Point3D":
        return cls(float(landmark.x), float(landmark.y), float(landmark.z))


def vector(a: Point3D | Sequence[float], b: Point3D | Sequence[float]) -> np.ndarray:
    """Return vector from point a to point b."""
    pa = a.as_array() if isinstance(a, Point3D) else np.asarray(a, dtype=float)
    pb = b.as_array() if isinstance(b, Point3D) else np.asarray(b, dtype=float)
    return pb - pa


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a unit vector; fallback to original vector if near-zero."""
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle in degrees between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def joint_angle_3d(a: Point3D, b: Point3D, c: Point3D) -> float:
    """Return interior angle at joint b formed by segments ba and bc."""
    ba = vector(b, a)
    bc = vector(b, c)
    return round(angle_between_vectors(ba, bc), 2)


def calculate_angle_2d(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Legacy 2D angle calculation for backward compatibility."""
    pa, pb, pc = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(pc[1] - pb[1], pc[0] - pb[0]) - np.arctan2(
        pa[1] - pb[1], pa[0] - pb[0]
    )
    angle = abs(float(np.degrees(radians)))
    return round(360.0 - angle if angle > 180.0 else angle, 2)


def get_landmark_point(
    landmarks,
    name: str,
    min_visibility: float = 0.0,
    use_world: bool = False,
) -> Point3D:
    """Extract a landmark as Point3D, optionally enforcing visibility."""
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    if min_visibility > 0 and lm.visibility < min_visibility:
        raise ValueError(f"Low visibility: {name} ({lm.visibility:.2f})")
    return Point3D.from_landmark(lm)


def get_landmark_xy(
    landmarks,
    name: str,
    min_visibility: float = 0.0,
) -> list[float]:
    """Extract normalized 2D coordinates for legacy pipelines."""
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    if min_visibility > 0 and lm.visibility < min_visibility:
        raise ValueError(f"Low visibility: {name} ({lm.visibility:.2f})")
    return [float(lm.x), float(lm.y)]


def midpoint(a: Point3D, b: Point3D) -> Point3D:
    return Point3D((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)


def project_to_vertical(v: np.ndarray) -> np.ndarray:
    """Project vector onto the vertical axis (MediaPipe y-axis, inverted gravity)."""
    vertical = np.array([0.0, -1.0, 0.0], dtype=float)
    return vertical


@dataclass
class AngleSet3D:
    """Complete 3D angle set for one body side."""

    side: str
    elbow_flexion: float
    knee_flexion: float
    hip_flexion: float
    shoulder_abduction: float
    shoulder_flexion: float
    trunk_inclination: float
    spine_alignment: float
    pelvic_tilt: float

    def to_dict(self) -> dict[str, float]:
        return {
            "Elbow_Flexion_3D": self.elbow_flexion,
            "Knee_Flexion_3D": self.knee_flexion,
            "Hip_Flexion_3D": self.hip_flexion,
            "Shoulder_Abduction_3D": self.shoulder_abduction,
            "Shoulder_Flexion_3D": self.shoulder_flexion,
            "Trunk_Inclination_3D": self.trunk_inclination,
            "Spine_Alignment_3D": self.spine_alignment,
            "Pelvic_Tilt_3D": self.pelvic_tilt,
        }

    def legacy_dict(self) -> dict[str, float]:
        """Map 3D metrics to legacy column names used by existing models."""
        return {
            "Elbow_Angle": self.elbow_flexion,
            "Shoulder_Angle": self.shoulder_flexion,
            "Hip_Angle": self.hip_flexion,
            "Knee_Angle": self.knee_flexion,
        }

    def combined_dict(self) -> dict[str, float]:
        merged = self.legacy_dict()
        merged.update(self.to_dict())
        return merged


def extract_angles_3d(
    landmarks,
    side: str = "RIGHT",
    min_visibility: float = 0.30,
) -> AngleSet3D:
    """Compute full 3D angle set for the given body side."""
    s = side.upper()
    shoulder = get_landmark_point(landmarks, f"{s}_SHOULDER", min_visibility)
    elbow = get_landmark_point(landmarks, f"{s}_ELBOW", min_visibility)
    wrist = get_landmark_point(landmarks, f"{s}_WRIST", min_visibility)
    hip = get_landmark_point(landmarks, f"{s}_HIP", min_visibility)
    knee = get_landmark_point(landmarks, f"{s}_KNEE", min_visibility)
    ankle = get_landmark_point(landmarks, f"{s}_ANKLE", min_visibility)

    opposite = "LEFT" if s == "RIGHT" else "RIGHT"
    opp_shoulder = get_landmark_point(landmarks, f"{opposite}_SHOULDER", min_visibility)
    opp_hip = get_landmark_point(landmarks, f"{opposite}_HIP", min_visibility)

    pelvis_center = midpoint(hip, opp_hip)
    shoulder_center = midpoint(shoulder, opp_shoulder)

    elbow_flexion = joint_angle_3d(shoulder, elbow, wrist)
    knee_flexion = joint_angle_3d(hip, knee, ankle)
    hip_flexion = joint_angle_3d(shoulder, hip, knee)

    upper_arm = normalize(vector(shoulder, elbow))
    torso = normalize(vector(hip, shoulder))
    shoulder_abduction = round(angle_between_vectors(upper_arm, torso), 2)

    forward = normalize(vector(pelvis_center, shoulder_center))
    vertical = np.array([0.0, -1.0, 0.0], dtype=float)
    shoulder_flexion = round(angle_between_vectors(forward, vertical), 2)

    trunk_vec = vector(pelvis_center, shoulder_center)
    trunk_inclination = round(angle_between_vectors(trunk_vec, vertical), 2)

    mid_spine = midpoint(pelvis_center, shoulder_center)
    upper_spine = vector(pelvis_center, mid_spine)
    lower_spine = vector(mid_spine, shoulder_center)
    spine_alignment = round(angle_between_vectors(upper_spine, lower_spine), 2)

    hip_line = vector(opp_hip, hip)
    horizontal = np.array([1.0, 0.0, 0.0], dtype=float)
    pelvic_tilt = round(angle_between_vectors(hip_line, horizontal), 2)

    return AngleSet3D(
        side=s,
        elbow_flexion=elbow_flexion,
        knee_flexion=knee_flexion,
        hip_flexion=hip_flexion,
        shoulder_abduction=shoulder_abduction,
        shoulder_flexion=shoulder_flexion,
        trunk_inclination=trunk_inclination,
        spine_alignment=spine_alignment,
        pelvic_tilt=pelvic_tilt,
    )


def extract_legacy_angles_2d(
    landmarks,
    side: str = "RIGHT",
    min_visibility: float = 0.30,
) -> dict[str, float]:
    """Extract legacy 2D angles for backward compatibility."""
    s = side.upper()
    shoulder = get_landmark_xy(landmarks, f"{s}_SHOULDER", min_visibility)
    elbow = get_landmark_xy(landmarks, f"{s}_ELBOW", min_visibility)
    wrist = get_landmark_xy(landmarks, f"{s}_WRIST", min_visibility)
    hip = get_landmark_xy(landmarks, f"{s}_HIP", min_visibility)
    knee = get_landmark_xy(landmarks, f"{s}_KNEE", min_visibility)
    ankle = get_landmark_xy(landmarks, f"{s}_ANKLE", min_visibility)
    return {
        "Elbow_Angle": calculate_angle_2d(shoulder, elbow, wrist),
        "Shoulder_Angle": calculate_angle_2d(hip, shoulder, elbow),
        "Hip_Angle": calculate_angle_2d(shoulder, hip, knee),
        "Knee_Angle": calculate_angle_2d(hip, knee, ankle),
    }


def _side_chain_visibility(landmarks, side: str, min_visibility: float) -> float:
    """Score how visible the arm/leg chain is for one body side."""
    joints = ("SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE")
    score = 0.0
    for joint in joints:
        lm = landmarks[mp_pose.PoseLandmark[f"{side}_{joint}"].value]
        if lm.visibility >= min_visibility:
            score += lm.visibility
    return score


def best_side_angles(
    landmarks,
    min_visibility: float = 0.30,
    prefer_3d: bool = True,
) -> tuple[Optional[dict[str, float]], Optional[str]]:
    """Pick the side with the best landmark visibility, not always RIGHT first."""
    best_angles: Optional[dict[str, float]] = None
    best_side: Optional[str] = None
    best_score = -1.0

    for side in ("RIGHT", "LEFT"):
        try:
            if prefer_3d:
                angle_set = extract_angles_3d(landmarks, side, min_visibility)
                angles = angle_set.combined_dict()
            else:
                angles = extract_legacy_angles_2d(landmarks, side, min_visibility)
            score = _side_chain_visibility(landmarks, side, min_visibility)
            if score > best_score:
                best_score = score
                best_angles = angles
                best_side = side
        except ValueError:
            continue
    return best_angles, best_side


def bilateral_angles_3d(
    landmarks,
    min_visibility: float = 0.30,
) -> dict[str, float]:
    """Return left/right 3D angles for symmetry analysis."""
    output: dict[str, float] = {}
    for side in ("LEFT", "RIGHT"):
        try:
            angles = extract_angles_3d(landmarks, side, min_visibility)
            for key, value in angles.to_dict().items():
                prefix = "Left" if side == "LEFT" else "Right"
                metric = key.replace("_3D", "")
                output[f"{prefix}_{metric}"] = value
        except ValueError:
            continue
    return output
