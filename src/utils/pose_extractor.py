"""MediaPipe pose extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import mediapipe as mp

from src.biomechanics.angles_3d import best_side_angles
from src.biomechanics.posture_metrics import merge_posture_into_angles
from src.biomechanics.view_aware_angles import apply_view_aware_angles, detect_camera_view

mp_pose = mp.solutions.pose


@dataclass
class PoseFrame:
    """Processed pose result for one video frame."""

    landmarks: Any
    world_landmarks: Any
    angles: dict[str, float]
    side_used: str
    image_landmarks: Any
    camera_view: str = "oblique"


class PoseProcessor:
    """Wrapper around MediaPipe Pose with 3D-aware angle extraction."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        min_visibility: float = 0.30,
        use_3d: bool = True,
        include_posture: bool = True,
    ) -> None:
        self.min_visibility = min_visibility
        self.use_3d = use_3d
        self.include_posture = include_posture
        self._pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def __enter__(self) -> "PoseProcessor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._pose.close()

    def process_frame(self, bgr_frame) -> Optional[PoseFrame]:
        """Run pose estimation on a BGR image and return enriched angles."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_landmarks:
            return None

        image_lms = results.pose_landmarks.landmark
        lms = (
            results.pose_world_landmarks.landmark
            if results.pose_world_landmarks
            else image_lms
        )

        angles, side = best_side_angles(lms, self.min_visibility, prefer_3d=self.use_3d)
        if angles is None or side is None:
            angles, side = best_side_angles(lms, max(0.12, self.min_visibility * 0.5), prefer_3d=False)
        if angles is None or side is None:
            # Still return landmarks so the UI can draw the skeleton overlay.
            return PoseFrame(
                landmarks=lms,
                world_landmarks=(
                    results.pose_world_landmarks.landmark
                    if results.pose_world_landmarks
                    else None
                ),
                angles={},
                side_used="RIGHT",
                image_landmarks=image_lms,
            )

        if self.include_posture:
            angles = merge_posture_into_angles(lms, angles, self.min_visibility)

        world_lms = (
            results.pose_world_landmarks.landmark
            if results.pose_world_landmarks
            else None
        )
        camera_view = detect_camera_view(image_lms, world_lms)
        angles = apply_view_aware_angles(
            angles,
            world_lms or lms,
            image_lms,
            side,
            min_visibility=max(0.1, self.min_visibility * 0.5),
        )

        return PoseFrame(
            landmarks=lms,
            world_landmarks=world_lms,
            angles=angles,
            side_used=side,
            image_landmarks=image_lms,
            camera_view=camera_view,
        )
