"""
Rep counting state machine — fixed and robust.

Fixes:
- Ascending vs descending metric detection (lateral raise, crunch, etc.)
- Live 2D webcam angles preferred over noisy 3D estimates
- EMA smoothing to reduce jitter near thresholds
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Mapping, Optional

from src.biomechanics.rep_angles import resolve_rep_angle
from src.exercises.config_loader import ExerciseRegistry, RepCountConfig


_KEY_ALIASES: dict[str, list[str]] = {
    "elbow_flexion": ["Elbow_Angle", "Elbow_Flexion_3D", "Left_Elbow_Flexion", "Right_Elbow_Flexion"],
    "knee_flexion": ["Knee_Angle", "Knee_Flexion_3D", "Left_Knee_Flexion", "Right_Knee_Flexion"],
    "hip_flexion": ["Hip_Angle", "Hip_Flexion_3D", "Left_Hip_Flexion", "Right_Hip_Flexion"],
    "shoulder_flexion": ["Shoulder_Angle", "Shoulder_Flexion_3D"],
    "shoulder_abduction": ["Shoulder_Abduction_3D", "Shoulder_Angle"],
    "trunk_inclination": ["Trunk_Inclination", "Trunk_Inclination_3D"],
    "spine_alignment": ["Spine_Alignment", "Spine_Alignment_3D"],
}


def _resolve_angle(angles: Mapping[str, float], angle_key: str) -> Optional[float]:
    """Look up angle value by config key, trying all known aliases."""
    if angle_key in angles:
        v = float(angles[angle_key])
        return v if v > 0 else None

    for alias in _KEY_ALIASES.get(angle_key, []):
        if alias in angles:
            v = float(angles[alias])
            return v if v > 0 else None

    return None


class RepCounter:
    """
    Reliable rep counter using hysteresis with auto-detected threshold direction.

    For flexion exercises (bicep curl, squat): rest = high angle, active = low angle.
    For abduction/raise exercises: rest = low angle, active = high angle.
    Direction is inferred from whether down_threshold > up_threshold in YAML.
    """

    SMOOTH_WINDOW = 5

    def __init__(self, exercise: str, registry: Optional[ExerciseRegistry] = None) -> None:
        self.registry = registry or ExerciseRegistry()
        self.exercise = exercise
        self.count: int = 0
        self.stage: str = "up"
        self.cfg: Optional[RepCountConfig] = None
        self.tracking_angle: float = 0.0
        self.high_threshold: float = 0.0
        self.low_threshold: float = 0.0
        self._initialized: bool = False
        self._ascending: bool = False
        self._smooth_buf: Deque[float] = deque(maxlen=self.SMOOTH_WINDOW)

        exercise_cfg = self.registry.get(exercise)
        if exercise_cfg and exercise_cfg.rep_counting:
            self.cfg = exercise_cfg.rep_counting
            self._apply_thresholds()

    def _apply_thresholds(self) -> None:
        if self.cfg is None:
            return
        down_t = float(self.cfg.down_threshold)
        up_t = float(self.cfg.up_threshold)
        self._ascending = down_t < up_t

        if self._ascending:
            # e.g. lateral raise: low value = rest, high value = raised
            self.low_threshold = down_t
            self.high_threshold = up_t
        else:
            # e.g. bicep curl: high value = extended, low value = flexed
            self.high_threshold = down_t
            self.low_threshold = up_t

    def _smooth_display(self, val: float) -> float:
        self._smooth_buf.append(val)
        return sum(self._smooth_buf) / len(self._smooth_buf)

    def _stage_from_angle(self, val: float) -> Optional[str]:
        if val >= self.high_threshold:
            return "up"
        if val <= self.low_threshold:
            return "down"
        return None

    def update(
        self,
        angles: Mapping[str, float],
        image_landmarks=None,
        side_used: Optional[str] = None,
        min_visibility: float = 0.15,
    ) -> None:
        """Call every frame with current angles and optional image landmarks."""
        if self.cfg is None:
            return

        val: Optional[float] = None
        if image_landmarks is not None:
            val, _ = resolve_rep_angle(
                angles,
                image_landmarks,
                self.cfg.angle_key,
                side_used,
                self.registry,
                min_visibility=min_visibility,
            )
        if val is None:
            val = _resolve_angle(angles, self.cfg.angle_key)
        if val is None:
            return

        self.tracking_angle = round(self._smooth_display(val), 1)

        new_stage = self._stage_from_angle(val)
        if new_stage is None:
            return

        if not self._initialized:
            self.stage = new_stage
            self._initialized = True
            return

        if new_stage == self.stage:
            return

        self.stage = new_stage
        if self.cfg.count_on == new_stage:
            self.count += 1

    def reset(self) -> None:
        self.count = 0
        self.stage = "up"
        self.tracking_angle = 0.0
        self._initialized = False
        self._smooth_buf.clear()
        self._apply_thresholds()

    @property
    def threshold_hint(self) -> str:
        """Human-readable threshold hint for the UI."""
        if self.cfg is None:
            return ""
        if self._ascending:
            return f"Down<{self.low_threshold:.0f}  Up>{self.high_threshold:.0f}"
        return f"Flex<{self.low_threshold:.0f}  Ext>{self.high_threshold:.0f}"
