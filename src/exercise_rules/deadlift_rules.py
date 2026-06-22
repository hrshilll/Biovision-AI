"""Deadlift-specific biomechanical error detection."""

from __future__ import annotations

from typing import List, Mapping, Optional

from src.exercise_rules.base import ExerciseRuleEngine, FormIssue, _get


class DeadliftRuleEngine(ExerciseRuleEngine):
    exercise_key = "Deadlifts"

    def evaluate(
        self,
        angles: Mapping[str, float],
        landmarks=None,
        history: Optional[List[Mapping[str, float]]] = None,
    ) -> List[FormIssue]:
        issues: List[FormIssue] = []

        spine = _get(angles, "Spine_Alignment", "Spine_Alignment_3D", "spine_alignment")
        trunk = _get(angles, "Trunk_Inclination", "Trunk_Inclination_3D", "trunk_inclination")
        hip = _get(angles, "Hip_Angle", "Hip_Flexion_3D", "hip_flexion")
        knee = _get(angles, "Knee_Angle", "Knee_Flexion_3D", "knee_flexion")

        if spine < 155:
            issues.append(
                FormIssue(
                    code="rounded_back",
                    message="Rounded lower back detected.",
                    severity="error",
                    metric="Spine_Alignment",
                    value=spine,
                )
            )
        if history and len(history) >= 5:
            recent_hips = [_get(h, "Hip_Angle", "Hip_Flexion_3D") for h in history[-5:]]
            recent_knees = [_get(h, "Knee_Angle", "Knee_Flexion_3D") for h in history[-5:]]
            hip_rise = recent_hips[-1] - recent_hips[0]
            knee_change = recent_knees[-1] - recent_knees[0]
            if hip_rise > 8 and abs(knee_change) < 3:
                issues.append(
                    FormIssue(
                        code="early_hip_rise",
                        message="Early hip rise — hips shooting up before bar leaves floor.",
                        severity="error",
                        metric="Hip_Angle",
                        value=hip,
                    )
                )
        if trunk < 5 and hip < 170 and knee > 165:
            issues.append(
                FormIssue(
                    code="hyperextension_lockout",
                    message="Hyperextension at lockout — avoid over-arching.",
                    severity="warning",
                    metric="Trunk_Inclination",
                    value=trunk,
                )
            )
        return issues
