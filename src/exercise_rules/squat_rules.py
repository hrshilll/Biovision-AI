"""Squat-specific biomechanical error detection."""

from __future__ import annotations

from typing import List, Mapping, Optional

from src.exercise_rules.base import ExerciseRuleEngine, FormIssue, _get


class SquatRuleEngine(ExerciseRuleEngine):
    exercise_key = "Squats"

    def evaluate(
        self,
        angles: Mapping[str, float],
        landmarks=None,
        history: Optional[List[Mapping[str, float]]] = None,
    ) -> List[FormIssue]:
        issues: List[FormIssue] = []

        knee = _get(angles, "Knee_Angle", "Knee_Flexion_3D", "knee_flexion")
        hip = _get(angles, "Hip_Angle", "Hip_Flexion_3D", "hip_flexion")
        trunk = _get(angles, "Trunk_Inclination", "Trunk_Inclination_3D", "trunk_inclination")
        valgus = _get(angles, "Knee_ValGus_Index", default=0.0)
        pelvic = _get(angles, "Pelvic_Tilt", "Pelvic_Tilt_3D", "pelvic_tilt")

        if valgus > 12:
            issues.append(
                FormIssue(
                    code="knee_valgus",
                    message="Knees collapsing inward.",
                    severity="error",
                    metric="Knee_ValGus_Index",
                    value=valgus,
                )
            )
        if valgus < -8 and valgus != 0:
            issues.append(
                FormIssue(
                    code="knee_varus",
                    message="Knees bowing outward excessively.",
                    severity="warning",
                    metric="Knee_ValGus_Index",
                    value=valgus,
                )
            )
        if trunk > 35:
            issues.append(
                FormIssue(
                    code="forward_lean",
                    message="Excessive forward lean detected.",
                    severity="error",
                    metric="Trunk_Inclination",
                    value=trunk,
                )
            )
        if knee > 125 and hip > 125:
            issues.append(
                FormIssue(
                    code="insufficient_depth",
                    message="Depth below recommended range.",
                    severity="warning",
                    metric="Knee_Angle",
                    value=knee,
                )
            )
        if pelvic > 25:
            issues.append(
                FormIssue(
                    code="heel_lift",
                    message="Possible heel lift or ankle mobility limitation.",
                    severity="warning",
                    metric="Pelvic_Tilt",
                    value=pelvic,
                )
            )
        return issues
