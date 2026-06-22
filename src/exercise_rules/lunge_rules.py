"""Lunge-specific biomechanical error detection."""

from __future__ import annotations

from typing import List, Mapping, Optional

from src.exercise_rules.base import ExerciseRuleEngine, FormIssue, _get


class LungeRuleEngine(ExerciseRuleEngine):
    exercise_key = "Lunges"

    def evaluate(
        self,
        angles: Mapping[str, float],
        landmarks=None,
        history: Optional[List[Mapping[str, float]]] = None,
    ) -> List[FormIssue]:
        issues: List[FormIssue] = []

        knee = _get(angles, "Knee_Angle", "Knee_Flexion_3D", "knee_flexion")
        trunk = _get(angles, "Trunk_Inclination", "Trunk_Inclination_3D", "trunk_inclination")
        valgus = _get(angles, "Knee_ValGus_Index", default=0.0)
        hip_level = _get(angles, "Hip_Levelness", default=100.0)

        if valgus > 10:
            issues.append(
                FormIssue(
                    code="knee_tracking",
                    message="Front knee tracking issue — align knee over ankle.",
                    severity="error",
                    metric="Knee_ValGus_Index",
                    value=valgus,
                )
            )
        if knee > 120:
            issues.append(
                FormIssue(
                    code="short_step",
                    message="Short step length — increase lunge stride.",
                    severity="warning",
                    metric="Knee_Angle",
                    value=knee,
                )
            )
        if trunk > 25:
            issues.append(
                FormIssue(
                    code="trunk_instability",
                    message="Trunk instability — stay upright.",
                    severity="warning",
                    metric="Trunk_Inclination",
                    value=trunk,
                )
            )
        if hip_level < 85:
            issues.append(
                FormIssue(
                    code="hip_drop",
                    message="Pelvic drop detected — stabilize hips.",
                    severity="warning",
                    metric="Hip_Levelness",
                    value=hip_level,
                )
            )
        return issues
