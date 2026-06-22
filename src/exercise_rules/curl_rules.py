"""Bicep curl-specific biomechanical error detection."""

from __future__ import annotations

from typing import List, Mapping, Optional

from src.exercise_rules.base import ExerciseRuleEngine, FormIssue, _get


class CurlRuleEngine(ExerciseRuleEngine):
    exercise_key = "Bicep_curls"

    def evaluate(
        self,
        angles: Mapping[str, float],
        landmarks=None,
        history: Optional[List[Mapping[str, float]]] = None,
    ) -> List[FormIssue]:
        issues: List[FormIssue] = []
        view = angles.get("Camera_View", "oblique")

        elbow = _get(angles, "Elbow_Angle", "Elbow_Flexion_3D", "elbow_flexion")
        shoulder = _get(angles, "Shoulder_Angle", "Shoulder_Flexion_3D", "shoulder_flexion")
        hip = _get(angles, "Hip_Angle", "Hip_Flexion_3D", "hip_flexion")
        trunk = _get(angles, "Trunk_Inclination", "Trunk_Inclination_3D")

        shoulder_limit = 50 if view == "front" else 40
        if shoulder > shoulder_limit:
            issues.append(
                FormIssue(
                    code="shoulder_swing",
                    message="Shoulder swing detected — isolate the biceps.",
                    severity="warning",
                    metric="Shoulder_Angle",
                    value=shoulder,
                )
            )

        if history and len(history) >= 5:
            recent_elbows = [_get(h, "Elbow_Angle", "Elbow_Flexion_3D") for h in history[-5:]]
            drift = max(recent_elbows) - min(recent_elbows)
            if drift > 35 and shoulder > 30:
                issues.append(
                    FormIssue(
                        code="elbow_drift",
                        message="Elbow drifting forward — keep elbows pinned.",
                        severity="warning",
                        metric="Elbow_Angle",
                        value=elbow,
                    )
                )

        if history and len(history) >= 8:
            peak_flexion = min(_get(h, "Elbow_Angle", "Elbow_Flexion_3D") for h in history[-20:])
            if peak_flexion > 60:
                issues.append(
                    FormIssue(
                        code="partial_rom",
                        message="Partial range of motion — curl deeper.",
                        severity="warning",
                        metric="Elbow_Angle",
                        value=elbow,
                    )
                )

        if view == "front" and trunk > 20:
            issues.append(
                FormIssue(
                    code="torso_swing",
                    message="Torso leaning — reduce momentum.",
                    severity="warning",
                    metric="Trunk_Inclination",
                    value=trunk,
                )
            )
        elif view != "front" and hip < 155:
            issues.append(
                FormIssue(
                    code="torso_swing",
                    message="Torso leaning — reduce momentum.",
                    severity="warning",
                    metric="Hip_Angle",
                    value=hip,
                )
            )

        return issues
