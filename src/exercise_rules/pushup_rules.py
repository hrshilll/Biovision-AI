"""Pushup-specific biomechanical error detection."""

from __future__ import annotations

from typing import List, Mapping, Optional

from src.exercise_rules.base import ExerciseRuleEngine, FormIssue, _get


class PushupRuleEngine(ExerciseRuleEngine):
    exercise_key = "Pushups"

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
        spine = _get(angles, "Spine_Alignment", "Spine_Alignment_3D", "spine_alignment")
        plank = _get(angles, "Plank_Body_Angle", default=hip)
        flare = _get(angles, "Elbow_Flare_Front", default=shoulder)

        flare_metric = flare if view == "front" else shoulder
        flare_limit = 55 if view == "front" else 95
        if flare_metric > flare_limit:
            issues.append(
                FormIssue(
                    code="elbow_flare",
                    message="Elbow flare — keep elbows closer to torso.",
                    severity="warning",
                    metric="Shoulder_Angle",
                    value=flare_metric,
                )
            )

        if elbow > 120:
            issues.append(
                FormIssue(
                    code="insufficient_depth",
                    message="Insufficient depth — chest not low enough.",
                    severity="warning",
                    metric="Elbow_Angle",
                    value=elbow,
                )
            )

        body_line = plank if plank > 0 else hip
        if body_line < 155 or spine < 155:
            issues.append(
                FormIssue(
                    code="hip_sag",
                    message="Hip sag detected — maintain straight plank line.",
                    severity="error",
                    metric="Hip_Angle",
                    value=body_line,
                )
            )

        return issues
