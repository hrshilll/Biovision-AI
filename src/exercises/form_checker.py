"""Form checking combining YAML thresholds and exercise-specific rules."""

from __future__ import annotations

from typing import List, Mapping, Optional, Tuple

from src.exercise_rules import get_rule_engine
from src.exercise_rules.base import FormIssue
from src.exercises.config_loader import ExerciseRegistry


def check_form(
    angles: Mapping[str, float],
    exercise: str,
    landmarks=None,
    history: Optional[List[Mapping[str, float]]] = None,
    registry: Optional[ExerciseRegistry] = None,
) -> Tuple[bool, List[FormIssue]]:
    """Return whether form is acceptable and a list of detected issues."""
    registry = registry or ExerciseRegistry()
    cfg = registry.get(exercise)
    issues: List[FormIssue] = []
    view = str(angles.get("Camera_View", "oblique"))

    if cfg:
        for metric_key, threshold in cfg.angle_thresholds.items():
            val = registry.resolve_angle_value(angles, metric_key)
            if val <= 0:
                continue
            lo, hi = float(threshold["lo"]), float(threshold["hi"])
            if view == "oblique":
                margin = max(5.0, (hi - lo) * 0.12)
                lo -= margin
                hi += margin
            if not (lo <= val <= hi):
                reverse = {v: k for k, v in cfg.legacy_angle_map.items()}
                legacy_name = reverse.get(metric_key, metric_key)
                issues.append(
                    FormIssue(
                        code=f"{metric_key}_out_of_range",
                        message=str(threshold["message"]),
                        severity="warning",
                        metric=legacy_name,
                        value=val,
                        lo=float(threshold["lo"]),
                        hi=float(threshold["hi"]),
                    )
                )
        engine = get_rule_engine(cfg.rule_module)
        if engine:
            rule_issues = engine.evaluate(angles, landmarks, history)
            seen_codes = {i.code for i in issues}
            for issue in rule_issues:
                if issue.code not in seen_codes:
                    issues.append(issue)
                    seen_codes.add(issue.code)

    is_good = len(issues) == 0
    return is_good, issues


def issues_to_legacy_tuples(issues: List[FormIssue]) -> List[tuple]:
    """Convert FormIssue objects to legacy live_inference tuple format."""
    return [issue.to_tuple() for issue in issues]
