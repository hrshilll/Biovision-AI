"""Exercise-specific biomechanical rule checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional


@dataclass
class FormIssue:
    """A detected form error with human-readable feedback."""

    code: str
    message: str
    severity: str = "warning"
    metric: Optional[str] = None
    value: Optional[float] = None
    lo: float = 0.0
    hi: float = 0.0

    def to_tuple(self) -> tuple:
        """Legacy tuple format for live_inference compatibility."""
        return (self.metric or self.code, self.message, self.value or 0.0, self.lo, self.hi)


class ExerciseRuleEngine:
    """Base class for exercise-specific error detection."""

    exercise_key: str = "generic"

    def evaluate(
        self,
        angles: Mapping[str, float],
        landmarks=None,
        history: Optional[List[Mapping[str, float]]] = None,
    ) -> List[FormIssue]:
        raise NotImplementedError


def _get(angles: Mapping[str, float], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in angles:
            return float(angles[key])
    return default
