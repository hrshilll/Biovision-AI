"""Universal form scoring system (0-100)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from src.biomechanics.kinematics import SessionKinematics
from src.biomechanics.posture_metrics import posture_score_from_metrics, compute_posture_metrics
from src.exercise_rules.base import FormIssue
from src.exercises.config_loader import ExerciseRegistry


@dataclass
class FormScore:
    """Weighted form score with category subscores."""

    total: float
    rom: float
    stability: float
    symmetry: float
    tempo: float
    posture: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Form_Score": self.total,
            "ROM_Score": self.rom,
            "Stability_Score": self.stability,
            "Symmetry_Score": self.symmetry,
            "Tempo_Score": self.tempo,
            "Posture_Score": self.posture,
        }

    def formatted(self) -> str:
        return (
            f"Form Score: {self.total:.0f}/100\n"
            f"ROM: {self.rom:.0f} | Stability: {self.stability:.0f} | "
            f"Symmetry: {self.symmetry:.0f} | Tempo: {self.tempo:.0f} | "
            f"Posture: {self.posture:.0f}"
        )


class FormScorer:
    """Compute weighted form scores from angles, kinematics, and rule issues."""

    DEFAULT_WEIGHTS = {
        "rom": 0.30,
        "stability": 0.25,
        "symmetry": 0.20,
        "tempo": 0.15,
        "posture": 0.10,
    }

    def __init__(self, registry: Optional[ExerciseRegistry] = None) -> None:
        self.registry = registry or ExerciseRegistry()

    def score_threshold_compliance(
        self,
        exercise: str,
        angles: Mapping[str, float],
    ) -> float:
        cfg = self.registry.get(exercise)
        if not cfg or not cfg.angle_thresholds:
            return 100.0
        passed = 0
        total = 0
        for metric_key, threshold in cfg.angle_thresholds.items():
            val = self.registry.resolve_angle_value(angles, metric_key)
            lo, hi = float(threshold["lo"]), float(threshold["hi"])
            total += 1
            if lo <= val <= hi:
                passed += 1
        return round(100.0 * passed / max(total, 1), 1)

    def score_from_issues(self, issues: List[FormIssue]) -> float:
        if not issues:
            return 100.0
        penalty = sum(15 if i.severity == "error" else 8 for i in issues)
        return round(max(0.0, 100.0 - penalty), 1)

    def compute(
        self,
        exercise: str,
        angles: Mapping[str, float],
        kinematics: Optional[SessionKinematics] = None,
        issues: Optional[List[FormIssue]] = None,
        landmarks=None,
    ) -> FormScore:
        cfg = self.registry.get(exercise)
        weights = dict(self.DEFAULT_WEIGHTS)
        if cfg and cfg.scoring_weights:
            weights.update(cfg.scoring_weights)
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v / total_w for k, v in weights.items()}

        rom_score = 100.0
        if kinematics and kinematics.rom:
            rom_values = list(kinematics.rom.values())
            rom_score = min(max(rom_values) * 0.8, 100.0) if rom_values else 100.0
        rom_score = round(min(rom_score, 100.0), 1)

        stability_score = kinematics.stability_score if kinematics else 100.0
        symmetry_score = kinematics.symmetry_score if kinematics else 100.0
        tempo_score = kinematics.tempo_score if kinematics else 100.0

        posture_score = 100.0
        if landmarks is not None:
            metrics = compute_posture_metrics(landmarks)
            if metrics:
                posture_score = posture_score_from_metrics(metrics)
        threshold_score = self.score_threshold_compliance(exercise, angles)
        issue_score = self.score_from_issues(issues or [])
        posture_score = round((posture_score + threshold_score + issue_score) / 3, 1)

        total = (
            rom_score * weights["rom"]
            + stability_score * weights["stability"]
            + symmetry_score * weights["symmetry"]
            + tempo_score * weights["tempo"]
            + posture_score * weights["posture"]
        )
        return FormScore(
            total=round(total, 1),
            rom=rom_score,
            stability=stability_score,
            symmetry=symmetry_score,
            tempo=tempo_score,
            posture=posture_score,
        )
