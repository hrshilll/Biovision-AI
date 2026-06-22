"""Exercise configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.utils.config import load_exercises_config


@dataclass
class RepCountConfig:
    angle_key: str
    down_threshold: float
    up_threshold: float
    count_on: str


@dataclass
class ExerciseConfig:
    key: str
    display_name: str
    category: str
    isometric: bool
    primary_joints: List[str]
    legacy_angle_map: Dict[str, str]
    angle_thresholds: Dict[str, Dict[str, Any]]
    rep_counting: Optional[RepCountConfig]
    scoring_weights: Dict[str, float]
    rule_module: Optional[str]


class ExerciseRegistry:
    """Load and query exercise definitions from YAML."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        raw = dict(config or load_exercises_config())
        self._exercises: Dict[str, ExerciseConfig] = {}
        self._folder_aliases: Dict[str, str] = raw.get("folder_aliases", {})
        for key, data in raw.get("exercises", {}).items():
            rep_raw = data.get("rep_counting")
            rep_cfg = None
            if rep_raw:
                rep_cfg = RepCountConfig(
                    angle_key=rep_raw["angle_key"],
                    down_threshold=float(rep_raw["down_threshold"]),
                    up_threshold=float(rep_raw["up_threshold"]),
                    count_on=rep_raw["count_on"],
                )
            self._exercises[key] = ExerciseConfig(
                key=key,
                display_name=data.get("display_name", key),
                category=data.get("category", "general"),
                isometric=bool(data.get("isometric", False)),
                primary_joints=list(data.get("primary_joints", [])),
                legacy_angle_map=dict(data.get("legacy_angle_map", {})),
                angle_thresholds=dict(data.get("angle_thresholds", {})),
                rep_counting=rep_cfg,
                scoring_weights=dict(data.get("scoring_weights", {})),
                rule_module=data.get("rule_module"),
            )

    def all_exercises(self) -> Dict[str, ExerciseConfig]:
        return dict(self._exercises)

    def get(self, exercise: str) -> Optional[ExerciseConfig]:
        if exercise in self._exercises:
            return self._exercises[exercise]
        normalized = exercise.strip().replace(" ", "_")
        if normalized in self._exercises:
            return self._exercises[normalized]
        alias = self._folder_aliases.get(exercise.lower().replace(" ", "_"))
        if alias and alias in self._exercises:
            return self._exercises[alias]
        for key in self._exercises:
            if key.lower() == exercise.lower():
                return self._exercises[key]
        return None

    def menu_items(self) -> List[Tuple[str, str]]:
        """Return numbered menu entries for live inference."""
        items = sorted(self._exercises.items(), key=lambda x: x[1].display_name)
        return [(str(i + 1), key) for i, (key, _) in enumerate(items)]

    def resolve_angle_value(self, angles: Mapping[str, float], metric_key: str) -> float:
        """Resolve config metric key to actual angle dictionary value."""
        aliases = {
            "elbow_flexion": ["Elbow_Angle", "Elbow_Flexion_3D", "elbow_flexion"],
            "knee_flexion": ["Knee_Angle", "Knee_Flexion_3D", "knee_flexion"],
            "hip_flexion": ["Hip_Angle", "Hip_Flexion_3D", "hip_flexion"],
            "shoulder_flexion": ["Shoulder_Angle", "Shoulder_Flexion_3D", "shoulder_flexion"],
            "shoulder_abduction": ["Shoulder_Abduction_3D", "shoulder_abduction"],
            "trunk_inclination": ["Trunk_Inclination", "Trunk_Inclination_3D", "trunk_inclination"],
            "spine_alignment": ["Spine_Alignment", "Spine_Alignment_3D", "spine_alignment"],
        }
        for candidate in aliases.get(metric_key, [metric_key]):
            if candidate in angles:
                return float(angles[candidate])
        return 0.0

    def legacy_rules(self, exercise: str) -> Dict[str, Tuple[float, float, str]]:
        """Return legacy RULES dict format: angle -> (lo, hi, message)."""
        cfg = self.get(exercise)
        if not cfg:
            return {}
        rules: Dict[str, Tuple[float, float, str]] = {}
        reverse_map = {v: k for k, v in cfg.legacy_angle_map.items()}
        for metric_key, threshold in cfg.angle_thresholds.items():
            legacy_name = reverse_map.get(metric_key)
            if not legacy_name:
                legacy_name = metric_key.replace("_", " ").title().replace(" ", "_")
                if not legacy_name.endswith("_Angle"):
                    legacy_name = legacy_name + "_Angle"
            rules[legacy_name] = (
                float(threshold["lo"]),
                float(threshold["hi"]),
                str(threshold["message"]),
            )
        return rules
