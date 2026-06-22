"""Unit tests for exercise configuration."""

from src.exercises.config_loader import ExerciseRegistry
from src.exercises.form_checker import check_form


def test_registry_loads_exercises():
    registry = ExerciseRegistry()
    exercises = registry.all_exercises()
    assert "Squats" in exercises
    assert "Bicep_curls" in exercises
    assert len(exercises) >= 20


def test_squat_form_check():
    registry = ExerciseRegistry()
    angles = {
        "Elbow_Angle": 120,
        "Shoulder_Angle": 100,
        "Hip_Angle": 140,
        "Knee_Angle": 140,
        "Trunk_Inclination": 40,
        "Knee_ValGus_Index": 15,
    }
    is_good, issues = check_form(angles, "Squats", registry=registry)
    assert not is_good
    assert any("inward" in i.message.lower() or "lean" in i.message.lower() for i in issues)
