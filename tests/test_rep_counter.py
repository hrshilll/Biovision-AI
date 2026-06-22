"""Unit tests for rep counting state machine."""

from src.exercises.config_loader import ExerciseRegistry
from src.exercises.rep_counter import RepCounter


def _simulate_reps(exercise: str, angle_sequence: list[float]) -> int:
    registry = ExerciseRegistry()
    counter = RepCounter(exercise, registry)
    for angle in angle_sequence:
        counter.update({"Elbow_Angle": angle, "Knee_Angle": angle, "Hip_Angle": angle})
    return counter.count


def test_bicep_curl_counts_on_extension():
  # extended -> curled -> extended = 1 rep
    count = _simulate_reps("Bicep_curls", [160, 160, 110, 110, 100, 100, 150, 150])
    assert count >= 1


def test_squat_counts_on_descent():
    registry = ExerciseRegistry()
    counter = RepCounter("Squats", registry)
    for angle in [170, 170, 120, 120, 90, 90, 160, 160]:
        counter.update({"Knee_Angle": angle})
    assert counter.count >= 1


def test_lateral_raise_ascending_thresholds():
    registry = ExerciseRegistry()
    counter = RepCounter("Lateral_Raise", registry)
    assert counter._ascending is True
    for angle in [20, 20, 80, 80, 25, 25, 75, 75]:
        counter.update({"Shoulder_Abduction_3D": angle})
    assert counter.count >= 1


def test_rep_counter_initializes_without_counting():
    registry = ExerciseRegistry()
    counter = RepCounter("Bicep_curls", registry)
    counter.update({"Elbow_Angle": 160})
    assert counter.count == 0
    assert counter.stage == "up"
