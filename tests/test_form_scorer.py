"""Unit tests for form scoring."""

from src.biomechanics.kinematics import SessionKinematics
from src.exercises.form_scorer import FormScorer


def test_form_score_range():
    scorer = FormScorer()
    kinematics = SessionKinematics(
        rom={"Elbow_Angle": 80},
        stability_score=85,
        symmetry_score=90,
        tempo_score=80,
    )
    angles = {
        "Elbow_Angle": 90,
        "Shoulder_Angle": 30,
        "Hip_Angle": 170,
        "Knee_Angle": 170,
    }
    score = scorer.compute("Bicep_curls", angles, kinematics)
    assert 0 <= score.total <= 100
    assert score.rom >= 0
    assert "Form Score" in score.formatted()
