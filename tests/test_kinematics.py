"""Unit tests for kinematic tracking."""

import time

from src.biomechanics.kinematics import KinematicTracker


def test_kinematic_velocity_and_rom():
    tracker = KinematicTracker()
    t0 = time.time()
    tracker.update(t0, {"Elbow_Angle": 90.0})
    tracker.update(t0 + 0.1, {"Elbow_Angle": 100.0})
    tracker.update(t0 + 0.2, {"Elbow_Angle": 110.0})
    rom = tracker.compute_rom(["Elbow_Angle"])
    assert rom["Elbow_Angle"] == 20.0
    summary = tracker.summarize(["Elbow_Angle"])
    assert summary.stability_score >= 0
    assert summary.smoothness_score >= 0
