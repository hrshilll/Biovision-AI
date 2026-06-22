"""Temporal kinematic metrics: velocity, acceleration, ROM, smoothness."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass
class KinematicSample:
    """Single-frame kinematic snapshot."""

    timestamp: float
    angles: Mapping[str, float]
    velocities: Dict[str, float] = field(default_factory=dict)
    accelerations: Dict[str, float] = field(default_factory=dict)


@dataclass
class SessionKinematics:
    """Aggregated kinematic metrics for a session or rep window."""

    rom: Dict[str, float] = field(default_factory=dict)
    mean_velocity: Dict[str, float] = field(default_factory=dict)
    peak_velocity: Dict[str, float] = field(default_factory=dict)
    mean_acceleration: Dict[str, float] = field(default_factory=dict)
    stability_score: float = 100.0
    symmetry_score: float = 100.0
    smoothness_score: float = 100.0
    trajectory_consistency: float = 100.0
    tempo_score: float = 100.0


class KinematicTracker:
    """Track angle derivatives and session-level biomechanical metrics."""

    def __init__(self, history_size: int = 120) -> None:
        self.history_size = history_size
        self._timestamps: Deque[float] = deque(maxlen=history_size)
        self._angle_history: Dict[str, Deque[float]] = {}
        self._velocity_history: Dict[str, Deque[float]] = {}
        self._prev_velocities: Dict[str, float] = {}
        self._prev_timestamp: Optional[float] = None
        self._all_samples: List[KinematicSample] = []

    def reset(self) -> None:
        self._timestamps.clear()
        self._angle_history.clear()
        self._velocity_history.clear()
        self._prev_velocities.clear()
        self._prev_timestamp = None
        self._all_samples.clear()

    def update(self, timestamp: float, angles: Mapping[str, float]) -> KinematicSample:
        """Append a frame and compute instantaneous velocity/acceleration."""
        dt = 0.0
        if self._prev_timestamp is not None:
            dt = max(timestamp - self._prev_timestamp, 1e-6)

        velocities: Dict[str, float] = {}
        accelerations: Dict[str, float] = {}

        for key, value in angles.items():
            if key not in self._angle_history:
                self._angle_history[key] = deque(maxlen=self.history_size)
                self._velocity_history[key] = deque(maxlen=self.history_size)

            history = self._angle_history[key]
            if history and dt > 0:
                velocity = (value - history[-1]) / dt
            else:
                velocity = 0.0

            if key in self._prev_velocities and dt > 0:
                acceleration = (velocity - self._prev_velocities[key]) / dt
            else:
                acceleration = 0.0

            history.append(value)
            self._velocity_history[key].append(velocity)
            velocities[key] = round(velocity, 3)
            accelerations[key] = round(acceleration, 3)
            self._prev_velocities[key] = velocity

        self._timestamps.append(timestamp)
        self._prev_timestamp = timestamp

        sample = KinematicSample(
            timestamp=timestamp,
            angles=dict(angles),
            velocities=velocities,
            accelerations=accelerations,
        )
        self._all_samples.append(sample)
        return sample

    def compute_rom(self, angle_keys: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Range of motion per tracked angle key."""
        keys = angle_keys or self._angle_history.keys()
        rom: Dict[str, float] = {}
        for key in keys:
            values = self._angle_history.get(key)
            if values and len(values) > 1:
                rom[key] = round(max(values) - min(values), 2)
        return rom

    def compute_stability_score(self, primary_keys: Iterable[str]) -> float:
        """Lower angular velocity variance implies higher stability."""
        variances: List[float] = []
        for key in primary_keys:
            velocities = self._velocity_history.get(key)
            if velocities and len(velocities) > 2:
                variances.append(float(np.var(velocities)))
        if not variances:
            return 100.0
        score = 100.0 - min(np.mean(variances) * 2.5, 100.0)
        return round(max(score, 0.0), 1)

    def compute_symmetry_score(self, bilateral_pairs: Mapping[str, str]) -> float:
        """Compare left/right angle means; perfect symmetry yields 100."""
        diffs: List[float] = []
        for left_key, right_key in bilateral_pairs.items():
            left_vals = self._angle_history.get(left_key)
            right_vals = self._angle_history.get(right_key)
            if left_vals and right_vals:
                diffs.append(abs(float(np.mean(left_vals)) - float(np.mean(right_vals))))
        if not diffs:
            return 100.0
        score = 100.0 - min(float(np.mean(diffs)) * 1.5, 100.0)
        return round(max(score, 0.0), 1)

    def compute_smoothness_score(self, primary_keys: Iterable[str]) -> float:
        """Jerk-based smoothness proxy using acceleration variance."""
        jerk_vars: List[float] = []
        for key in primary_keys:
            accels = [
                sample.accelerations.get(key, 0.0)
                for sample in self._all_samples
                if key in sample.accelerations
            ]
            if len(accels) > 3:
                jerk_vars.append(float(np.var(np.diff(accels))))
        if not jerk_vars:
            return 100.0
        score = 100.0 - min(float(np.mean(jerk_vars)) * 0.05, 100.0)
        return round(max(score, 0.0), 1)

    def compute_trajectory_consistency(self, primary_key: str) -> float:
        """Compare successive angle trajectories using correlation."""
        values = list(self._angle_history.get(primary_key, []))
        if len(values) < 10:
            return 100.0
        mid = len(values) // 2
        first = np.array(values[:mid], dtype=float)
        second = np.array(values[mid : mid * 2], dtype=float)
        if len(first) != len(second) or len(first) < 3:
            return 100.0
        corr = float(np.corrcoef(first, second)[0, 1])
        if np.isnan(corr):
            return 100.0
        return round(max(min((corr + 1) * 50, 100.0), 0.0), 1)

    def compute_tempo_score(self, rep_durations: List[float], target_seconds: float = 2.5) -> float:
        """Score tempo consistency relative to an ideal rep duration."""
        if not rep_durations:
            return 100.0
        deviations = [abs(d - target_seconds) / target_seconds for d in rep_durations]
        score = 100.0 - min(float(np.mean(deviations)) * 100.0, 100.0)
        return round(max(score, 0.0), 1)

    def summarize(
        self,
        primary_keys: Iterable[str],
        bilateral_pairs: Optional[Mapping[str, str]] = None,
        rep_durations: Optional[List[float]] = None,
    ) -> SessionKinematics:
        """Compute full session kinematic summary."""
        primary = list(primary_keys)
        return SessionKinematics(
            rom=self.compute_rom(primary),
            mean_velocity={
                key: round(float(np.mean(self._velocity_history[key])), 3)
                for key in primary
                if key in self._velocity_history and self._velocity_history[key]
            },
            peak_velocity={
                key: round(float(np.max(np.abs(self._velocity_history[key]))), 3)
                for key in primary
                if key in self._velocity_history and self._velocity_history[key]
            },
            mean_acceleration={
                key: round(
                    float(np.mean([s.accelerations.get(key, 0.0) for s in self._all_samples])),
                    3,
                )
                for key in primary
                if self._all_samples
            },
            stability_score=self.compute_stability_score(primary),
            symmetry_score=self.compute_symmetry_score(bilateral_pairs or {}),
            smoothness_score=self.compute_smoothness_score(primary),
            trajectory_consistency=self.compute_trajectory_consistency(
                primary[0] if primary else "Elbow_Angle"
            ),
            tempo_score=self.compute_tempo_score(rep_durations or []),
        )
