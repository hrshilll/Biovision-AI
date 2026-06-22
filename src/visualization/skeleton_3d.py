"""Interactive 3D skeleton visualization using Plotly."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

SKELETON_CONNECTIONS: List[Tuple[int, int]] = list(mp_pose.POSE_CONNECTIONS)

JOINT_LABELS = {
    mp_pose.PoseLandmark.NOSE.value: "Nose",
    mp_pose.PoseLandmark.LEFT_SHOULDER.value: "L.Shoulder",
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value: "R.Shoulder",
    mp_pose.PoseLandmark.LEFT_ELBOW.value: "L.Elbow",
    mp_pose.PoseLandmark.RIGHT_ELBOW.value: "R.Elbow",
    mp_pose.PoseLandmark.LEFT_WRIST.value: "L.Wrist",
    mp_pose.PoseLandmark.RIGHT_WRIST.value: "R.Wrist",
    mp_pose.PoseLandmark.LEFT_HIP.value: "L.Hip",
    mp_pose.PoseLandmark.RIGHT_HIP.value: "R.Hip",
    mp_pose.PoseLandmark.LEFT_KNEE.value: "L.Knee",
    mp_pose.PoseLandmark.RIGHT_KNEE.value: "R.Knee",
    mp_pose.PoseLandmark.LEFT_ANKLE.value: "L.Ankle",
    mp_pose.PoseLandmark.RIGHT_ANKLE.value: "R.Ankle",
}


class Skeleton3DViewer:
    """Real-time 3D skeleton viewer with angle overlays and error highlighting."""

    def __init__(self, min_visibility: float = 0.30) -> None:
        self.min_visibility = min_visibility
        self._fig = None
        self._enabled = False
        self._rep_count = 0
        self._exercise_name = ""

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> bool:
        """Initialize Plotly figure. Returns False if plotly is unavailable."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return False

        self._fig = go.Figure()
        self._apply_layout()
        self._enabled = True
        return True

    def _apply_layout(self, subtitle: str = "") -> None:
        title = "BioVision AI — 3D Skeleton"
        if self._exercise_name:
            title += f" | {self._exercise_name}"
        if self._rep_count:
            title += f" | Reps: {self._rep_count}"
        if subtitle:
            title += f"<br><span style='font-size:12px'>{subtitle}</span>"

        self._fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(color="#E8F4FF", size=18)),
            paper_bgcolor="#0B1020",
            plot_bgcolor="#0B1020",
            font=dict(color="#D7E3F4"),
            scene=dict(
                xaxis=dict(
                    title="X (m)",
                    backgroundcolor="#111827",
                    gridcolor="#2A3A55",
                    zerolinecolor="#3B4F72",
                    showbackground=True,
                ),
                yaxis=dict(
                    title="Y (m)",
                    backgroundcolor="#111827",
                    gridcolor="#2A3A55",
                    zerolinecolor="#3B4F72",
                    showbackground=True,
                ),
                zaxis=dict(
                    title="Z (m)",
                    backgroundcolor="#111827",
                    gridcolor="#2A3A55",
                    zerolinecolor="#3B4F72",
                    showbackground=True,
                ),
                aspectmode="data",
                bgcolor="#0B1020",
                camera=dict(eye=dict(x=1.6, y=1.2, z=1.4)),
            ),
            margin=dict(l=0, r=0, t=70, b=0),
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            uirevision="biovision-3d",
        )

    def _extract_points(
        self,
        landmarks,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str], List[str]]:
        xs, ys, zs, indices, labels, coord_text = [], [], [], [], [], []
        vis_thresh = max(0.12, self.min_visibility * 0.5)
        for idx, lm in enumerate(landmarks):
            if lm.visibility < vis_thresh:
                continue
            xs.append(lm.x)
            ys.append(-lm.y)  # invert Y so body stands upright in plot
            zs.append(lm.z)
            indices.append(idx)
            label = JOINT_LABELS.get(idx, mp_pose.PoseLandmark(idx).name)
            labels.append(label)
            coord_text.append(f"{label}<br>({lm.x:+.3f}, {lm.y:+.3f}, {lm.z:+.3f})")
        return (
            np.array(xs),
            np.array(ys),
            np.array(zs),
            indices,
            labels,
            coord_text,
        )

    def _issue_joint_indices(self, issues: Sequence, side_used: str) -> set:
        side = (side_used or "RIGHT").upper()
        mapping = {
            "Elbow_Angle": f"{side}_ELBOW",
            "Shoulder_Angle": f"{side}_SHOULDER",
            "Hip_Angle": f"{side}_HIP",
            "Knee_Angle": f"{side}_KNEE",
            "Trunk_Inclination": f"{side}_HIP",
            "Spine_Alignment": f"{side}_SHOULDER",
        }
        bad = set()
        for issue in issues:
            metric = issue[0] if isinstance(issue, tuple) else getattr(issue, "metric", "")
            name = mapping.get(metric or "")
            if name and hasattr(mp_pose.PoseLandmark, name):
                bad.add(mp_pose.PoseLandmark[name].value)
        return bad

    def update(
        self,
        landmarks,
        angles: Optional[Mapping[str, float]] = None,
        issues: Optional[Sequence] = None,
        side_used: str = "RIGHT",
        rep_count: int = 0,
        exercise_name: str = "",
    ) -> None:
        """Update the 3D plot with current pose."""
        if not self._enabled or self._fig is None or landmarks is None:
            return

        import plotly.graph_objects as go

        self._rep_count = rep_count
        self._exercise_name = exercise_name

        xs, ys, zs, indices, labels, coord_text = self._extract_points(landmarks)
        if len(xs) == 0:
            return

        index_map = {idx: i for i, idx in enumerate(indices)}
        bad_joints = self._issue_joint_indices(issues or [], side_used)

        colors = ["#FF4D6D" if idx in bad_joints else "#3DFF8A" for idx in indices]

        self._fig.data = []

        # Floor reference grid
        if len(xs) > 0:
            floor_y = float(np.min(ys) - 0.05)
            gx = np.linspace(float(np.min(xs) - 0.2), float(np.max(xs) + 0.2), 8)
            gz = np.linspace(float(np.min(zs) - 0.2), float(np.max(zs) + 0.2), 8)
            grid_x, grid_z = np.meshgrid(gx, gz)
            grid_y = np.full_like(grid_x, floor_y)
            self._fig.add_trace(
                go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=grid_z,
                    colorscale=[[0, "#1A2438"], [1, "#1A2438"]],
                    showscale=False,
                    opacity=0.35,
                    name="Floor",
                )
            )

        self._fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers+text",
                marker=dict(size=9, color=colors, line=dict(width=2, color="#FFFFFF")),
                text=labels,
                textposition="top center",
                textfont=dict(size=10, color="#E8F4FF"),
                customdata=coord_text,
                hovertemplate="%{customdata}<extra></extra>",
                name="Joints",
            )
        )

        bone_x, bone_y, bone_z = [], [], []
        for a_idx, b_idx in SKELETON_CONNECTIONS:
            if a_idx in index_map and b_idx in index_map:
                ai, bi = index_map[a_idx], index_map[b_idx]
                bone_x.extend([xs[ai], xs[bi], None])
                bone_y.extend([ys[ai], ys[bi], None])
                bone_z.extend([zs[ai], zs[bi], None])

        self._fig.add_trace(
            go.Scatter3d(
                x=bone_x,
                y=bone_y,
                z=bone_z,
                mode="lines",
                line=dict(color="#4CC9F0", width=7),
                name="Bones",
            )
        )

        subtitle = ""
        if angles:
            subtitle = " | ".join(
                f"{k.replace('_', ' ')}: {v:.1f}°"
                for k, v in list(angles.items())[:8]
                if k.endswith("_Angle") or k.endswith("_3D")
            )
        self._apply_layout(subtitle=subtitle)

    def render_to_html(self, html_path: str, auto_open: bool = False) -> None:
        """Write current figure to HTML for live refresh in browser."""
        if not self._enabled or self._fig is None:
            return
        self._fig.write_html(
            html_path,
            auto_open=auto_open,
            include_plotlyjs="cdn",
            full_html=True,
        )

    def close(self) -> None:
        self._enabled = False
        self._fig = None
