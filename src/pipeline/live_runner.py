"""
BioVision AI — Live webcam inference with 3D biomechanics.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.session_metrics import SessionAnalyzer, export_analytics_report, export_session_excel
from src.biomechanics.kinematics import KinematicTracker
from src.exercises.config_loader import ExerciseRegistry
from src.exercises.form_checker import check_form, issues_to_legacy_tuples
from src.exercises.form_scorer import FormScorer
from src.exercises.rep_counter import RepCounter
from src.utils.config import get_path, load_settings
from src.utils.logging_config import setup_logging
from src.utils.pose_extractor import PoseProcessor
from src.visualization.skeleton_3d import Skeleton3DViewer

logger = setup_logging(__name__)

# Colours (BGR)
GREEN = (50, 205, 50)
RED = (30, 30, 220)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DGREY = (35, 35, 35)
LGREY = (170, 170, 170)
YELLOW = (0, 215, 255)
CYAN = (255, 200, 0)

mp_pose = mp.solutions.pose

LANDMARK_EXPORT_JOINTS = [
    ("L_Shoulder", mp_pose.PoseLandmark.LEFT_SHOULDER),
    ("R_Shoulder", mp_pose.PoseLandmark.RIGHT_SHOULDER),
    ("L_Elbow", mp_pose.PoseLandmark.LEFT_ELBOW),
    ("R_Elbow", mp_pose.PoseLandmark.RIGHT_ELBOW),
    ("L_Wrist", mp_pose.PoseLandmark.LEFT_WRIST),
    ("R_Wrist", mp_pose.PoseLandmark.RIGHT_WRIST),
    ("L_Hip", mp_pose.PoseLandmark.LEFT_HIP),
    ("R_Hip", mp_pose.PoseLandmark.RIGHT_HIP),
    ("L_Knee", mp_pose.PoseLandmark.LEFT_KNEE),
    ("R_Knee", mp_pose.PoseLandmark.RIGHT_KNEE),
    ("L_Ankle", mp_pose.PoseLandmark.LEFT_ANKLE),
    ("R_Ankle", mp_pose.PoseLandmark.RIGHT_ANKLE),
]


def snapshot_landmarks(frame_n: int, world_lms, image_lms, exercise: str) -> Dict:
    src = world_lms or image_lms
    if src is None:
        return {}
    row = {"Frame": frame_n, "Exercise": exercise, "Coord_System": "meters" if world_lms else "normalized"}
    for name, landmark in LANDMARK_EXPORT_JOINTS:
        lm = src[landmark.value]
        row[f"{name}_X"] = round(lm.x, 5)
        row[f"{name}_Y"] = round(lm.y, 5)
        row[f"{name}_Z"] = round(lm.z, 5)
        row[f"{name}_Vis"] = round(lm.visibility, 3)
    return row


def load_models(model_dir: Path):
    paths = {
        "clf": model_dir / "form_classifier.pkl",
        "encoder": model_dir / "label_encoder.pkl",
        "scaler": model_dir / "scaler.pkl",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        logger.error("Missing model files: %s. Run train_model.py first.", missing)
        return None, None, None, None
    with paths["clf"].open("rb") as handle:
        clf = pickle.load(handle)
    with paths["encoder"].open("rb") as handle:
        encoder = pickle.load(handle)
    with paths["scaler"].open("rb") as handle:
        scaler = pickle.load(handle)
    feature_path = model_dir / "feature_columns.pkl"
    feature_cols = None
    if feature_path.exists():
        with feature_path.open("rb") as handle:
            feature_cols = pickle.load(handle)
    return clf, encoder, scaler, feature_cols


def build_feature_vector(
    angles: Dict[str, float],
    exercise: str,
    encoder,
    scaler,
    feature_cols: Optional[List[str]],
) -> np.ndarray:
    legacy = ["Elbow_Angle", "Shoulder_Angle", "Hip_Angle", "Knee_Angle"]
    cols = feature_cols or legacy
    try:
        ex_enc = encoder.transform([exercise])[0]
    except ValueError:
        ex_enc = 0
    values = [float(angles.get(c, 0.0)) for c in cols]
    values.append(float(ex_enc))
    return scaler.transform([values])


def txt(img, text, pos, scale=0.55, color=WHITE, bold=False):
    t = 2 if bold else 1
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, t + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)


def panel(img, x1, y1, x2, y2, color=DGREY, alpha=0.82):
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


def bar(img, x, y, w, h, pct, fg, bg=DGREY):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(w * pct), y + h), fg, -1)


KEY_LANDMARK_LABELS = {
    mp_pose.PoseLandmark.LEFT_SHOULDER.value: "L.Sh",
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value: "R.Sh",
    mp_pose.PoseLandmark.LEFT_ELBOW.value: "L.El",
    mp_pose.PoseLandmark.RIGHT_ELBOW.value: "R.El",
    mp_pose.PoseLandmark.LEFT_WRIST.value: "L.Wr",
    mp_pose.PoseLandmark.RIGHT_WRIST.value: "R.Wr",
    mp_pose.PoseLandmark.LEFT_HIP.value: "L.Hp",
    mp_pose.PoseLandmark.RIGHT_HIP.value: "R.Hp",
    mp_pose.PoseLandmark.LEFT_KNEE.value: "L.Kn",
    mp_pose.PoseLandmark.RIGHT_KNEE.value: "R.Kn",
}


JOINT_ANGLE_MAP = {
    "LEFT_ELBOW": "Elbow_Angle",
    "RIGHT_ELBOW": "Elbow_Angle",
    "LEFT_SHOULDER": "Shoulder_Angle",
    "RIGHT_SHOULDER": "Shoulder_Angle",
    "LEFT_HIP": "Hip_Angle",
    "RIGHT_HIP": "Hip_Angle",
    "LEFT_KNEE": "Knee_Angle",
    "RIGHT_KNEE": "Knee_Angle",
    "LEFT_WRIST": "Elbow_Angle",
    "RIGHT_WRIST": "Elbow_Angle",
    "LEFT_ANKLE": "Knee_Angle",
    "RIGHT_ANKLE": "Knee_Angle",
}


def _joint_color(name: str, angles: dict, rules: dict) -> tuple:
    """Return green/red/cyan for a joint based on form rules."""
    angle_key = JOINT_ANGLE_MAP.get(name)
    if angle_key and angle_key in rules and angle_key in angles:
        lo, hi, _ = rules[angle_key]
        return GREEN if lo <= angles[angle_key] <= hi else RED
    return (100, 255, 255)


def draw_skeleton_colored(img, lms, angles, rules, w, h, min_vis, is_good=True, show_labels=False):
    """
    Draw pose skeleton — white bones (photo style), green/red when form is off.
    """
    vis_thresh = max(0.12, min_vis * 0.5)

    for conn in mp_pose.POSE_CONNECTIONS:
        a_lm = lms[conn[0]]
        b_lm = lms[conn[1]]
        if a_lm.visibility < vis_thresh or b_lm.visibility < vis_thresh:
            continue
        ax, ay = int(a_lm.x * w), int(a_lm.y * h)
        bx, by = int(b_lm.x * w), int(b_lm.y * h)

        a_name = mp_pose.PoseLandmark(conn[0]).name
        b_name = mp_pose.PoseLandmark(conn[1]).name
        a_col = _joint_color(a_name, angles, rules)
        b_col = _joint_color(b_name, angles, rules)
        if a_col == RED or b_col == RED:
            bone_col = RED
        elif a_col == GREEN and b_col == GREEN and rules:
            bone_col = GREEN
        else:
            bone_col = WHITE

        cv2.line(img, (ax, ay), (bx, by), bone_col, 4, cv2.LINE_AA)

    for idx, lm in enumerate(lms):
        if lm.visibility < vis_thresh:
            continue
        name = mp_pose.PoseLandmark(idx).name
        color = _joint_color(name, angles, rules) if rules else WHITE
        if not rules:
            color = WHITE
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 7, WHITE, 1, cv2.LINE_AA)


def draw_left_panel(
    img, exercise, is_good, confidence, issues, angles, rep_counter,
    fps, h, form_score, registry: ExerciseRegistry, min_vis, coord_lines=None,
):
    """
    Left overlay panel matching photo reference:
    - GOOD/BAD bar → exercise name → angle bars → rep counter → details
    """
    pw = 300
    panel(img, 0, 0, pw, h, (18, 18, 22), 0.78)

    # 1. GOOD / BAD bar
    hcol = GREEN if is_good else RED
    cv2.rectangle(img, (0, 0), (pw, 42), hcol, -1)
    txt(img, "GOOD FORM" if is_good else "BAD FORM", (12, 30), scale=0.85, color=WHITE, bold=True)

    # 2. Exercise name
    cfg = registry.get(exercise)
    display = cfg.display_name if cfg else exercise.replace("_", " ")
    y_cursor = 62
    txt(img, display.upper(), (12, y_cursor), scale=0.55, color=WHITE, bold=True)
    y_cursor += 22

    # 3. Angle bars (above rep counter — matches photo)
    rules = registry.legacy_rules(exercise)
    angle_items: List[Tuple[str, float]] = []
    primary = ["Elbow_Angle", "Shoulder_Angle", "Hip_Angle", "Knee_Angle"]
    for key in primary:
        if angles and key in angles:
            angle_items.append((key, angles[key]))
    if len(angle_items) < 4:
        for aname, val in (angles or {}).items():
            if aname.endswith("_Angle") and (aname, val) not in angle_items:
                angle_items.append((aname, val))
    if not angle_items and rep_counter and rep_counter.tracking_angle > 0:
        if cfg and cfg.rep_counting:
            legacy = {v: k for k, v in cfg.legacy_angle_map.items()}.get(
                cfg.rep_counting.angle_key, "Elbow_Angle"
            )
            angle_items.append((legacy, rep_counter.tracking_angle))

    bar_w = pw - 20
    for aname, val in angle_items[:4]:
        lo, hi, _ = rules.get(aname, (0, 180, ""))
        ok = lo <= val <= hi
        acol = GREEN if ok else RED
        bar_h = 11
        cv2.rectangle(img, (10, y_cursor), (10 + bar_w, y_cursor + bar_h), (45, 45, 50), -1)
        fill_ratio = min(max((val - lo) / max(hi - lo, 1), 0), 1) if hi > lo else 0.5
        fill_w = max(4, int(bar_w * fill_ratio))
        cv2.rectangle(img, (10, y_cursor), (10 + fill_w, y_cursor + bar_h), acol, -1)
        y_cursor += 18

    y_cursor += 10

    # 4. Rep counter (large number + UP/DOWN)
    if rep_counter.cfg is None:
        txt(img, "HOLD", (12, y_cursor + 30), scale=0.7, color=LGREY, bold=True)
        y_cursor += 50
    else:
        rep_str = str(rep_counter.count)
        txt(img, rep_str, (12, y_cursor + 38), scale=2.8, color=WHITE, bold=True)
        stage_color = YELLOW if rep_counter.stage == "up" else CYAN
        txt(img, rep_counter.stage.upper(), (100, y_cursor + 38), scale=0.85, color=stage_color, bold=True)
        y_cursor += 52

    # 5. Below rep counter — tracking angle + per-joint values
    if rep_counter and rep_counter.cfg and rep_counter.tracking_angle > 0:
        txt(img, f"Angle: {rep_counter.tracking_angle:.0f}", (12, y_cursor), scale=0.44, color=LGREY)
        y_cursor += 18
        hint = rep_counter.threshold_hint
        if hint:
            txt(img, hint, (12, y_cursor), scale=0.36, color=CYAN)
            y_cursor += 16

    view_label = (angles or {}).get("Camera_View", "")
    if view_label:
        txt(img, f"View: {view_label.upper()}", (12, y_cursor), scale=0.36, color=LGREY)
        y_cursor += 14

    for aname, val in angle_items[:4]:
        lo, hi, _ = rules.get(aname, (0, 180, ""))
        ok = lo <= val <= hi
        acol = GREEN if ok else RED
        label = aname.replace("_Angle", "")
        txt(img, f"{label}: {val:.0f}", (12, y_cursor), scale=0.40, color=acol)
        y_cursor += 16

    y_cursor += 6
    txt(img, f"Score: {form_score.total:.0f}/100", (12, y_cursor), scale=0.42, color=CYAN)
    y_cursor += 18

    if issues:
        for _, reason, val, lo, hi in issues[:2]:
            short = reason[:32] + ".." if len(reason) > 34 else reason
            txt(img, short, (12, y_cursor), scale=0.36, color=ORANGE)
            y_cursor += 16

    txt(img, f"FPS: {fps:.0f}", (12, h - 28), scale=0.38, color=LGREY)
    txt(img, "Q/SPACE = stop", (12, h - 12), scale=0.36, color=LGREY)


def draw_exercise_menu(
    frame,
    h,
    w,
    menu_items: List[Tuple[str, str]],
    registry: ExerciseRegistry,
    selection_buffer: str = "",
    scroll_offset: int = 0,
    items_per_page: int = 18,
):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    panel_w = min(760, w - 40)
    panel_h = min(h - 40, 620)
    px = (w - panel_w) // 2
    py = 20
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (28, 28, 36), -1)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), CYAN, 2)

    txt(frame, "BioVision AI", (px + 24, py + 42), scale=1.0, color=CYAN, bold=True)
    txt(frame, "Select Exercise", (px + 24, py + 72), scale=0.55, color=LGREY)
    txt(frame, "Type number + ENTER  |  BACKSPACE to edit  |  UP/DOWN to scroll", (px + 24, py + 96), scale=0.42, color=LGREY)

    input_y = py + 118
    cv2.rectangle(frame, (px + 24, input_y - 28), (px + panel_w - 24, input_y + 8), (45, 45, 55), -1)
    cv2.rectangle(frame, (px + 24, input_y - 28), (px + panel_w - 24, input_y + 8), CYAN, 1)
    display_buf = selection_buffer if selection_buffer else "_"
    txt(frame, f"Exercise #: {display_buf}", (px + 36, input_y - 6), scale=0.62, color=WHITE, bold=True)

    visible = menu_items[scroll_offset: scroll_offset + items_per_page]
    col_w = (panel_w - 48) // 2
    row_h = 34
    start_y = py + 140
    for i, (key, name) in enumerate(visible):
        col = i % 2
        row = i // 2
        x = px + 24 + col * col_w
        y = start_y + row * row_h
        cfg = registry.get(name)
        label = cfg.display_name if cfg else name.replace("_", " ")
        is_match = selection_buffer and key.startswith(selection_buffer)
        is_exact = key == selection_buffer
        bg = (55, 90, 55) if is_exact else ((45, 65, 45) if is_match else (38, 38, 48))
        fg = YELLOW if is_exact else (WHITE if is_match else LGREY)
        cv2.rectangle(frame, (x, y - 20), (x + col_w - 12, y + 8), bg, -1)
        cv2.rectangle(frame, (x, y - 20), (x + col_w - 12, y + 8), CYAN if is_exact else (70, 70, 80), 1)
        txt(frame, f"{key:>2}. {label}", (x + 10, y), scale=0.48, color=fg, bold=is_exact)

    total_pages = max(1, (len(menu_items) + items_per_page - 1) // items_per_page)
    current_page = scroll_offset // items_per_page + 1
    txt(
        frame,
        f"Page {current_page}/{total_pages}  ({len(menu_items)} exercises)",
        (px + 24, py + panel_h - 36),
        scale=0.42,
        color=LGREY,
    )
    txt(frame, "Q = Quit", (px + panel_w - 120, py + panel_h - 36), scale=0.42, color=LGREY)


def _menu_keys(menu_items: List[Tuple[str, str]]) -> List[str]:
    return [key for key, _ in menu_items]


def _has_longer_prefix(buffer: str, keys: List[str]) -> bool:
    return any(k.startswith(buffer) and len(k) > len(buffer) for k in keys)


def _resolve_menu_selection(buffer: str, menu_map: Dict[str, str]) -> Optional[str]:
    if buffer in menu_map:
        return menu_map[buffer]
    return None


def save_session(
    session_log,
    exercise,
    session_dir: Path,
    analyzer: SessionAnalyzer,
    kinematics_tracker,
    form_scorer,
    registry,
    landmarks_log=None,
):
    if not session_log:
        logger.warning("No data to save.")
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(session_log)
    base = session_dir / f"{exercise}_{ts}"
    df.to_csv(base.with_suffix(".csv"), index=False)
    cfg = registry.get(exercise)
    primary_keys = ["Elbow_Angle", "Knee_Angle", "Hip_Angle", "Shoulder_Angle"]
    kin_summary = kinematics_tracker.summarize(
        primary_keys=primary_keys,
        bilateral_pairs={
            "Left_Elbow_Flexion": "Right_Elbow_Flexion",
            "Left_Knee_Flexion": "Right_Knee_Flexion",
        },
    )
    last_angles = {k: v for k, v in df.iloc[-1].items() if isinstance(v, (int, float))}
    score = form_scorer.compute(exercise, last_angles, kin_summary)
    analytics = analyzer.analyze(session_log, exercise, score, kin_summary)
    summary = analytics.to_summary_dict()
    landmarks_df = pd.DataFrame(landmarks_log) if landmarks_log else None
    excel_path = base.with_suffix(".xlsx")
    try:
        export_session_excel(df, str(excel_path), summary, landmarks_df=landmarks_df)
        logger.info("Excel saved: %s", excel_path)
    except Exception as exc:
        logger.warning("Excel export skipped: %s", exc)
    export_analytics_report(analytics, str(base.with_name(base.name + "_analytics.txt")))
    logger.info("Session saved: %s (%d frames)", base, len(df))


def run_live(enable_3d: bool = False) -> None:
    settings = load_settings()
    pose_cfg = settings.get("pose", {})
    infer_cfg = settings.get("inference", {})
    min_vis = float(pose_cfg.get("min_visibility", 0.30))
    smooth_n = int(infer_cfg.get("smooth_window", 8))
    model_dir = get_path("model_dir", "models")
    session_dir = get_path("session_dir", "session_results")
    session_dir.mkdir(parents=True, exist_ok=True)

    clf, encoder, scaler, feature_cols = load_models(model_dir)
    if clf is None:
        return

    registry = ExerciseRegistry()
    form_scorer = FormScorer(registry)
    analyzer = SessionAnalyzer()
    menu_items = registry.menu_items()
    menu_map = {k: v for k, v in menu_items}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error(
            "Cannot open webcam. Grant camera access in "
            "System Settings > Privacy & Security > Camera."
        )
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(infer_cfg.get("webcam_width", 1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(infer_cfg.get("webcam_height", 720)))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("BioVision AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BioVision AI", 1280, 720)

    warmup_ok = False
    for _ in range(60):
        ret, _frame = cap.read()
        if ret and _frame is not None and _frame.size > 0:
            warmup_ok = True
            break
        time.sleep(0.05)
    if not warmup_ok:
        logger.error(
            "Webcam opened but no frames received. Close other apps using the "
            "camera, then try again."
        )
        cap.release()
        return

    logger.info("BioVision AI ready — select an exercise from the menu.")
    read_fail_streak = 0

    viewer = Skeleton3DViewer(min_visibility=min_vis)
    html_3d_path = session_dir / "skeleton_3d_live.html"
    if enable_3d:
        if not viewer.start():
            logger.warning("Plotly not available — 3D view disabled.")
            enable_3d = False

    current_exercise = None
    rep_counter = None
    kinematics_tracker = KinematicTracker()
    angle_history: Deque[Dict[str, float]] = deque(maxlen=60)
    pred_buf: Deque[int] = deque(maxlen=smooth_n)
    conf_buf: Deque[float] = deque(maxlen=smooth_n)
    session_log: List[Dict] = []
    landmarks_log: List[Dict] = []
    prev_time = time.time()
    frame_n = 0
    frame_3d_counter = 0
    selection_buffer = ""
    selection_updated_at = 0.0
    menu_scroll = 0
    menu_keys = _menu_keys(menu_items)
    items_per_page = 18
    selection_timeout_s = 0.85

    with PoseProcessor(
        min_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
        model_complexity=int(pose_cfg.get("model_complexity", 1)),
        min_visibility=min_vis,
        use_3d=True,
        include_posture=True,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                read_fail_streak += 1
                if read_fail_streak > 60:
                    logger.error("Webcam stopped delivering frames.")
                    break
                time.sleep(0.03)
                continue
            read_fail_streak = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if current_exercise is None:
                menu_result = pose.process_frame(frame)
                if menu_result is not None and menu_result.image_landmarks is not None:
                    draw_skeleton_colored(
                        frame,
                        menu_result.image_landmarks,
                        {},
                        {},
                        w,
                        h,
                        min_vis,
                        is_good=True,
                    )

                draw_exercise_menu(
                    frame, h, w, menu_items, registry,
                    selection_buffer=selection_buffer,
                    scroll_offset=menu_scroll,
                    items_per_page=items_per_page,
                )
                cv2.imshow("BioVision AI", frame)
                key = cv2.waitKey(1) & 0xFF

                if (
                    selection_buffer
                    and selection_updated_at > 0
                    and time.time() - selection_updated_at >= selection_timeout_s
                ):
                    resolved = _resolve_menu_selection(selection_buffer, menu_map)
                    if resolved:
                        current_exercise = resolved
                        rep_counter = RepCounter(current_exercise, registry)
                        kinematics_tracker.reset()
                        angle_history.clear()
                        pred_buf.clear()
                        conf_buf.clear()
                        session_log = []
                        landmarks_log = []
                        frame_n = 0
                        selection_buffer = ""
                        logger.info("Exercise: %s", current_exercise)
                    else:
                        selection_buffer = ""

                if key == ord("q"):
                    logger.info("Quit requested from menu.")
                    break
                if key in (13, 10):  # Enter
                    resolved = _resolve_menu_selection(selection_buffer, menu_map)
                    if resolved:
                        current_exercise = resolved
                        rep_counter = RepCounter(current_exercise, registry)
                        kinematics_tracker.reset()
                        angle_history.clear()
                        pred_buf.clear()
                        conf_buf.clear()
                        session_log = []
                        landmarks_log = []
                        frame_n = 0
                        selection_buffer = ""
                        logger.info("Exercise: %s", current_exercise)
                elif key in (8, 127):  # Backspace / Delete
                    selection_buffer = selection_buffer[:-1]
                    selection_updated_at = time.time() if selection_buffer else 0.0
                elif key in (ord("w"), 82):  # up arrow
                    menu_scroll = max(0, menu_scroll - items_per_page)
                elif key in (ord("s"), 84):  # down arrow
                    max_scroll = max(0, len(menu_items) - items_per_page)
                    menu_scroll = min(max_scroll, menu_scroll + items_per_page)
                elif ord("0") <= key <= ord("9"):
                    char = chr(key)
                    if not selection_buffer and char in menu_map and not _has_longer_prefix(char, menu_keys):
                        current_exercise = menu_map[char]
                        rep_counter = RepCounter(current_exercise, registry)
                        kinematics_tracker.reset()
                        angle_history.clear()
                        pred_buf.clear()
                        conf_buf.clear()
                        session_log = []
                        landmarks_log = []
                        frame_n = 0
                        selection_buffer = ""
                        logger.info("Exercise: %s", current_exercise)
                    else:
                        selection_buffer += char
                        selection_updated_at = time.time()
                        if selection_buffer in menu_map and not _has_longer_prefix(selection_buffer, menu_keys):
                            current_exercise = menu_map[selection_buffer]
                            rep_counter = RepCounter(current_exercise, registry)
                            kinematics_tracker.reset()
                            angle_history.clear()
                            pred_buf.clear()
                            conf_buf.clear()
                            session_log = []
                            landmarks_log = []
                            frame_n = 0
                            selection_buffer = ""
                            logger.info("Exercise: %s", current_exercise)
                continue

            frame_n += 1
            result = pose.process_frame(frame)
            angles = None
            side_used = None
            is_good = True
            issues = []
            prediction = 1
            confidence = 0.5
            lms = None
            world_lms = None
            form_score = form_scorer.compute(current_exercise, {})

            kin_summary = None
            if result is not None:
                angles = result.angles or {}
                side_used = result.side_used
                lms = result.image_landmarks
                world_lms = result.world_landmarks or result.landmarks
                rules = registry.legacy_rules(current_exercise)

                if rep_counter is not None:
                    rep_counter.update(
                        angles,
                        image_landmarks=lms,
                        side_used=side_used,
                        min_visibility=max(0.12, min_vis * 0.5),
                    )

                if lms is not None:
                    landmarks_log.append(
                        snapshot_landmarks(frame_n, result.world_landmarks, lms, current_exercise)
                    )

                if angles:
                    angle_history.append(dict(angles))
                    kinematics_tracker.update(time.time(), angles)
                    is_good, form_issues = check_form(
                        angles, current_exercise, world_lms, list(angle_history), registry
                    )
                    issues = issues_to_legacy_tuples(form_issues)
                    kin_summary = kinematics_tracker.summarize(
                        primary_keys=["Elbow_Angle", "Knee_Angle", "Hip_Angle"],
                    )
                    form_score = form_scorer.compute(
                        current_exercise, angles, kin_summary, form_issues, world_lms
                    )
                    feat = build_feature_vector(angles, current_exercise, encoder, scaler, feature_cols)
                    pred = clf.predict(feat)[0]
                    proba = clf.predict_proba(feat)[0]
                    pred_buf.append(int(pred))
                    conf_buf.append(float(proba[pred]))
                    prediction = int(round(np.mean(pred_buf)))
                    confidence = float(np.mean(conf_buf))

                if lms is not None:
                    draw_skeleton_colored(
                        frame, lms, angles, rules, w, h, min_vis, is_good=is_good,
                    )

                if rep_counter is not None and lms is not None:
                    kin = kin_summary or kinematics_tracker.summarize(
                        primary_keys=["Elbow_Angle", "Knee_Angle", "Hip_Angle"],
                    )
                    session_log.append(
                        {
                            "Frame": frame_n,
                            "Exercise": current_exercise,
                            "Elbow_Angle": angles.get("Elbow_Angle"),
                            "Shoulder_Angle": angles.get("Shoulder_Angle"),
                            "Hip_Angle": angles.get("Hip_Angle"),
                            "Knee_Angle": angles.get("Knee_Angle"),
                            "Trunk_Inclination": angles.get("Trunk_Inclination"),
                            "Rep_Tracking_Angle": rep_counter.tracking_angle,
                            "Rep_Stage": rep_counter.stage,
                            "Form": "Good" if is_good else "Bad",
                            "ML_Prediction": "Good" if prediction == 1 else "Bad",
                            "Confidence_Pct": round(confidence * 100, 1),
                            "Reps": rep_counter.count,
                            "Issues": "; ".join(i[1] for i in issues),
                            **form_score.to_dict(),
                            "Stability": kin.stability_score,
                            "Symmetry": kin.symmetry_score,
                            "Smoothness": kin.smoothness_score,
                        }
                    )

                exercise_cfg = registry.get(current_exercise)
                exercise_label = exercise_cfg.display_name if exercise_cfg else current_exercise
                if enable_3d and world_lms is not None:
                    frame_3d_counter += 1
                    if frame_3d_counter % 2 == 0:
                        viewer.update(
                            world_lms,
                            angles or {},
                            issues,
                            side_used or "RIGHT",
                            rep_count=rep_counter.count if rep_counter else 0,
                            exercise_name=exercise_label,
                        )
                        viewer.render_to_html(str(html_3d_path), auto_open=(frame_3d_counter == 2))

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            coord_lines = None
            if enable_3d and result is not None:
                # Prefer MediaPipe world coords (meters). Fallback to normalized image coords.
                coord_src = result.world_landmarks or result.image_landmarks
                suffix = "m" if result.world_landmarks else "norm"
                try:
                    lw = coord_src[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    rw = coord_src[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    lh = coord_src[mp_pose.PoseLandmark.LEFT_HIP.value]
                    coord_lines = [
                        f"L.Wrist: ({lw.x:+.3f}, {lw.y:+.3f}, {lw.z:+.3f}) {suffix}",
                        f"R.Wrist: ({rw.x:+.3f}, {rw.y:+.3f}, {rw.z:+.3f}) {suffix}",
                        f"L.Hip:   ({lh.x:+.3f}, {lh.y:+.3f}, {lh.z:+.3f}) {suffix}",
                    ]
                except Exception:
                    coord_lines = None

            draw_left_panel(
                frame, current_exercise, is_good, confidence, issues,
                angles, rep_counter, fps, h, form_score, registry, min_vis,
                coord_lines=coord_lines,
            )
            cv2.imshow("BioVision AI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord(" ")):
                save_session(
                    session_log, current_exercise, session_dir, analyzer,
                    kinematics_tracker, form_scorer, registry, landmarks_log,
                )
                current_exercise = None
                rep_counter = None
                landmarks_log = []

    if session_log and current_exercise:
        save_session(
            session_log, current_exercise, session_dir, analyzer,
            kinematics_tracker, form_scorer, registry, landmarks_log,
        )

    cap.release()
    cv2.destroyAllWindows()
    viewer.close()
    logger.info("Session ended.")


def parse_args():
    parser = argparse.ArgumentParser(description="BioVision AI live inference")
    parser.add_argument("--3d-view", dest="enable_3d", action="store_true", help="Enable 3D skeleton view")
    return parser.parse_args()


def main():
    args = parse_args()
    run_live(enable_3d=args.enable_3d)


if __name__ == "__main__":
    main()
