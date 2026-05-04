import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

INPUT_CSV    = "gym_dataset.csv"
OUTPUT_EXCEL = "form_analysis_report.xlsx"
OUTPUT_TXT   = "form_analysis_report.txt"

RULES = {
    "bicep_curl": {
        "Elbow_Angle"    : (20,  160, "Elbow not fully curling or not fully extending — incomplete range of motion"),
        "Shoulder_Angle" : (0,   40,  "Upper arm swinging away from body — using momentum instead of muscle"),
        "Hip_Angle"      : (160, 180, "Torso leaning back — using body swing to lift the weight"),
        "Knee_Angle"     : (160, 180, "Knees bending — unstable base, increases injury risk"),
    },
    "pushup": {
        "Elbow_Angle"    : (60,  170, "Elbow not reaching 90° at bottom or not fully extending at top"),
        "Shoulder_Angle" : (30,  90,  "Shoulders flaring too wide or collapsing inward"),
        "Hip_Angle"      : (160, 180, "Hips sagging or piking — body not in a straight plank line"),
        "Knee_Angle"     : (160, 180, "Knees bent — not maintaining full plank position"),
    },
    "plank": {
        "Elbow_Angle"    : (80,  100, "Elbows not at 90° — incorrect forearm plank position"),
        "Shoulder_Angle" : (80,  100, "Shoulders not stacked over elbows — misaligned upper body"),
        "Hip_Angle"      : (160, 180, "Hips sagging down or piking up — body not in a straight line"),
        "Knee_Angle"     : (160, 180, "Knees bent — legs should be fully extended in a plank"),
    },
    "squat": {
        "Elbow_Angle"    : (60,  180, "Arms not in a stable position during squat"),
        "Shoulder_Angle" : (60,  180, "Torso collapsing forward — excessive forward lean"),
        "Hip_Angle"      : (50,  120, "Not reaching parallel depth or collapsing too deep with bad spine"),
        "Knee_Angle"     : (60,  120, "Knees not bending enough or caving inward past toes"),
    },
    "deadlift": {
        "Elbow_Angle"    : (160, 180, "Arms bent during lift — should be straight, not pulling with arms"),
        "Shoulder_Angle" : (60,  180, "Shoulders rounding forward — high injury risk to upper back"),
        "Hip_Angle"      : (60,  160, "Hips rising too fast (stiff-leg) or not hinging properly"),
        "Knee_Angle"     : (100, 170, "Knees locking out too early or not enough knee bend at start"),
    },
}

H_FILL    = PatternFill("solid", fgColor="1F4E79")
GOOD_FILL = PatternFill("solid", fgColor="C6EFCE")
BAD_FILL  = PatternFill("solid", fgColor="FFCCCC")
WARN_FILL = PatternFill("solid", fgColor="FFEB9C")
ALT_FILL  = PatternFill("solid", fgColor="F2F2F2")
H_FONT    = Font(bold=True, color="FFFFFF", size=11)
THIN      = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"),  bottom=Side(style="thin")
)
CENTER    = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT      = Alignment(horizontal="left",   vertical="center", wrap_text=True)


def hcell(ws, row, col, value, width=None):
    c = ws.cell(row=row, column=col, value=value)
    c.fill = H_FILL; c.font = H_FONT; c.alignment = CENTER; c.border = THIN
    if width:
        ws.column_dimensions[get_column_letter(col)].width = width
    return c


def dcell(ws, row, col, value, fill=None, bold=False, align=CENTER):
    c = ws.cell(row=row, column=col, value=value)
    if fill: c.fill = fill
    c.font = Font(bold=bold, size=10)
    c.alignment = align
    c.border = THIN
    return c


def get_rules(exercise: str) -> dict:
    key = exercise.lower().replace(" ", "_").replace("-", "_")
    for k in RULES:
        if k in key or key in k:
            return RULES[k]
    return {
        "Elbow_Angle"    : (10,  170, "Elbow angle out of expected range"),
        "Shoulder_Angle" : (0,   180, "Shoulder angle out of expected range"),
        "Hip_Angle"      : (50,  180, "Hip angle out of expected range"),
        "Knee_Angle"     : (50,  180, "Knee angle out of expected range"),
    }


def analyse_group(df_video: pd.DataFrame, exercise: str) -> dict:
    rules  = get_rules(exercise)
    issues = {}
    for angle, (lo, hi, reason) in rules.items():
        bad = df_video[(df_video[angle] < lo) | (df_video[angle] > hi)]
        if not bad.empty:
            issues[angle] = {
                "reason"    : reason,
                "range"     : (lo, hi),
                "bad_count" : len(bad),
                "total"     : len(df_video),
                "pct"       : round(100 * len(bad) / len(df_video), 1),
                "mean"      : round(df_video[angle].mean(), 1),
                "frames"    : bad[["Frame_Number", angle]].values.tolist(),
            }
    return issues


def build_excel(df: pd.DataFrame):
    wb = Workbook()

    ws = wb.active
    ws.title = "Overview"
    ws.row_dimensions[1].height = 24
    ws.freeze_panes = "A2"

    ov_cols = [
        ("Exercise", 18), ("Video File", 38), ("Form", 8),
        ("Frames Analysed", 16), ("Issues Found", 14), ("Verdict", 30),
    ]
    for i, (name, w) in enumerate(ov_cols, 1):
        hcell(ws, 1, i, name, w)

    data_row = 2
    for (exercise, video_file, form_label), grp in df.groupby(
        ["Exercise", "Video_File", "Form_Label"]
    ):
        issues  = analyse_group(grp, exercise)
        verdict = "✅ Good Form" if form_label == 1 else (
            "❌ Bad Form — " + "; ".join(issues.keys()) if issues else "❌ Bad Form"
        )
        fill    = GOOD_FILL if form_label == 1 else BAD_FILL

        vals = [exercise, video_file, "Good" if form_label == 1 else "Bad",
                len(grp), len(issues), verdict]
        for col, val in enumerate(vals, 1):
            dcell(ws, data_row, col, val, fill=fill,
                  bold=(col in (3, 5, 6)),
                  align=LEFT if col == 6 else CENTER)
        ws.row_dimensions[data_row].height = 18
        data_row += 1

    ws2 = wb.create_sheet("Detailed Issues")
    ws2.freeze_panes = "A2"
    ws2.row_dimensions[1].height = 24

    det_cols = [
        ("Exercise", 18), ("Video File", 36), ("Form", 8),
        ("Joint Angle", 15), ("Acceptable Range", 18), ("Your Average", 13),
        ("Bad Frames", 12), ("Total Frames", 13), ("% Bad", 8),
        ("Why It's Bad", 55),
    ]
    for i, (name, w) in enumerate(det_cols, 1):
        hcell(ws2, 1, i, name, w)

    det_row = 2
    for (exercise, video_file, form_label), grp in df.groupby(
        ["Exercise", "Video_File", "Form_Label"]
    ):
        issues = analyse_group(grp, exercise)
        if not issues:
            continue
        fill = GOOD_FILL if form_label == 1 else BAD_FILL
        for angle, info in issues.items():
            vals = [
                exercise, video_file, "Good" if form_label == 1 else "Bad",
                angle,
                f"{info['range'][0]}° – {info['range'][1]}°",
                f"{info['mean']}°",
                info["bad_count"], info["total"],
                f"{info['pct']}%",
                info["reason"],
            ]
            for col, val in enumerate(vals, 1):
                dcell(ws2, det_row, col, val,
                      fill=fill if col in (1, 2, 3) else (WARN_FILL if col == 9 else None),
                      bold=(col == 4),
                      align=LEFT if col == 10 else CENTER)
            ws2.row_dimensions[det_row].height = 30
            det_row += 1

    ws3 = wb.create_sheet("Bad Frame Timestamps")
    ws3.freeze_panes = "A2"
    ws3.row_dimensions[1].height = 24

    ts_cols = [
        ("Exercise", 18), ("Video File", 36), ("Joint Angle", 15),
        ("Frame Number", 14), ("Angle Value", 12), ("Acceptable Range", 18),
        ("Deviation", 12),
    ]
    for i, (name, w) in enumerate(ts_cols, 1):
        hcell(ws3, 1, i, name, w)

    ts_row = 2
    for (exercise, video_file, form_label), grp in df.groupby(
        ["Exercise", "Video_File", "Form_Label"]
    ):
        if form_label == 1:
            continue   
        issues = analyse_group(grp, exercise)
        for angle, info in issues.items():
            lo, hi = info["range"]
            for frame_num, angle_val in info["frames"]:
                deviation = round(
                    min(abs(angle_val - lo), abs(angle_val - hi)), 1
                )
                vals = [exercise, video_file, angle,
                        int(frame_num), round(angle_val, 1),
                        f"{lo}° – {hi}°", f"{deviation}°"]
                for col, val in enumerate(vals, 1):
                    dcell(ws3, ts_row, col, val,
                          fill=BAD_FILL if col in (4, 5, 7) else None,
                          bold=(col == 7),
                          align=CENTER)
                ws3.row_dimensions[ts_row].height = 16
                ts_row += 1

    ws4 = wb.create_sheet("Summary Stats")
    ws4.freeze_panes = "A2"
    ws4.row_dimensions[1].height = 24

    st_cols = [
        ("Exercise", 18), ("Form", 8), ("Total Frames", 14),
        ("Avg Elbow°", 12), ("Avg Shoulder°", 14),
        ("Avg Hip°", 10), ("Avg Knee°", 10),
        ("Min Elbow°", 12), ("Max Elbow°", 12),
    ]
    for i, (name, w) in enumerate(st_cols, 1):
        hcell(ws4, 1, i, name, w)

    summary = (
        df.groupby(["Exercise", "Form"])
        .agg(
            Total_Frames   = ("Frame_Number", "count"),
            Avg_Elbow      = ("Elbow_Angle",    "mean"),
            Avg_Shoulder   = ("Shoulder_Angle", "mean"),
            Avg_Hip        = ("Hip_Angle",       "mean"),
            Avg_Knee       = ("Knee_Angle",      "mean"),
            Min_Elbow      = ("Elbow_Angle",    "min"),
            Max_Elbow      = ("Elbow_Angle",    "max"),
        )
        .round(1)
        .reset_index()
    )

    for i, row in enumerate(summary.itertuples(index=False), start=2):
        fill = GOOD_FILL if row.Form == "Good" else BAD_FILL
        vals = [row.Exercise, row.Form, row.Total_Frames,
                row.Avg_Elbow, row.Avg_Shoulder, row.Avg_Hip, row.Avg_Knee,
                row.Min_Elbow, row.Max_Elbow]
        for col, val in enumerate(vals, 1):
            dcell(ws4, i, col, val, fill=fill, bold=(col <= 2), align=CENTER)
        ws4.row_dimensions[i].height = 18

    wb.save(OUTPUT_EXCEL)
    print(f"📊  Excel report saved → {OUTPUT_EXCEL}")

def build_text(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  BIOVISION AI — FORM ANALYSIS REPORT")
    lines.append("=" * 72)

    for (exercise, video_file, form_label), grp in df.groupby(
        ["Exercise", "Video_File", "Form_Label"]
    ):
        verdict = "✅  GOOD FORM" if form_label == 1 else "❌  BAD FORM"
        lines.append(f"\nExercise  : {exercise}")
        lines.append(f"Video     : {video_file}")
        lines.append(f"Verdict   : {verdict}")
        lines.append(f"Frames    : {len(grp)} motion frames analysed")

        issues = analyse_group(grp, exercise)

        if not issues:
            lines.append("  → All joint angles within acceptable range.")
        else:
            lines.append("\n  ── Issues ──────────────────────────────────")
            for angle, info in issues.items():
                lines.append(f"\n  [{angle}]")
                lines.append(f"    Why       : {info['reason']}")
                lines.append(f"    Range     : {info['range'][0]}° – {info['range'][1]}°")
                lines.append(f"    Average   : {info['mean']}°")
                lines.append(f"    Bad frames: {info['bad_count']} / {info['total']} ({info['pct']}%)")
                sample = info["frames"][:8]
                ts = ", ".join(f"frame {int(f)} ({v}°)" for f, v in sample)
                lines.append(f"    When      : {ts}" + (" ..." if len(info["frames"]) > 8 else ""))

        lines.append("\n" + "-" * 72)

    return "\n".join(lines)


def main():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌  {INPUT_CSV} not found. Run create_dataset.py first.")
        return

    if df.empty:
        print("❌  Dataset is empty.")
        return

    print(f"📂  Loaded {len(df)} rows from {INPUT_CSV}\n")

    build_excel(df)

    report = build_text(df)
    print(report)
    with open(OUTPUT_TXT, "w") as f:
        f.write(report)
    print(f"\n📄  Text report saved → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
