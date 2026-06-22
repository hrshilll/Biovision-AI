"""
BioVision AI — Workout Analytics Dashboard
Run: streamlit run src/dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

SESSION_DIR = PROJECT_ROOT / "session_results"


@st.cache_data
def load_session_files() -> list[Path]:
    if not SESSION_DIR.exists():
        return []
    return sorted(SESSION_DIR.glob("*.csv"), reverse=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def page_session_summary(df: pd.DataFrame) -> None:
    st.subheader("Session Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", len(df))
    c2.metric("Reps", int(df["Reps"].max()) if "Reps" in df.columns else 0)
    c3.metric(
        "Avg Form Score",
        f"{df['Form_Score'].mean():.1f}" if "Form_Score" in df.columns else "N/A",
    )
    c4.metric(
        "Good Form %",
        f"{(df['Form'] == 'Good').mean() * 100:.1f}%"
        if "Form" in df.columns
        else "N/A",
    )
    st.dataframe(df.head(100), use_container_width=True)


def page_form_analysis(df: pd.DataFrame) -> None:
    st.subheader("Form Analysis")
    score_cols = [c for c in df.columns if c.endswith("_Score")]
    if score_cols:
        st.bar_chart(df[score_cols].mean())
    if "Form" in df.columns:
        st.write("Form distribution")
        st.bar_chart(df["Form"].value_counts())


def page_exercise_trends(all_dfs: list[pd.DataFrame]) -> None:
    st.subheader("Exercise Trends")
    if not all_dfs:
        st.info("No session data available.")
        return
    combined = pd.concat(all_dfs, ignore_index=True)
    if "Exercise" in combined.columns and "Form_Score" in combined.columns:
        trend = combined.groupby("Exercise")["Form_Score"].mean().sort_values(ascending=False)
        st.bar_chart(trend)
    elif "Exercise" in combined.columns:
        st.bar_chart(combined["Exercise"].value_counts())


def page_error_breakdown(df: pd.DataFrame) -> None:
    st.subheader("Error Breakdown")
    if "Issues" not in df.columns and "Error_Codes" not in df.columns:
        st.info("No error data in this session.")
        return
    col = "Issues" if "Issues" in df.columns else "Error_Codes"
    errors = []
    for val in df[col].dropna():
        for part in str(val).split(";"):
            msg = part.strip()
            if msg:
                errors.append(msg)
    if not errors:
        st.success("No errors detected in this session.")
        return
    err_df = pd.Series(errors).value_counts().reset_index()
    err_df.columns = ["Error", "Count"]
    st.dataframe(err_df, use_container_width=True)
    st.bar_chart(err_df.set_index("Error")["Count"])


def page_rom_analysis(df: pd.DataFrame) -> None:
    st.subheader("ROM Analysis")
    rom_cols = [c for c in df.columns if c.startswith("ROM_")]
    angle_cols = [c for c in df.columns if "Angle" in c or "Flexion" in c]
    cols = rom_cols or angle_cols[:4]
    if not cols:
        st.info("No ROM or angle columns found.")
        return
    summary = df[cols].agg(["min", "max", "mean"]).T
    summary["ROM"] = summary["max"] - summary["min"]
    st.dataframe(summary, use_container_width=True)
    st.line_chart(df[cols[:4]])


def main() -> None:
    st.set_page_config(page_title="BioVision AI Dashboard", layout="wide")
    st.title("BioVision AI — Workout Analytics Dashboard")

    files = load_session_files()
    if not files:
        st.warning(f"No session CSV files found in `{SESSION_DIR}`. Run live inference first.")
        return

    selected = st.sidebar.selectbox("Session file", files, format_func=lambda p: p.name)
    df = load_csv(selected)
    all_dfs = [load_csv(f) for f in files[:20]]

    page = st.sidebar.radio(
        "Page",
        [
            "Session Summary",
            "Form Analysis",
            "Exercise Trends",
            "Error Breakdown",
            "ROM Analysis",
        ],
    )

    if page == "Session Summary":
        page_session_summary(df)
    elif page == "Form Analysis":
        page_form_analysis(df)
    elif page == "Exercise Trends":
        page_exercise_trends(all_dfs)
    elif page == "Error Breakdown":
        page_error_breakdown(df)
    elif page == "ROM Analysis":
        page_rom_analysis(df)


if __name__ == "__main__":
    main()
