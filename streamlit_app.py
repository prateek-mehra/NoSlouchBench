from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noslouchbench.habits_report import read_report_rows, write_habits_reports  # noqa: E402


st.set_page_config(page_title="NoSlouchBench Reports", page_icon="📊", layout="wide")
st.title("NoSlouchBench Habit Reports")
st.caption("Local dashboard for daily and weekly posture stats")

with st.sidebar:
    st.header("Data Sources")
    summaries_dir = Path(st.text_input("Summaries directory", value="outputs/summaries"))
    sessions_dir = Path(st.text_input("Sessions directory", value="outputs/sessions"))
    reports_dir = Path(st.text_input("Reports directory", value="outputs/reports"))
    timezone_name = st.text_input("Timezone", value="local")
    refresh = st.button("Refresh Reports", type="primary")

if refresh:
    try:
        artifacts = write_habits_reports(
            summaries_dir=summaries_dir,
            sessions_dir=sessions_dir,
            output_dir=reports_dir,
            timezone_name=timezone_name,
        )
        st.success(
            "Generated reports: "
            f"{artifacts['daily_csv'].name}, {artifacts['weekly_csv'].name}, "
            f"{artifacts['daily_md'].name}, {artifacts['weekly_md'].name}"
        )
    except Exception as exc:
        st.error(f"Failed to refresh reports: {exc}")

daily_rows, weekly_rows = read_report_rows(reports_dir)

if not daily_rows and not weekly_rows:
    st.info("No report files found yet. Click 'Refresh Reports' in the sidebar.")
else:
    latest_daily = daily_rows[-1] if daily_rows else None
    latest_weekly = weekly_rows[-1] if weekly_rows else None

    col1, col2, col3, col4 = st.columns(4)
    if latest_daily:
        col1.metric("Daily Sitting Hours", f"{latest_daily['hours_sitting']:.2f}")
        col2.metric("Daily Slouch Minutes", f"{latest_daily['slouch_minutes']:.1f}")
    else:
        col1.metric("Daily Sitting Hours", "-")
        col2.metric("Daily Slouch Minutes", "-")

    if latest_weekly:
        col3.metric("Weekly Sitting Hours", f"{latest_weekly['hours_sitting']:.2f}")
        col4.metric("Weekly Beep Events", f"{int(latest_weekly['beep_events'])}")
    else:
        col3.metric("Weekly Sitting Hours", "-")
        col4.metric("Weekly Beep Events", "-")

    tab_daily, tab_weekly, tab_files = st.tabs(["Daily", "Weekly", "Files"])

    with tab_daily:
        st.subheader("Daily Stats")
        if not daily_rows:
            st.write("No daily rows available.")
        else:
            st.dataframe(daily_rows, use_container_width=True)

    with tab_weekly:
        st.subheader("Weekly Stats")
        if not weekly_rows:
            st.write("No weekly rows available.")
        else:
            st.dataframe(weekly_rows, use_container_width=True)

    with tab_files:
        st.subheader("Generated Files")
        for name in ["habits_daily.md", "habits_weekly.md", "habits_daily.csv", "habits_weekly.csv"]:
            p = reports_dir / name
            if p.exists():
                st.write(f"- {p}")
            else:
                st.write(f"- Missing: {p}")

        daily_md = reports_dir / "habits_daily.md"
        weekly_md = reports_dir / "habits_weekly.md"
        if daily_md.exists():
            with st.expander("Preview habits_daily.md"):
                st.markdown(daily_md.read_text(encoding="utf-8"))
        if weekly_md.exists():
            with st.expander("Preview habits_weekly.md"):
                st.markdown(weekly_md.read_text(encoding="utf-8"))
