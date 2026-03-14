from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noslouchbench.habits_report import reports_need_refresh, resolve_timezone, write_habits_reports  # noqa: E402

SUMMARIES_DIR = Path("outputs/summaries")
SESSIONS_DIR = Path("outputs/sessions")
REPORTS_DIR = Path("outputs/reports")
INSTANCES_DIR = Path("outputs/slouch_instances")
TIMEZONE_NAME = "local"
BLOCKED_INSTANCE_MIN_DURATION_SECONDS = 2.0


def load_slouch_instances(instances_dir: Path, timezone_name: str) -> list[dict]:
    tzinfo = resolve_timezone(timezone_name)
    items: list[dict] = []

    for path in sorted(instances_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue

        for row in raw:
            if not isinstance(row, dict):
                continue
            start_ts = row.get("start_timestamp_utc")
            end_ts = row.get("end_timestamp_utc")
            image_path = row.get("image_path")
            duration_seconds = row.get("duration_seconds", 0.0)
            if not start_ts:
                continue
            try:
                duration_seconds = float(duration_seconds)
            except (TypeError, ValueError):
                continue
            if duration_seconds <= BLOCKED_INSTANCE_MIN_DURATION_SECONDS:
                # Show only instances that crossed the blocker-gating duration.
                continue

            try:
                start_dt_utc = datetime.fromisoformat(str(start_ts))
                if start_dt_utc.tzinfo is None:
                    continue
                start_dt_local = start_dt_utc.astimezone(tzinfo)
            except Exception:
                continue

            week_start_date = start_dt_local.date() - timedelta(days=start_dt_local.date().weekday())
            week_start = week_start_date.isoformat()
            week_end = (week_start_date + timedelta(days=6)).isoformat()

            items.append(
                {
                    "session_id": str(row.get("session_id", "")),
                    "timestamp_local": start_dt_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "day": start_dt_local.date().isoformat(),
                    "week": f"{week_start} to {week_end}",
                    "duration_seconds": duration_seconds,
                    "image_path": str(image_path) if image_path else "",
                    "start_timestamp_utc": str(start_ts),
                    "end_timestamp_utc": str(end_ts) if end_ts else "",
                }
            )

    items.sort(key=lambda x: x["start_timestamp_utc"], reverse=True)
    return items


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        k = row.get(key, "unknown")
        grouped.setdefault(str(k), []).append(row)
    return grouped


def render_instance_rows(rows: list[dict]) -> None:
    header_cols = st.columns([2.2, 1.1, 2.7], gap="small")
    header_cols[0].markdown("**timestamp**")
    header_cols[1].markdown("**duration (s)**")
    header_cols[2].markdown("**image**")
    st.markdown("---")

    for row in rows:
        cols = st.columns([2.2, 1.1, 2.7], gap="small")
        cols[0].write(row["timestamp_local"])
        cols[1].write(f"{float(row['duration_seconds']):.2f}")
        image_path = Path(row["image_path"])
        if image_path.exists():
            cols[2].image(str(image_path), use_container_width=True)
        else:
            cols[2].caption(f"Image missing: {image_path}")
        st.markdown("---")


def show_cumulative(rows: list[dict]) -> None:
    total_count = len(rows)
    total_duration = sum(float(r["duration_seconds"]) for r in rows)
    c1, c2 = st.columns(2)
    c1.metric("Cumulative Count", f"{total_count}")
    c2.metric("Cumulative Duration", f"{total_duration:.2f} s")


st.set_page_config(page_title="NoSlouchBench Reports", page_icon="📊", layout="wide")
st.title("NoSlouchBench Slouch Instances")
st.caption("Day-wise and week-wise slouch instances")

refresh = st.button("Refresh Reports", type="primary")
if refresh:
    try:
        write_habits_reports(
            summaries_dir=SUMMARIES_DIR,
            sessions_dir=SESSIONS_DIR,
            output_dir=REPORTS_DIR,
            timezone_name=TIMEZONE_NAME,
        )
        st.success("Reports refreshed")
    except Exception as exc:
        st.error(f"Failed to refresh reports: {exc}")
else:
    try:
        if reports_need_refresh(REPORTS_DIR, mode="both", timezone_name=TIMEZONE_NAME):
            write_habits_reports(
                summaries_dir=SUMMARIES_DIR,
                sessions_dir=SESSIONS_DIR,
                output_dir=REPORTS_DIR,
                timezone_name=TIMEZONE_NAME,
            )
    except Exception:
        pass

slouch_instances = load_slouch_instances(INSTANCES_DIR, TIMEZONE_NAME)

if not slouch_instances:
    st.info("No slouch instances found yet. Run webcam sessions first.")
else:
    tab_daily_instances, tab_weekly_instances = st.tabs(["Daily Slouch Instances", "Weekly Slouch Instances"])

    with tab_daily_instances:
        st.subheader("Day-wise Slouch Instances")
        by_day = group_by(slouch_instances, "day")
        day_options = sorted(by_day.keys(), reverse=True)
        selected_day = st.selectbox("Select day", options=day_options, key="day_instances")
        selected_rows = by_day[selected_day]
        show_cumulative(selected_rows)
        render_instance_rows(selected_rows)

    with tab_weekly_instances:
        st.subheader("Week-wise Slouch Instances")
        by_week = group_by(slouch_instances, "week")
        week_options = sorted(by_week.keys(), reverse=True)
        selected_week = st.selectbox("Select week", options=week_options, key="week_instances")
        selected_rows = by_week[selected_week]
        show_cumulative(selected_rows)
        render_instance_rows(selected_rows)
