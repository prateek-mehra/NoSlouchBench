from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


@dataclass
class SessionHabitRecord:
    session_id: str
    model_name: str
    generated_local: datetime
    day: date
    week_start: date
    week_end: date
    sitting_seconds: float
    slouch_seconds: float
    beep_events: int
    beep_source: str


def resolve_timezone(timezone_name: str):
    tz_key = timezone_name.strip() if timezone_name else "local"
    if tz_key.lower() == "local":
        return datetime.now().astimezone().tzinfo
    if tz_key.lower() in {"utc", "z"}:
        return timezone.utc
    try:
        return ZoneInfo(tz_key)
    except Exception as exc:
        raise ValueError(f"Unsupported timezone: {timezone_name}") from exc


def load_session_habit_records(summaries_dir: Path, sessions_dir: Path, tzinfo) -> list[SessionHabitRecord]:
    records: list[SessionHabitRecord] = []
    for summary_path in sorted(summaries_dir.glob("*.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        session_id = str(data.get("session_id", "")).strip()
        if not session_id:
            continue

        generated_local = _parse_summary_timestamp(data, tzinfo)
        day = generated_local.date()
        week_start = day - timedelta(days=day.weekday())
        week_end = week_start + timedelta(days=6)

        sitting_seconds = float(data.get("duration_seconds", 0.0))
        slouch_ratio = float(data.get("slouch_ratio", 0.0))
        slouch_seconds = max(sitting_seconds * slouch_ratio, 0.0)

        session_log = sessions_dir / f"{session_id}.jsonl"
        beep_events, beep_source = count_beep_events(session_log)

        records.append(
            SessionHabitRecord(
                session_id=session_id,
                model_name=str(data.get("model_name", "unknown")),
                generated_local=generated_local,
                day=day,
                week_start=week_start,
                week_end=week_end,
                sitting_seconds=sitting_seconds,
                slouch_seconds=slouch_seconds,
                beep_events=beep_events,
                beep_source=beep_source,
            )
        )
    return records


def count_beep_events(session_log_path: Path) -> tuple[int, str]:
    if not session_log_path.exists():
        return 0, "missing_log"

    count = 0
    prev_label: str | None = None
    try:
        with session_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = str(item.get("posture_label", "")).strip().lower()
                if not label:
                    continue
                if label == "slouch" and prev_label != "slouch":
                    count += 1
                prev_label = label
    except OSError:
        return 0, "missing_log"

    return count, "log"


def aggregate_daily(records: list[SessionHabitRecord]) -> list[dict[str, Any]]:
    grouped: dict[date, list[SessionHabitRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.day].append(rec)

    rows: list[dict[str, Any]] = []
    for day in sorted(grouped):
        runs = grouped[day]
        beep_source = _aggregate_beep_source([r.beep_source for r in runs])
        rows.append(
            {
                "date": day.isoformat(),
                "hours_sitting": sum(r.sitting_seconds for r in runs) / 3600.0,
                "slouch_minutes": sum(r.slouch_seconds for r in runs) / 60.0,
                "beep_events": sum(r.beep_events for r in runs),
                "sessions": len(runs),
                "beep_source": beep_source,
            }
        )
    return rows


def aggregate_weekly(records: list[SessionHabitRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[date, date], list[SessionHabitRecord]] = defaultdict(list)
    for rec in records:
        grouped[(rec.week_start, rec.week_end)].append(rec)

    rows: list[dict[str, Any]] = []
    for week_start, week_end in sorted(grouped):
        runs = grouped[(week_start, week_end)]
        beep_source = _aggregate_beep_source([r.beep_source for r in runs])
        rows.append(
            {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "week": f"{week_start.isoformat()} to {week_end.isoformat()}",
                "hours_sitting": sum(r.sitting_seconds for r in runs) / 3600.0,
                "slouch_minutes": sum(r.slouch_seconds for r in runs) / 60.0,
                "beep_events": sum(r.beep_events for r in runs),
                "sessions": len(runs),
                "beep_source": beep_source,
            }
        )
    return rows


def write_habits_reports(
    summaries_dir: Path,
    sessions_dir: Path,
    output_dir: Path,
    timezone_name: str = "local",
) -> dict[str, Path]:
    tzinfo = resolve_timezone(timezone_name)
    records = load_session_habit_records(summaries_dir, sessions_dir, tzinfo)
    daily_rows = aggregate_daily(records)
    weekly_rows = aggregate_weekly(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_csv = output_dir / "habits_daily.csv"
    weekly_csv = output_dir / "habits_weekly.csv"
    daily_md = output_dir / "habits_daily.md"
    weekly_md = output_dir / "habits_weekly.md"

    _write_daily_csv(daily_rows, daily_csv)
    _write_weekly_csv(weekly_rows, weekly_csv)
    _write_daily_markdown(daily_rows, daily_md)
    _write_weekly_markdown(weekly_rows, weekly_md)

    return {
        "daily_csv": daily_csv,
        "weekly_csv": weekly_csv,
        "daily_md": daily_md,
        "weekly_md": weekly_md,
    }


def render_terminal_summary(daily_rows: list[dict[str, Any]], weekly_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("Daily Habit Stats")
    if not daily_rows:
        lines.append("  No sessions recorded.")
    else:
        lines.append("  Date       | Hours Sitting | Slouch Minutes | Beep Events | Sessions")
        lines.append("  -----------|---------------|----------------|-------------|---------")
        for row in daily_rows[-7:]:
            lines.append(
                "  {date} | {hours:13.2f} | {slouch:14.1f} | {beeps:11d} | {sessions:8d}".format(
                    date=row["date"],
                    hours=row["hours_sitting"],
                    slouch=row["slouch_minutes"],
                    beeps=int(row["beep_events"]),
                    sessions=int(row["sessions"]),
                )
            )

    lines.append("")
    lines.append("Weekly Habit Stats")
    if not weekly_rows:
        lines.append("  No sessions recorded.")
    else:
        lines.append("  Week                  | Hours Sitting | Slouch Minutes | Beep Events | Sessions")
        lines.append("  ----------------------|---------------|----------------|-------------|---------")
        for row in weekly_rows[-4:]:
            lines.append(
                "  {week:22} | {hours:13.2f} | {slouch:14.1f} | {beeps:11d} | {sessions:8d}".format(
                    week=row["week"],
                    hours=row["hours_sitting"],
                    slouch=row["slouch_minutes"],
                    beeps=int(row["beep_events"]),
                    sessions=int(row["sessions"]),
                )
            )

    return "\n".join(lines)


def read_report_rows(report_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    daily_rows = _read_csv_rows(report_dir / "habits_daily.csv")
    weekly_rows = _read_csv_rows(report_dir / "habits_weekly.csv")

    for row in daily_rows:
        row["hours_sitting"] = float(row["hours_sitting"])
        row["slouch_minutes"] = float(row["slouch_minutes"])
        row["beep_events"] = int(row["beep_events"])
        row["sessions"] = int(row["sessions"])

    for row in weekly_rows:
        row["hours_sitting"] = float(row["hours_sitting"])
        row["slouch_minutes"] = float(row["slouch_minutes"])
        row["beep_events"] = int(row["beep_events"])
        row["sessions"] = int(row["sessions"])

    return daily_rows, weekly_rows


def reports_need_refresh(report_dir: Path, mode: str, timezone_name: str = "local") -> bool:
    tzinfo = resolve_timezone(timezone_name)
    now = datetime.now(tzinfo)

    daily_csv = report_dir / "habits_daily.csv"
    weekly_csv = report_dir / "habits_weekly.csv"

    if mode in {"daily", "both"}:
        if not daily_csv.exists():
            return True
        daily_mtime = datetime.fromtimestamp(daily_csv.stat().st_mtime, tz=tzinfo)
        if daily_mtime.date() < now.date():
            return True

    if mode in {"weekly", "both"}:
        if not weekly_csv.exists():
            return True
        weekly_mtime = datetime.fromtimestamp(weekly_csv.stat().st_mtime, tz=tzinfo)
        current_week_start = now.date() - timedelta(days=now.date().weekday())
        week_start_dt = datetime.combine(current_week_start, time.min, tzinfo=tzinfo)
        if weekly_mtime < week_start_dt:
            return True

    return False


def _parse_summary_timestamp(summary_data: dict[str, Any], tzinfo) -> datetime:
    generated_at = summary_data.get("generated_at_utc")
    if generated_at:
        try:
            dt = datetime.fromisoformat(str(generated_at))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(tzinfo)
        except ValueError:
            pass

    session_id = str(summary_data.get("session_id", ""))
    ts = _extract_timestamp_from_session_id(session_id)
    if ts is not None:
        return ts.astimezone(tzinfo)

    return datetime.now(tzinfo)


def _extract_timestamp_from_session_id(session_id: str) -> datetime | None:
    parts = session_id.split("_")
    for part in parts:
        if len(part) >= 16 and "T" in part and part.endswith("Z"):
            head = part[:16]
            try:
                return datetime.strptime(head + "Z", "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _aggregate_beep_source(sources: list[str]) -> str:
    uniq = set(sources)
    if uniq == {"log"}:
        return "log"
    if uniq == {"missing_log"}:
        return "missing_log"
    return "mixed"


def _write_daily_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["date", "hours_sitting", "slouch_minutes", "beep_events", "sessions", "beep_source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "date": row["date"],
                    "hours_sitting": f"{row['hours_sitting']:.4f}",
                    "slouch_minutes": f"{row['slouch_minutes']:.4f}",
                    "beep_events": int(row["beep_events"]),
                    "sessions": int(row["sessions"]),
                    "beep_source": row["beep_source"],
                }
            )


def _write_weekly_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "week_start",
        "week_end",
        "week",
        "hours_sitting",
        "slouch_minutes",
        "beep_events",
        "sessions",
        "beep_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "week_start": row["week_start"],
                    "week_end": row["week_end"],
                    "week": row["week"],
                    "hours_sitting": f"{row['hours_sitting']:.4f}",
                    "slouch_minutes": f"{row['slouch_minutes']:.4f}",
                    "beep_events": int(row["beep_events"]),
                    "sessions": int(row["sessions"]),
                    "beep_source": row["beep_source"],
                }
            )


def _write_daily_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Daily Habit Report\n\n")
        if not rows:
            f.write("No sessions recorded for this period.\n")
            return

        f.write("| Date | Hours Sitting | Slouch Minutes | Beep Events | Sessions |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {date} | {hours:.2f} | {slouch:.1f} | {beeps} | {sessions} |\n".format(
                    date=row["date"],
                    hours=row["hours_sitting"],
                    slouch=row["slouch_minutes"],
                    beeps=int(row["beep_events"]),
                    sessions=int(row["sessions"]),
                )
            )

        if any(row["beep_source"] != "log" for row in rows):
            f.write("\nNote: some beep counts used `missing_log` fallback (0 events).\n")


def _write_weekly_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Weekly Habit Report\n\n")
        if not rows:
            f.write("No sessions recorded for this period.\n")
            return

        f.write("| Week (Mon-Sun) | Hours Sitting | Slouch Minutes | Beep Events | Sessions |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {week} | {hours:.2f} | {slouch:.1f} | {beeps} | {sessions} |\n".format(
                    week=row["week"],
                    hours=row["hours_sitting"],
                    slouch=row["slouch_minutes"],
                    beeps=int(row["beep_events"]),
                    sessions=int(row["sessions"]),
                )
            )

        if any(row["beep_source"] != "log" for row in rows):
            f.write("\nNote: some beep counts used `missing_log` fallback (0 events).\n")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
