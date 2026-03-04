from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from noslouchbench.habits_report import (
    aggregate_daily,
    aggregate_weekly,
    count_beep_events,
    load_session_habit_records,
    resolve_timezone,
)


class HabitsReportTests(unittest.TestCase):
    def test_count_beep_events_from_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "session.jsonl"
            events = [
                {"posture_label": "upright"},
                {"posture_label": "slouch"},
                {"posture_label": "slouch"},
                {"posture_label": "upright"},
                {"posture_label": "slouch"},
            ]
            p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

            count, source = count_beep_events(p)

            self.assertEqual(count, 2)
            self.assertEqual(source, "log")

    def test_missing_log_sets_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.jsonl"
            count, source = count_beep_events(p)
            self.assertEqual(count, 0)
            self.assertEqual(source, "missing_log")

    def test_daily_weekly_aggregation_and_slouch_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summaries = root / "summaries"
            sessions = root / "sessions"
            summaries.mkdir()
            sessions.mkdir()

            summary_data = {
                "session_id": "yolo-pose_20260304T030000Z",
                "model_name": "yolo-pose",
                "duration_seconds": 3600,
                "slouch_ratio": 0.25,
                "generated_at_utc": "2026-03-04T03:00:00+00:00",
            }
            (summaries / "a.json").write_text(json.dumps(summary_data), encoding="utf-8")
            (sessions / "yolo-pose_20260304T030000Z.jsonl").write_text(
                "\n".join([
                    json.dumps({"posture_label": "upright"}),
                    json.dumps({"posture_label": "slouch"}),
                    json.dumps({"posture_label": "upright"}),
                ]),
                encoding="utf-8",
            )

            tzinfo = resolve_timezone("UTC")
            records = load_session_habit_records(summaries, sessions, tzinfo)
            daily = aggregate_daily(records)
            weekly = aggregate_weekly(records)

            self.assertEqual(len(daily), 1)
            self.assertAlmostEqual(daily[0]["hours_sitting"], 1.0)
            self.assertAlmostEqual(daily[0]["slouch_minutes"], 15.0)
            self.assertEqual(daily[0]["beep_events"], 1)

            self.assertEqual(len(weekly), 1)
            self.assertEqual(weekly[0]["sessions"], 1)


if __name__ == "__main__":
    unittest.main()
