from __future__ import annotations

import argparse
from pathlib import Path

from noslouchbench.habits_report import (
    aggregate_daily,
    aggregate_weekly,
    load_session_habit_records,
    render_terminal_summary,
    resolve_timezone,
    write_habits_reports,
)
from noslouchbench.report import aggregate_summaries, write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NoSlouchBench CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-webcam", help="Run posture benchmark on webcam")
    run.add_argument("--model", default="yolo-pose", help="Model name (e.g., yolo-pose)")
    run.add_argument("--camera-id", type=int, default=0, help="Webcam device id")
    run.add_argument(
        "--camera-name",
        default=None,
        help='Preferred webcam name (e.g., "Logitech C270"). Resolves to camera index when possible.',
    )
    run.add_argument("--duration-minutes", type=float, default=None, help="Stop automatically after N minutes")
    run.add_argument("--frame-skip", type=int, default=0, help="Process every N+1 frame")
    run.add_argument("--session-tag", default=None, help="Optional run tag appended to session id")
    run.add_argument("--display", action="store_true", help="Show annotated webcam window")
    run.add_argument("--no-beep", action="store_true", help="Disable continuous beep while slouching")
    run.add_argument(
        "--screen-blocker",
        action="store_true",
        help="Enable full-screen semi-transparent blocker while slouching (optional test flow).",
    )
    run.add_argument(
        "--screen-blocker-opacity",
        type=float,
        default=0.78,
        help="Opacity for blocker overlay (0.2-0.95). Higher means darker.",
    )
    run.add_argument(
        "--screen-blocker-kill-switch",
        default="Ctrl+Shift+K",
        help="Kill-switch hint shown on blocker overlay (default Ctrl+Shift+K).",
    )
    run.add_argument(
        "--no-lock-swipe-gesture",
        action="store_true",
        help="Do not disable trackpad swipe gesture while screen blocker is enabled.",
    )
    run.add_argument("--record-path", default=None, help="Optional output video path (e.g., outputs/debug/run.mp4)")
    run.add_argument("--output-dir", default="outputs", help="Directory for logs and summaries")
    run.add_argument("--config", default="configs/models.yaml", help="Model config YAML")

    summ = sub.add_parser("summarize", help="Aggregate session summaries by model")
    summ.add_argument("--input-dir", default="outputs/summaries", help="Directory containing session summary JSON")
    summ.add_argument("--output-dir", default="outputs/reports", help="Output directory for comparison reports")

    habits = sub.add_parser("summarize-habits", help="Generate daily/weekly human-readable habit reports")
    habits.add_argument("--summaries-dir", default="outputs/summaries", help="Directory containing session summaries")
    habits.add_argument("--sessions-dir", default="outputs/sessions", help="Directory containing frame-level JSONL logs")
    habits.add_argument("--output-dir", default="outputs/reports", help="Directory for habit reports")
    habits.add_argument("--timezone", default="local", help='Timezone for bucketing (e.g., "local", "UTC", "Asia/Kolkata")')
    habits.add_argument("--show", action="store_true", help="Print terminal summary for daily/weekly stats")

    sub.add_parser("list-cameras", help="List available webcam devices")

    analyze = sub.add_parser("analyze-video", help="Run posture model on a saved video and output diagnostics")
    analyze.add_argument("--model", default="yolo-pose", help="Model name (e.g., yolo-pose)")
    analyze.add_argument("--input-video", required=True, help="Path to recorded video file")
    analyze.add_argument("--output-dir", default="outputs/analysis", help="Where to write analysis JSON artifacts")
    analyze.add_argument("--config", default="configs/models.yaml", help="Model config YAML")

    return parser.parse_args()


def run_webcam(args: argparse.Namespace) -> int:
    from noslouchbench.camera import resolve_camera_id
    from noslouchbench.config import load_model_config
    from noslouchbench.detectors.factory import build_detector
    from noslouchbench.runner import WebcamBenchmarkRunner

    config_path = Path(args.config)
    model_cfg = load_model_config(config_path, args.model)
    detector = build_detector(args.model, model_cfg=model_cfg)
    camera_id = resolve_camera_id(args.camera_id, args.camera_name)
    print(f"Using camera_id={camera_id}")

    beep_on_slouch = (not args.no_beep) and (not args.screen_blocker)
    lock_swipe_gesture = args.screen_blocker and (not args.no_lock_swipe_gesture)
    if args.screen_blocker and not args.no_beep:
        print("Screen-blocker mode enabled: muting beep for blocker-only testing.")
    if lock_swipe_gesture:
        print("Screen-blocker mode: 3-finger swipe gesture will be temporarily disabled.")

    runner = WebcamBenchmarkRunner(
        detector=detector,
        model_name=args.model,
        output_dir=Path(args.output_dir),
        camera_id=camera_id,
        display=args.display,
        beep_on_slouch=beep_on_slouch,
        block_screen_on_slouch=args.screen_blocker,
        lock_swipe_gesture=lock_swipe_gesture,
        blocker_opacity=args.screen_blocker_opacity,
        blocker_kill_switch=args.screen_blocker_kill_switch,
        record_path=Path(args.record_path) if args.record_path else None,
        duration_minutes=args.duration_minutes,
        frame_skip=args.frame_skip,
        session_tag=args.session_tag,
    )

    artifacts = runner.run()
    print(f"Session complete: {artifacts.session_id}")
    print(f"Frame logs: {artifacts.event_log_path}")
    print(f"Summary: {artifacts.summary_path}")
    print("Key metrics:")
    print(f"  avg latency (ms): {artifacts.summary['latency_ms_avg']:.2f}")
    print(f"  p95 latency (ms): {artifacts.summary['latency_ms_p95']:.2f}")
    print(f"  detection rate: {artifacts.summary['detection_rate']:.3f}")
    print(f"  slouch ratio: {artifacts.summary['slouch_ratio']:.3f}")
    return 0


def summarize(args: argparse.Namespace) -> int:
    aggregated = aggregate_summaries(Path(args.input_dir))
    csv_path, md_path = write_reports(aggregated, Path(args.output_dir))
    print(f"Wrote comparison CSV: {csv_path}")
    print(f"Wrote comparison Markdown: {md_path}")
    return 0


def summarize_habits_cmd(args: argparse.Namespace) -> int:
    summaries_dir = Path(args.summaries_dir)
    sessions_dir = Path(args.sessions_dir)
    output_dir = Path(args.output_dir)

    artifacts = write_habits_reports(
        summaries_dir=summaries_dir,
        sessions_dir=sessions_dir,
        output_dir=output_dir,
        timezone_name=args.timezone,
    )
    print(f"Wrote daily CSV: {artifacts['daily_csv']}")
    print(f"Wrote weekly CSV: {artifacts['weekly_csv']}")
    print(f"Wrote daily Markdown: {artifacts['daily_md']}")
    print(f"Wrote weekly Markdown: {artifacts['weekly_md']}")

    if args.show:
        tzinfo = resolve_timezone(args.timezone)
        records = load_session_habit_records(summaries_dir=summaries_dir, sessions_dir=sessions_dir, tzinfo=tzinfo)
        daily_rows = aggregate_daily(records)
        weekly_rows = aggregate_weekly(records)
        print("")
        print(render_terminal_summary(daily_rows, weekly_rows))

    return 0


def list_cameras() -> int:
    from noslouchbench.camera import list_cameras_avfoundation

    cameras = list_cameras_avfoundation()
    if not cameras:
        print("No cameras found via ffmpeg AVFoundation listing.")
        return 1

    print("Available cameras:")
    for cam in cameras:
        print(f"  [{cam.idx}] {cam.name}")
    return 0


def analyze_video_cmd(args: argparse.Namespace) -> int:
    from noslouchbench.config import load_model_config
    from noslouchbench.detectors.factory import build_detector
    from noslouchbench.video_analysis import analyze_video

    config_path = Path(args.config)
    model_cfg = load_model_config(config_path, args.model)
    detector = build_detector(args.model, model_cfg=model_cfg)
    artifacts = analyze_video(
        detector=detector,
        model_name=args.model,
        input_video=Path(args.input_video),
        output_dir=Path(args.output_dir),
    )
    print(f"Analysis summary: {artifacts.summary_path}")
    print(f"Frame diagnostics: {artifacts.events_path}")
    print("Key diagnostics:")
    print(f"  slouch_ratio: {artifacts.summary['slouch_ratio']:.3f}")
    print(f"  yaw_ambiguous_ratio: {artifacts.summary['yaw_ambiguous_ratio']:.3f}")
    print(f"  slouch_given_yaw_ambiguous: {artifacts.summary['slouch_given_yaw_ambiguous']:.3f}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "run-webcam":
        return run_webcam(args)
    if args.command == "summarize":
        return summarize(args)
    if args.command == "summarize-habits":
        return summarize_habits_cmd(args)
    if args.command == "list-cameras":
        return list_cameras()
    if args.command == "analyze-video":
        return analyze_video_cmd(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
