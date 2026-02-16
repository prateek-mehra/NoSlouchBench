from __future__ import annotations

import argparse
from pathlib import Path

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
    run.add_argument("--output-dir", default="outputs", help="Directory for logs and summaries")
    run.add_argument("--config", default="configs/models.yaml", help="Model config YAML")

    summ = sub.add_parser("summarize", help="Aggregate session summaries by model")
    summ.add_argument("--input-dir", default="outputs/summaries", help="Directory containing session summary JSON")
    summ.add_argument("--output-dir", default="outputs/reports", help="Output directory for comparison reports")

    sub.add_parser("list-cameras", help="List available webcam devices")

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

    runner = WebcamBenchmarkRunner(
        detector=detector,
        model_name=args.model,
        output_dir=Path(args.output_dir),
        camera_id=camera_id,
        display=args.display,
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


def main() -> int:
    args = parse_args()
    if args.command == "run-webcam":
        return run_webcam(args)
    if args.command == "summarize":
        return summarize(args)
    if args.command == "list-cameras":
        return list_cameras()
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
