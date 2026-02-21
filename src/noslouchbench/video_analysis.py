from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from noslouchbench.detectors.base import BasePostureDetector


@dataclass
class AnalysisArtifacts:
    summary_path: Path
    events_path: Path
    summary: dict


def analyze_video(
    detector: BasePostureDetector,
    model_name: str,
    input_video: Path,
    output_dir: Path,
) -> AnalysisArtifacts:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / f"{input_video.stem}_analysis.jsonl"
    summary_path = output_dir / f"{input_video.stem}_analysis_summary.json"

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    frame_idx = 0
    processed = 0
    detected = 0
    slouch = 0
    upright = 0
    latencies: list[float] = []
    yaw_amb = 0
    slouch_yaw_amb = 0

    try:
        with events_path.open("w", encoding="utf-8") as logf:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                result = detector.infer(frame)
                processed += 1
                latencies.append(result.latency_ms)
                if result.detected:
                    detected += 1
                if result.posture_label == "slouch":
                    slouch += 1
                elif result.posture_label == "upright":
                    upright += 1

                yaw_flag = bool(result.metadata.get("yaw_ambiguous", False))
                if yaw_flag:
                    yaw_amb += 1
                    if result.posture_label == "slouch":
                        slouch_yaw_amb += 1

                record = {
                    "frame_idx": frame_idx,
                    "posture_label": result.posture_label,
                    "detected": result.detected,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                    "metadata": result.metadata,
                }
                logf.write(json.dumps(record) + "\n")
    finally:
        cap.release()
        detector.close()

    arr = np.array(latencies, dtype=np.float32) if latencies else np.array([0.0], dtype=np.float32)
    summary = {
        "video": str(input_video),
        "model_name": model_name,
        "processed_frames": processed,
        "detected_frames": detected,
        "detection_rate": (detected / processed) if processed else 0.0,
        "slouch_frames": slouch,
        "upright_frames": upright,
        "slouch_ratio": (slouch / processed) if processed else 0.0,
        "latency_ms_avg": float(np.mean(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "yaw_ambiguous_frames": yaw_amb,
        "slouch_during_yaw_ambiguous": slouch_yaw_amb,
        "yaw_ambiguous_ratio": (yaw_amb / processed) if processed else 0.0,
        "slouch_given_yaw_ambiguous": (slouch_yaw_amb / yaw_amb) if yaw_amb else 0.0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return AnalysisArtifacts(summary_path=summary_path, events_path=events_path, summary=summary)

