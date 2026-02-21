from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from noslouchbench.audio import SlouchBeeper
from noslouchbench.detectors.base import BasePostureDetector


@dataclass
class RunArtifacts:
    session_id: str
    event_log_path: Path
    summary_path: Path
    summary: dict


class WebcamBenchmarkRunner:
    def __init__(
        self,
        detector: BasePostureDetector,
        model_name: str,
        output_dir: Path,
        camera_id: int = 0,
        display: bool = True,
        beep_on_slouch: bool = True,
        duration_minutes: float | None = None,
        frame_skip: int = 0,
        session_tag: str | None = None,
    ) -> None:
        self.detector = detector
        self.model_name = model_name
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.display = display
        self.beep_on_slouch = beep_on_slouch
        self.duration_minutes = duration_minutes
        self.frame_skip = max(frame_skip, 0)
        self.session_tag = session_tag

    def run(self) -> RunArtifacts:
        session_id = self._build_session_id(self.model_name, self.session_tag)
        sessions_dir = self.output_dir / "sessions"
        summaries_dir = self.output_dir / "summaries"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        summaries_dir.mkdir(parents=True, exist_ok=True)

        event_log_path = sessions_dir / f"{session_id}.jsonl"
        summary_path = summaries_dir / f"{session_id}.json"

        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam camera_id={self.camera_id}")

        start = time.time()
        frame_idx = 0
        latencies_ms: list[float] = []
        detected_frames = 0
        slouch_frames = 0
        upright_frames = 0
        processed_frames = 0
        beeper = SlouchBeeper() if self.beep_on_slouch else None

        try:
            with event_log_path.open("w", encoding="utf-8") as logf:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame_idx += 1
                    if self.frame_skip and frame_idx % (self.frame_skip + 1) != 1:
                        continue

                    ts = datetime.now(timezone.utc).isoformat()
                    result = self.detector.infer(frame)

                    processed_frames += 1
                    latencies_ms.append(result.latency_ms)
                    if result.detected:
                        detected_frames += 1
                    if result.posture_label == "slouch":
                        slouch_frames += 1
                    elif result.posture_label == "upright":
                        upright_frames += 1

                    if beeper is not None:
                        if result.posture_label == "slouch":
                            beeper.start()
                        else:
                            beeper.stop()

                    record = {
                        "timestamp_utc": ts,
                        "session_id": session_id,
                        "model_name": self.model_name,
                        "frame_idx": frame_idx,
                        "detected": result.detected,
                        "posture_label": result.posture_label,
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "metadata": result.metadata,
                    }
                    logf.write(json.dumps(record) + "\n")

                    if self.display:
                        self._draw_overlay(frame, record)
                        cv2.imshow("NoSlouchBench Webcam", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if self.duration_minutes is not None:
                        elapsed_min = (time.time() - start) / 60.0
                        if elapsed_min >= self.duration_minutes:
                            break
        finally:
            if beeper is not None:
                beeper.close()
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

        elapsed_s = max(time.time() - start, 1e-6)
        summary = self._build_summary(
            session_id=session_id,
            elapsed_s=elapsed_s,
            processed_frames=processed_frames,
            detected_frames=detected_frames,
            slouch_frames=slouch_frames,
            upright_frames=upright_frames,
            latencies_ms=latencies_ms,
        )

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return RunArtifacts(
            session_id=session_id,
            event_log_path=event_log_path,
            summary_path=summary_path,
            summary=summary,
        )

    def _build_summary(
        self,
        session_id: str,
        elapsed_s: float,
        processed_frames: int,
        detected_frames: int,
        slouch_frames: int,
        upright_frames: int,
        latencies_ms: list[float],
    ) -> dict:
        latency_arr = np.array(latencies_ms, dtype=np.float32) if latencies_ms else np.array([0.0], dtype=np.float32)
        detection_rate = (detected_frames / processed_frames) if processed_frames else 0.0
        slouch_ratio = (slouch_frames / processed_frames) if processed_frames else 0.0
        upright_ratio = (upright_frames / processed_frames) if processed_frames else 0.0

        return {
            "session_id": session_id,
            "model_name": self.model_name,
            "duration_seconds": elapsed_s,
            "processed_frames": processed_frames,
            "fps_effective": processed_frames / elapsed_s,
            "detection_rate": detection_rate,
            "slouch_ratio": slouch_ratio,
            "upright_ratio": upright_ratio,
            "latency_ms_avg": float(np.mean(latency_arr)),
            "latency_ms_p95": float(np.percentile(latency_arr, 95)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _draw_overlay(frame, record: dict) -> None:
        label = record["posture_label"]
        color = (0, 0, 255) if label == "slouch" else (0, 255, 0)
        metadata = record.get("metadata", {})
        cv2.putText(
            frame,
            f"Model: {record['model_name']}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Posture: {label} ({record['confidence']:.2f})",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Latency: {record['latency_ms']:.1f} ms",
            (16, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press q to stop",
            (16, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        score = metadata.get("slouch_score")
        score_thr = metadata.get("slouch_score_threshold")
        if score is not None and score_thr is not None:
            cv2.putText(
                frame,
                f"SlouchScore: {score:.3f} (thr {score_thr:.3f})",
                (16, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 120),
                2,
            )
        side = metadata.get("selected_side")
        if side is not None:
            cv2.putText(
                frame,
                f"Side used: {side}",
                (16, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                2,
            )

        hfo = metadata.get("head_forward_offset")
        lean = metadata.get("torso_lean_angle_deg")
        neck = metadata.get("neck_drop")
        if hfo is not None and lean is not None and neck is not None:
            cv2.putText(
                frame,
                f"HFwd:{hfo:.2f} Lean:{lean:.1f}deg Neck:{neck:.2f}",
                (16, 208),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 220, 255),
                2,
            )
        if metadata.get("yaw_ambiguous"):
            cv2.putText(
                frame,
                "Head turn detected (ear overlap) - reducing slouch sensitivity",
                (16, 238),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 180, 255),
                2,
            )

        # Draw and label the exact landmarks used by the posture logic.
        points = metadata.get("landmarks_norm", {})
        if points:
            h, w, _ = frame.shape
            for part, xy in points.items():
                x = int(float(xy[0]) * w)
                y = int(float(xy[1]) * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    part,
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )

    @staticmethod
    def _build_session_id(model_name: str, session_tag: str | None) -> str:
        t = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{session_tag}" if session_tag else ""
        return f"{model_name}_{t}{suffix}"
