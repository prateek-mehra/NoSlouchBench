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
from noslouchbench.screen_blocker import ScreenBlocker
from noslouchbench.system_lock import TrackpadSwipeGuard


@dataclass
class RunArtifacts:
    session_id: str
    event_log_path: Path
    summary_path: Path
    slouch_instances_path: Path
    summary: dict


class WebcamBenchmarkRunner:
    BLOCKER_SLOUCH_SCORE_THRESHOLD = 0.12
    BLOCKER_SUSTAIN_SECONDS = 2.0

    def __init__(
        self,
        detector: BasePostureDetector,
        model_name: str,
        output_dir: Path,
        camera_id: int = 0,
        display: bool = True,
        beep_on_slouch: bool = True,
        block_screen_on_slouch: bool = False,
        lock_swipe_gesture: bool = False,
        blocker_opacity: float = 0.78,
        blocker_kill_switch: str = "Ctrl+Shift+K",
        record_path: Path | None = None,
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
        self.block_screen_on_slouch = block_screen_on_slouch
        self.lock_swipe_gesture = lock_swipe_gesture
        self.blocker_opacity = blocker_opacity
        self.blocker_kill_switch = blocker_kill_switch
        self.record_path = record_path
        self.duration_minutes = duration_minutes
        self.frame_skip = max(frame_skip, 0)
        self.session_tag = session_tag

    def run(self) -> RunArtifacts:
        session_id = self._build_session_id(self.model_name, self.session_tag)
        sessions_dir = self.output_dir / "sessions"
        summaries_dir = self.output_dir / "summaries"
        slouch_instances_dir = self.output_dir / "slouch_instances"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        summaries_dir.mkdir(parents=True, exist_ok=True)
        slouch_instances_dir.mkdir(parents=True, exist_ok=True)

        event_log_path = sessions_dir / f"{session_id}.jsonl"
        summary_path = summaries_dir / f"{session_id}.json"
        slouch_instances_path = slouch_instances_dir / f"{session_id}.json"

        cap, first_frame = self._open_camera_capture()

        start = time.time()
        frame_idx = 0
        latencies_ms: list[float] = []
        detected_frames = 0
        slouch_frames = 0
        upright_frames = 0
        processed_frames = 0
        beeper = SlouchBeeper() if self.beep_on_slouch else None
        blocker = None
        swipe_guard = None
        blocker_condition_since: float | None = None
        blocker_active = False
        active_slouch_instance: dict | None = None
        slouch_instances: list[dict] = []
        def persist_slouch_instances() -> None:
            with slouch_instances_path.open("w", encoding="utf-8") as f:
                json.dump(slouch_instances, f, indent=2)

        persist_slouch_instances()
        if self.block_screen_on_slouch:
            blocker = ScreenBlocker(opacity=self.blocker_opacity, kill_switch=self.blocker_kill_switch)
            if not blocker.available:
                print("Screen blocker unavailable in this environment. Continuing without screen blocking.")
                blocker = None
        if self.lock_swipe_gesture:
            swipe_guard = TrackpadSwipeGuard()
            if swipe_guard.supported:
                swipe_guard.activate()
                print("Trackpad swipe lock enabled for this session.")
            else:
                swipe_guard = None
        writer = None

        try:
            with event_log_path.open("w", encoding="utf-8") as logf:
                while True:
                    if first_frame is not None:
                        frame = first_frame
                        first_frame = None
                        ok = True
                    else:
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
                    if blocker is not None:
                        now_ts = time.time()
                        neck_len = result.metadata.get("neck_length_ratio")
                        neck_len_threshold = result.metadata.get("neck_length_slouch_threshold")
                        if neck_len is not None and neck_len_threshold is not None:
                            try:
                                should_block = float(neck_len) < float(neck_len_threshold)
                            except (TypeError, ValueError):
                                should_block = False
                        else:
                            raw_score = result.metadata.get("slouch_score")
                            try:
                                should_block = float(raw_score) > self.BLOCKER_SLOUCH_SCORE_THRESHOLD
                            except (TypeError, ValueError):
                                should_block = False

                        if blocker.killed:
                            blocker_condition_since = None
                            blocker_active = False
                            blocker.stop()
                        elif should_block:
                            if blocker_condition_since is None:
                                blocker_condition_since = now_ts
                            sustained = (now_ts - blocker_condition_since) >= self.BLOCKER_SUSTAIN_SECONDS
                            if sustained and not blocker_active:
                                blocker.start()
                                blocker_active = True
                        else:
                            blocker_condition_since = None
                            if blocker_active:
                                blocker.stop()
                                blocker_active = False

                        blocker_elapsed = 0.0 if blocker_condition_since is None else max(now_ts - blocker_condition_since, 0.0)
                        result.metadata["blocker_should_block"] = should_block
                        result.metadata["blocker_condition_elapsed_seconds"] = float(blocker_elapsed)
                        result.metadata["blocker_sustain_seconds"] = float(self.BLOCKER_SUSTAIN_SECONDS)
                        result.metadata["blocker_active"] = blocker_active

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

                    if result.posture_label == "slouch":
                        if active_slouch_instance is None:
                            image_name = f"{session_id}_slouch_{len(slouch_instances)+1:04d}.jpg"
                            image_path = slouch_instances_dir / image_name
                            labeled_frame = self._draw_slouch_snapshot_annotation(frame.copy(), result.metadata)
                            cv2.imwrite(str(image_path), labeled_frame)
                            active_slouch_instance = {
                                "session_id": session_id,
                                "start_timestamp_utc": ts,
                                "start_frame_idx": frame_idx,
                                "image_path": str(image_path),
                            }
                    elif active_slouch_instance is not None:
                        start_ts = datetime.fromisoformat(active_slouch_instance["start_timestamp_utc"])
                        end_ts = datetime.fromisoformat(ts)
                        duration_seconds = max((end_ts - start_ts).total_seconds(), 0.0)
                        active_slouch_instance["end_timestamp_utc"] = ts
                        active_slouch_instance["end_frame_idx"] = frame_idx
                        active_slouch_instance["duration_seconds"] = duration_seconds
                        slouch_instances.append(active_slouch_instance)
                        active_slouch_instance = None
                        persist_slouch_instances()

                    frame_to_draw = frame.copy()
                    self._draw_overlay(frame_to_draw, record)

                    if writer is None and self.record_path is not None:
                        self.record_path.parent.mkdir(parents=True, exist_ok=True)
                        h, w, _ = frame_to_draw.shape
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 20.0
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(self.record_path), fourcc, float(fps), (w, h))
                        if not writer.isOpened():
                            writer = None
                    if writer is not None:
                        writer.write(frame_to_draw)

                    if self.display:
                        cv2.imshow("NoSlouchBench Webcam", frame_to_draw)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if self.duration_minutes is not None:
                        elapsed_min = (time.time() - start) / 60.0
                        if elapsed_min >= self.duration_minutes:
                            break
        finally:
            if active_slouch_instance is not None:
                end_ts = datetime.now(timezone.utc).isoformat()
                start_ts = datetime.fromisoformat(active_slouch_instance["start_timestamp_utc"])
                duration_seconds = max(
                    (datetime.fromisoformat(end_ts) - start_ts).total_seconds(),
                    0.0,
                )
                active_slouch_instance["end_timestamp_utc"] = end_ts
                active_slouch_instance["end_frame_idx"] = frame_idx
                active_slouch_instance["duration_seconds"] = duration_seconds
                slouch_instances.append(active_slouch_instance)
                active_slouch_instance = None
                persist_slouch_instances()

            if beeper is not None:
                beeper.close()
            if blocker is not None:
                blocker.close()
            if swipe_guard is not None:
                swipe_guard.restore()
            if writer is not None:
                writer.release()
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
        persist_slouch_instances()

        return RunArtifacts(
            session_id=session_id,
            event_log_path=event_log_path,
            summary_path=summary_path,
            slouch_instances_path=slouch_instances_path,
            summary=summary,
        )

    def _open_camera_capture(self) -> tuple[cv2.VideoCapture, np.ndarray | None]:
        attempts: list[tuple[str, cv2.VideoCapture]] = []
        if platform.system() == "Darwin":
            attempts = [
                ("default", cv2.VideoCapture(self.camera_id)),
                ("avfoundation", cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)),
            ]
        else:
            attempts = [("default", cv2.VideoCapture(self.camera_id))]

        for _, cap in attempts:
            if not cap.isOpened():
                cap.release()
                continue

            ok, frame = cap.read()
            if ok:
                return cap, frame

            cap.release()

        attempted = ", ".join(name for name, _ in attempts)
        raise RuntimeError(
            f"Could not open webcam camera_id={self.camera_id} using backends: {attempted}"
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
        metadata = record.get("metadata", {})
        posture_color = (0, 0, 255) if label == "slouch" else (0, 255, 0)
        text_color = (0, 0, 0)
        cv2.putText(
            frame,
            f"Posture: {label}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            posture_color,
            2,
        )
        neck_len = metadata.get("neck_length_ratio")
        neck_len_threshold = metadata.get("neck_length_slouch_threshold")
        if neck_len is not None and neck_len_threshold is not None:
            cv2.putText(
                frame,
                f"NeckLen raw: {float(neck_len):.3f}, thresh: {float(neck_len_threshold):.3f}",
                (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
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
                    text_color,
                    1,
                )

    @staticmethod
    def _build_session_id(model_name: str, session_tag: str | None) -> str:
        t = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{session_tag}" if session_tag else ""
        return f"{model_name}_{t}{suffix}"

    @staticmethod
    def _draw_slouch_snapshot_annotation(frame: np.ndarray, metadata: dict | None) -> np.ndarray:
        md = metadata or {}
        h, w, _ = frame.shape
        pad = 18
        box_h = 82
        text_color = (0, 0, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        cv2.putText(
            frame,
            "BENT BACK (SLOUCH)",
            (pad, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            text_color,
            3,
        )
        score = md.get("slouch_score")
        thr = md.get("effective_slouch_threshold", md.get("slouch_score_threshold"))
        if score is not None and thr is not None:
            cv2.putText(
                frame,
                f"Score {float(score):.3f} > Threshold {float(thr):.3f}",
                (pad, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                text_color,
                2,
            )

        points = md.get("landmarks_norm", {}) or {}
        for name in ("left_ear", "right_ear", "left_shoulder", "right_shoulder"):
            xy = points.get(name)
            if not xy:
                continue
            x = int(float(xy[0]) * w)
            y = int(float(xy[1]) * h)
            cv2.circle(frame, (x, y), 6, (0, 220, 255), -1)
            cv2.putText(
                frame,
                name.replace("_", " "),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
            )

        side = str(md.get("selected_side", "")).lower()
        if side in {"left", "right"}:
            ear = points.get(f"{side}_ear")
            shoulder = points.get(f"{side}_shoulder")
            if ear and shoulder:
                p1 = (int(float(ear[0]) * w), int(float(ear[1]) * h))
                p2 = (int(float(shoulder[0]) * w), int(float(shoulder[1]) * h))
                cv2.line(frame, p1, p2, (0, 80, 255), 3)

        return frame
