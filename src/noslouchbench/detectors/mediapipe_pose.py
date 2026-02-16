from __future__ import annotations

import os
from pathlib import Path
import time

import cv2

# Prefer CPU execution for broader compatibility in local webcam benchmarking.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import mediapipe as mp
import numpy as np

from noslouchbench.detectors.base import BasePostureDetector, DetectionResult


class MediaPipePostureDetector(BasePostureDetector):
    name = "mediapipe"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        slouch_threshold: float = 0.38,
        task_model_path: str | None = None,
    ) -> None:
        self.slouch_threshold = slouch_threshold
        self.min_visibility = 0.2
        self.left_ear = 7
        self.right_ear = 8
        self.left_shoulder = 11
        self.right_shoulder = 12
        self.left_hip = 23
        self.right_hip = 24

        if hasattr(mp, "solutions"):
            self.backend = "solutions"
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        self.backend = "tasks"
        self._init_tasks_backend(
            task_model_path=task_model_path,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _init_tasks_backend(
        self,
        task_model_path: str | None,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        model_path = Path(task_model_path or "models/mediapipe/pose_landmarker_lite.task")
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe Tasks model not found at: {model_path}. "
                "Download it first (see README: scripts/download_pose_landmarker.py)."
            )

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as e:
            raise RuntimeError(
                "Installed mediapipe package does not provide either `solutions` or tasks vision APIs."
            ) from e

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._mp_image_fmt = mp.ImageFormat

    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            results = self.pose.process(frame_rgb)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if not results.pose_landmarks:
                return DetectionResult(
                    detected=False,
                    posture_label="unknown",
                    confidence=0.0,
                    latency_ms=latency_ms,
                    metadata={"reason": "no_pose_landmarks", "backend": self.backend},
                )
            lm = results.pose_landmarks.landmark
        else:
            mp_image = self._mp_image_cls(image_format=self._mp_image_fmt.SRGB, data=frame_rgb)
            results = self.pose.detect(mp_image)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if not results.pose_landmarks:
                return DetectionResult(
                    detected=False,
                    posture_label="unknown",
                    confidence=0.0,
                    latency_ms=latency_ms,
                    metadata={"reason": "no_pose_landmarks", "backend": self.backend},
                )
            lm = results.pose_landmarks[0]

        tracked = {
            "left_ear": self._lm_xyv(lm[self.left_ear]),
            "right_ear": self._lm_xyv(lm[self.right_ear]),
            "left_shoulder": self._lm_xyv(lm[self.left_shoulder]),
            "right_shoulder": self._lm_xyv(lm[self.right_shoulder]),
            "left_hip": self._lm_xyv(lm[self.left_hip]),
            "right_hip": self._lm_xyv(lm[self.right_hip]),
        }

        core_points = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if min(tracked[k][2] for k in core_points) < self.min_visibility:
            return DetectionResult(
                detected=False,
                posture_label="unknown",
                confidence=0.0,
                latency_ms=latency_ms,
                metadata={"reason": "low_landmark_visibility", "backend": self.backend},
            )

        left_side_ok = min(
            tracked["left_ear"][2], tracked["left_shoulder"][2], tracked["left_hip"][2]
        ) >= self.min_visibility
        right_side_ok = min(
            tracked["right_ear"][2], tracked["right_shoulder"][2], tracked["right_hip"][2]
        ) >= self.min_visibility

        if not left_side_ok and not right_side_ok:
            return DetectionResult(
                detected=False,
                posture_label="unknown",
                confidence=0.0,
                latency_ms=latency_ms,
                metadata={
                    "reason": "insufficient_side_landmarks",
                    "backend": self.backend,
                },
            )

        if left_side_ok and right_side_ok:
            left_vis = tracked["left_ear"][2] + tracked["left_shoulder"][2] + tracked["left_hip"][2]
            right_vis = tracked["right_ear"][2] + tracked["right_shoulder"][2] + tracked["right_hip"][2]
            side = "left" if left_vis >= right_vis else "right"
        else:
            side = "left" if left_side_ok else "right"

        ear = tracked[f"{side}_ear"]
        shoulder = tracked[f"{side}_shoulder"]
        hip = tracked[f"{side}_hip"]

        shoulder_y = shoulder[1]
        hip_y = hip[1]

        torso_len = max(hip_y - shoulder_y, 1e-4)
        # Side-view slouch cues:
        # 1) Neck drop: ear moving below shoulder.
        # 2) Head-forward offset: ear horizontal offset from shoulder.
        # 3) Torso lean angle: shoulder-hip line deviating from vertical.
        neck_drop = max(0.0, (ear[1] - shoulder[1]) / torso_len)
        head_forward_offset = abs(ear[0] - shoulder[0]) / torso_len

        dx = shoulder[0] - hip[0]
        dy = hip[1] - shoulder[1]
        torso_lean_angle_deg = float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-4))))
        torso_lean_norm = min(torso_lean_angle_deg / 35.0, 1.0)

        slouch_score = (
            0.45 * head_forward_offset
            + 0.30 * torso_lean_norm
            + 0.25 * neck_drop
        )

        posture_label = "slouch" if slouch_score >= self.slouch_threshold else "upright"
        confidence = float(
            min(max(abs(slouch_score - self.slouch_threshold) / max(self.slouch_threshold, 1e-4), 0.0), 1.0)
        )

        return DetectionResult(
            detected=True,
            posture_label=posture_label,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "slouch_score": float(slouch_score),
                "slouch_score_threshold": float(self.slouch_threshold),
                "head_forward_offset": float(head_forward_offset),
                "torso_lean_angle_deg": float(torso_lean_angle_deg),
                "neck_drop": float(neck_drop),
                "torso_len": float(torso_len),
                "selected_side": side,
                "backend": self.backend,
                "landmarks_norm": {k: [float(v[0]), float(v[1])] for k, v in tracked.items()},
            },
        )

    def close(self) -> None:
        if hasattr(self.pose, "close"):
            self.pose.close()

    @staticmethod
    def _lm_xyv(landmark) -> tuple[float, float, float]:
        return (
            float(landmark.x),
            float(landmark.y),
            float(getattr(landmark, "visibility", 1.0)),
        )
