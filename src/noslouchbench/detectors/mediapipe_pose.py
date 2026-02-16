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
        slouch_threshold: float = 0.08,
        task_model_path: str | None = None,
    ) -> None:
        self.slouch_threshold = slouch_threshold
        self.left_shoulder = 11
        self.right_shoulder = 12
        self.left_ear = 7
        self.right_ear = 8
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

        shoulder_y = (lm[self.left_shoulder].y + lm[self.right_shoulder].y) / 2.0
        ear_y = (lm[self.left_ear].y + lm[self.right_ear].y) / 2.0
        hip_y = (lm[self.left_hip].y + lm[self.right_hip].y) / 2.0

        torso_len = max(hip_y - shoulder_y, 1e-4)
        normalized_head_drop = (ear_y - shoulder_y) / torso_len
        slouch_score = normalized_head_drop - self.slouch_threshold

        posture_label = "slouch" if slouch_score > 0 else "upright"
        confidence = float(min(max(abs(slouch_score) / max(self.slouch_threshold, 1e-4), 0.0), 1.0))

        return DetectionResult(
            detected=True,
            posture_label=posture_label,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "normalized_head_drop": float(normalized_head_drop),
                "slouch_threshold": float(self.slouch_threshold),
                "torso_len": float(torso_len),
                "backend": self.backend,
            },
        )

    def close(self) -> None:
        if hasattr(self.pose, "close"):
            self.pose.close()
