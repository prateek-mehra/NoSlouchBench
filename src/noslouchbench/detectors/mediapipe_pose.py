from __future__ import annotations

import time

import cv2
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
    ) -> None:
        self.slouch_threshold = slouch_threshold
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.left_shoulder = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        self.right_shoulder = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        self.left_ear = self.mp_pose.PoseLandmark.LEFT_EAR.value
        self.right_ear = self.mp_pose.PoseLandmark.RIGHT_EAR.value
        self.left_hip = self.mp_pose.PoseLandmark.LEFT_HIP.value
        self.right_hip = self.mp_pose.PoseLandmark.RIGHT_HIP.value

    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results.pose_landmarks:
            return DetectionResult(
                detected=False,
                posture_label="unknown",
                confidence=0.0,
                latency_ms=latency_ms,
                metadata={"reason": "no_pose_landmarks"},
            )

        lm = results.pose_landmarks.landmark
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
            },
        )

    def close(self) -> None:
        self.pose.close()

