from __future__ import annotations

import time

import numpy as np

from noslouchbench.detectors.base import BasePostureDetector, DetectionResult


class YoloPosePostureDetector(BasePostureDetector):
    name = "yolo-pose"

    # COCO keypoints used by YOLO pose models.
    KP_LEFT_EAR = 3
    KP_RIGHT_EAR = 4
    KP_LEFT_SHOULDER = 5
    KP_RIGHT_SHOULDER = 6
    KP_LEFT_HIP = 11
    KP_RIGHT_HIP = 12

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        keypoint_confidence_threshold: float = 0.25,
        slouch_threshold: float = 0.30,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "YOLO-Pose backend requires ultralytics. Install with: pip install ultralytics"
            ) from e

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self.slouch_threshold = slouch_threshold
        self.imgsz = imgsz
        self.device = device
        self.min_visibility = keypoint_confidence_threshold

    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            return DetectionResult(
                detected=False,
                posture_label="unknown",
                confidence=0.0,
                latency_ms=latency_ms,
                metadata={"reason": "no_result", "backend": "yolo-pose"},
            )

        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            return DetectionResult(
                detected=False,
                posture_label="unknown",
                confidence=0.0,
                latency_ms=latency_ms,
                metadata={"reason": "no_keypoints", "backend": "yolo-pose"},
            )

        kxy_all = result.keypoints.xy
        kcf_all = result.keypoints.conf
        person_idx = self._select_person(kxy_all, kcf_all)

        kxy = kxy_all[person_idx].cpu().numpy()
        if kcf_all is None:
            kcf = np.ones((kxy.shape[0],), dtype=np.float32)
        else:
            kcf = kcf_all[person_idx].cpu().numpy()

        h, w, _ = frame_bgr.shape

        def point(idx: int) -> tuple[float, float, float]:
            return (
                float(kxy[idx][0] / max(w, 1)),
                float(kxy[idx][1] / max(h, 1)),
                float(kcf[idx]),
            )

        tracked = {
            "left_ear": point(self.KP_LEFT_EAR),
            "right_ear": point(self.KP_RIGHT_EAR),
            "left_shoulder": point(self.KP_LEFT_SHOULDER),
            "right_shoulder": point(self.KP_RIGHT_SHOULDER),
            "left_hip": point(self.KP_LEFT_HIP),
            "right_hip": point(self.KP_RIGHT_HIP),
        }

        left_upper_ok = min(
            tracked["left_ear"][2], tracked["left_shoulder"][2]
        ) >= self.min_visibility
        right_upper_ok = min(
            tracked["right_ear"][2], tracked["right_shoulder"][2]
        ) >= self.min_visibility

        if not left_upper_ok and not right_upper_ok:
            return DetectionResult(
                detected=False,
                posture_label="unknown",
                confidence=0.0,
                latency_ms=latency_ms,
                metadata={"reason": "insufficient_upper_body_landmarks", "backend": "yolo-pose"},
            )

        if left_upper_ok and right_upper_ok:
            left_vis = tracked["left_ear"][2] + tracked["left_shoulder"][2] + 0.5 * tracked["left_hip"][2]
            right_vis = tracked["right_ear"][2] + tracked["right_shoulder"][2] + 0.5 * tracked["right_hip"][2]
            side = "left" if left_vis >= right_vis else "right"
        else:
            side = "left" if left_upper_ok else "right"

        ear = tracked[f"{side}_ear"]
        shoulder = tracked[f"{side}_shoulder"]
        hip = tracked[f"{side}_hip"]

        torso_len, scale_source = self._compute_scale(tracked, side=side)
        neck_drop = max(0.0, (ear[1] - shoulder[1]) / torso_len)
        head_forward_offset = abs(ear[0] - shoulder[0]) / torso_len

        if hip[2] >= self.min_visibility:
            dx = shoulder[0] - hip[0]
            dy = hip[1] - shoulder[1]
            torso_lean_angle_deg = float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-4))))
            lean_source = f"{side}_shoulder_to_{side}_hip"
        else:
            torso_lean_angle_deg = 0.0
            lean_source = "hip_missing"
        torso_lean_norm = min(torso_lean_angle_deg / 35.0, 1.0)

        slouch_score = 0.45 * head_forward_offset + 0.30 * torso_lean_norm + 0.25 * neck_drop
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
                "scale_source": scale_source,
                "lean_source": lean_source,
                "selected_side": side,
                "backend": "yolo-pose",
                "landmarks_norm": {k: [float(v[0]), float(v[1])] for k, v in tracked.items()},
            },
        )

    @staticmethod
    def _select_person(kxy_all, kcf_all) -> int:
        num_people = len(kxy_all)
        if num_people == 1:
            return 0
        if kcf_all is None:
            return 0
        means = []
        for i in range(num_people):
            means.append(float(kcf_all[i].mean().item()))
        return int(np.argmax(np.array(means)))

    def _compute_scale(self, tracked: dict[str, tuple[float, float, float]], side: str) -> tuple[float, str]:
        shoulder = tracked[f"{side}_shoulder"]
        hip = tracked[f"{side}_hip"]
        if hip[2] >= self.min_visibility:
            return max(hip[1] - shoulder[1], 1e-4), f"{side}_torso"

        other_side = "right" if side == "left" else "left"
        other_shoulder = tracked[f"{other_side}_shoulder"]
        shoulder_span = abs(other_shoulder[0] - shoulder[0])
        if other_shoulder[2] >= self.min_visibility and shoulder_span > 1e-4:
            # Shoulder span fallback when hips are not confidently visible.
            return max(shoulder_span, 1e-4), "shoulder_span"

        return 0.15, "default_constant"

    def close(self) -> None:
        return None
