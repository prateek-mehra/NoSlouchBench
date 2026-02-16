from __future__ import annotations

from noslouchbench.detectors.base import BasePostureDetector
from noslouchbench.detectors.mediapipe_pose import MediaPipePostureDetector
from noslouchbench.detectors.yolo_pose import YoloPosePostureDetector


def build_detector(model_name: str, model_cfg: dict | None = None) -> BasePostureDetector:
    cfg = model_cfg or {}
    key = model_name.lower()

    if key == "mediapipe":
        return MediaPipePostureDetector(
            min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
            slouch_threshold=float(cfg.get("slouch_threshold", 0.08)),
            task_model_path=cfg.get("task_model_path"),
        )

    if key == "yolo-pose":
        return YoloPosePostureDetector(
            model_path=str(cfg.get("model_path", "yolo11n-pose.pt")),
            conf_threshold=float(cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(cfg.get("iou_threshold", 0.45)),
            keypoint_confidence_threshold=float(cfg.get("keypoint_confidence_threshold", 0.25)),
            slouch_threshold=float(cfg.get("slouch_threshold", 0.38)),
            imgsz=int(cfg.get("imgsz", 640)),
            device=str(cfg.get("device", "cpu")),
        )

    if key == "openpose":
        raise NotImplementedError(
            f"Model '{model_name}' is scaffolded but not implemented yet. "
            "Use --model yolo-pose."
        )

    raise ValueError(f"Unsupported model: {model_name}")
