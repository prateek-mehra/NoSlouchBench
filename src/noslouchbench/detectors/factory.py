from __future__ import annotations

from noslouchbench.detectors.base import BasePostureDetector
from noslouchbench.detectors.mediapipe_pose import MediaPipePostureDetector


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

    if key in {"yolo-pose", "openpose"}:
        raise NotImplementedError(
            f"Model '{model_name}' is scaffolded but not implemented yet. "
            "Start with --model mediapipe."
        )

    raise ValueError(f"Unsupported model: {model_name}")
