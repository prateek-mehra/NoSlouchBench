from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectionResult:
    detected: bool
    posture_label: str
    confidence: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePostureDetector:
    name = "base"

    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None

