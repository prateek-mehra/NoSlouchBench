#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import urllib.request


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
OUT_PATH = Path("models/mediapipe/pose_landmarker_lite.task")


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {OUT_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, OUT_PATH)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

