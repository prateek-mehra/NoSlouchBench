#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_model_week.sh yolo-pose "Logitech C270"

MODEL="${1:-yolo-pose}"
CAMERA_NAME="${2:-Logitech C270}"
SESSION_TAG="${3:-week_run}"

PYTHONPATH=src python3 -m noslouchbench.cli run-webcam \
  --model "${MODEL}" \
  --camera-name "${CAMERA_NAME}" \
  --session-tag "${SESSION_TAG}" \
  --display
