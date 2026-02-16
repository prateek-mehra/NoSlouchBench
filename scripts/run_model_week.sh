#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_model_week.sh yolo-pose

MODEL="${1:-yolo-pose}"
SESSION_TAG="${2:-week_run}"

PYTHONPATH=src python3 -m noslouchbench.cli run-webcam \
  --model "${MODEL}" \
  --session-tag "${SESSION_TAG}" \
  --display
