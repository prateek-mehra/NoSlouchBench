# NoSlouchBench

NoSlouchBench is a reproducible benchmarking framework for evaluating posture detection pipelines built on top of MediaPipe, YOLO-Pose, OpenPose, and similar pose estimation backbones.

It standardizes:
- Dataset splits
- Posture classification logic
- Evaluation metrics, including accuracy, robustness, and inference latency

The goal is fair, deployment-aware comparison of real-time posture detection systems.

## Current Status

This repository is now set up for live webcam benchmarking with the first model:
- `mediapipe` (implemented)
- `yolo-pose` (scaffolded, not yet implemented)
- `openpose` (scaffolded, not yet implemented)

## Quick Start (Webcam)

1. Create and activate a Python 3.11 environment (recommended).
2. Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If MediaPipe was already installed in this environment, force reinstall:

```bash
pip install --upgrade --force-reinstall "mediapipe>=0.10.14,<0.10.31"
```

Download the MediaPipe Tasks pose model (required when your install is tasks-only):

```bash
python scripts/download_pose_landmarker.py
```

3. Run MediaPipe benchmark on webcam:

```bash
PYTHONPATH=src python -m noslouchbench.cli run-webcam --model mediapipe --duration-minutes 60 --display
```

4. For a long-run weekly session, you can keep it running with no duration cap:

```bash
PYTHONPATH=src python -m noslouchbench.cli run-webcam --model mediapipe --display
```

Press `q` in the display window to stop.

## Outputs

Each run generates:
- Frame-level logs: `outputs/sessions/<session_id>.jsonl`
- Session summary: `outputs/summaries/<session_id>.json`

To aggregate model summaries after a week:

```bash
PYTHONPATH=src python -m noslouchbench.cli summarize --input-dir outputs/summaries
```

This produces:
- `outputs/reports/model_comparison.csv`
- `outputs/reports/model_comparison.md`

## Benchmark Metrics Tracked

- Average inference latency (ms)
- P95 inference latency (ms)
- Detection rate (robustness proxy)
- Slouch ratio
- Upright ratio

## Posture Rule (MediaPipe)

`mediapipe` currently classifies slouch by a normalized geometric rule:

- Designed for side-view webcam setups.
- Uses only body-side landmarks (ear, shoulder, hip) and picks the more visible side.
- Computes a weighted slouch score from:
  - head-forward offset (`|ear_x - shoulder_x| / torso_len`)
  - torso lean angle (shoulder-hip line vs vertical)
  - neck drop (`max(0, ear_y - shoulder_y) / torso_len`)
- Classifies as `slouch` when `slouch_score >= slouch_threshold`.

Default threshold is `0.38` (tuned for side camera).  
Tune in `/Users/prateek/Downloads/_Projects/Personal/codex/NoSlouchBench/configs/models.yaml` under `models.mediapipe.slouch_threshold`.

## Implementation Notes

- The MediaPipe detector is adapted from the same local-inference approach used in your `focusguard` codebase (`mediapipe` + webcam loop).
- Posture logic currently uses pose landmarks and a configurable slouch threshold based on head-vs-shoulder vertical offset normalized by torso length.
