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

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run MediaPipe benchmark on webcam:

```bash
python -m noslouchbench.cli run-webcam --model mediapipe --duration-minutes 60 --display
```

4. For a long-run weekly session, you can keep it running with no duration cap:

```bash
python -m noslouchbench.cli run-webcam --model mediapipe --display
```

Press `q` in the display window to stop.

## Outputs

Each run generates:
- Frame-level logs: `outputs/sessions/<session_id>.jsonl`
- Session summary: `outputs/summaries/<session_id>.json`

To aggregate model summaries after a week:

```bash
python -m noslouchbench.cli summarize --input-dir outputs/summaries
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

## Implementation Notes

- The MediaPipe detector is adapted from the same local-inference approach used in your `focusguard` codebase (`mediapipe` + webcam loop).
- Posture logic currently uses pose landmarks and a configurable slouch threshold based on head-vs-shoulder vertical offset normalized by torso length.
