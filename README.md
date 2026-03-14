# NoSlouchBench

NoSlouchBench is a reproducible benchmarking framework for evaluating posture detection pipelines built on top of MediaPipe, YOLO-Pose, OpenPose, and similar pose estimation backbones.

It standardizes:
- Dataset splits
- Posture classification logic
- Evaluation metrics, including accuracy, robustness, and inference latency

The goal is fair, deployment-aware comparison of real-time posture detection systems.

## Demo Video

<video src="https://github.com/prateek-mehra/NoSlouchBench/raw/main/outputs/demo/final_demo.mp4" controls muted playsinline width="900"></video>

![Demo GIF](https://github.com/prateek-mehra/NoSlouchBench/raw/main/outputs/demo/final_demo.gif)

## Current Status

This repository is now set up for live webcam benchmarking with:
- `yolo-pose` (implemented, default)
- `mediapipe` (implemented, optional fallback)
- `openpose` (scaffolded, not yet implemented)

## Quick Start (Webcam)

1. Create and activate a Python 3.11 environment (recommended).
2. Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

YOLO-Pose model weights are auto-downloaded by Ultralytics on first run.

If MediaPipe fallback is needed, force reinstall:

```bash
pip install --upgrade --force-reinstall "mediapipe>=0.10.14,<0.10.31"
```

Download the MediaPipe Tasks pose model only if using `--model mediapipe` with tasks-only MediaPipe:

```bash
python scripts/download_pose_landmarker.py
```

3. List cameras first, then pick the appropriate one:

```bash
PYTHONPATH=src python -m noslouchbench.cli list-cameras
```

Then run YOLO-Pose benchmark on webcam (default model):

```bash
PYTHONPATH=src python3 -m noslouchbench.cli run-webcam --model yolo-pose --camera-id 0  --display --screen-blocker
```

For Logitech C270 specifically (or any camera name shown above):

```bash
PYTHONPATH=src python -m noslouchbench.cli run-webcam --model yolo-pose --camera-name "Logitech C270" --display
```

By default, a continuous beep is played while posture is classified as `slouch`, and it stops when posture becomes `upright`.
To disable audio alerts:

```bash
PYTHONPATH=src python -m noslouchbench.cli run-webcam --model yolo-pose --camera-name "Logitech C270" --display --no-beep
```

Final run command:

```bash
PYTHONPATH=src python3 -m noslouchbench.cli run-webcam --model yolo-pose --camera-id 0  --display --screen-blocker
```

When slouch is detected, a semi-transparent full-screen layer appears and is removed once posture returns to `upright`.
Kill switch (for safety): `Ctrl+Shift+K` (can be changed with `--screen-blocker-kill-switch`).
While screen-blocker mode is enabled, 3-finger horizontal swipe gesture is temporarily disabled for the session and restored on exit.  
To skip this behavior, pass `--no-lock-swipe-gesture`.

To record a debug session (with overlays) for later analysis:

```bash
PYTHONPATH=src python -m noslouchbench.cli run-webcam --model yolo-pose --camera-name "Logitech C270" --display --record-path outputs/debug/repro.mp4
```

Then analyze that saved video:

```bash
PYTHONPATH=src python -m noslouchbench.cli analyze-video --model yolo-pose --input-video outputs/debug/repro.mp4
```

4. For a long-run weekly session, you can keep it running with no duration cap:

```bash
PYTHONPATH=src python3 -m noslouchbench.cli run-webcam --model yolo-pose --camera-id 0  --display --screen-blocker
```

Press `q` in the display window to stop.

## Outputs

Each run generates:
- Frame-level logs: `outputs/sessions/<session_id>.jsonl`
- Session summary: `outputs/summaries/<session_id>.json`
- Slouch instances + snapshots: `outputs/slouch_instances/<session_id>.json` and `outputs/slouch_instances/*.jpg`

To aggregate model summaries after a week:

```bash
PYTHONPATH=src python -m noslouchbench.cli summarize --input-dir outputs/summaries
```

This produces:
- `outputs/reports/model_comparison.csv`
- `outputs/reports/model_comparison.md`

## Habit Reports (Daily/Weekly)

Generate human-readable daily and weekly posture stats:

```bash
PYTHONPATH=src python -m noslouchbench.cli summarize-habits --show
```

This produces:
- `outputs/reports/habits_daily.csv`
- `outputs/reports/habits_weekly.csv`
- `outputs/reports/habits_daily.md`
- `outputs/reports/habits_weekly.md`

Reported metrics:
- `Hours Sitting`: total webcam runtime
- `Slouch Minutes`: `duration_seconds * slouch_ratio`
- `Beep Events`: number of transitions into `slouch` (slouch episodes)
- `Sessions`: number of runs in that day/week bucket

If a session log is missing, beep count falls back to `0` and CSV marks `beep_source=missing_log`.

## Streamlit Dashboard

View habit reports locally in a basic UI:

```bash
PYTHONPATH=src streamlit run streamlit_app.py
```

In the UI:
- Set summaries/sessions/reports directories in the sidebar.
- Click `Refresh Reports` to regenerate daily/weekly files.
- View latest daily/weekly headline metrics and full tables.
- View day-wise and week-wise slouch instances with:
  - timestamp
  - duration (seconds)
  - slouch snapshot image

## Automation (cron)

Helper script:

```bash
scripts/schedule_reports.sh
```

Example crontab (local timezone):

```cron
# Refresh local report artifacts daily at 9:00 PM
0 21 * * * cd /Users/prateek/Downloads/_Projects/Personal/codex/NoSlouchBench && /usr/bin/env bash scripts/schedule_reports.sh >> outputs/reports/cron_reports.log 2>&1
```

## Start In Background At Login (macOS)

Use the launchd helper to keep NoSlouchBench running in the background after login and restart it automatically if it exits.

Install + start with default args:

```bash
scripts/autostart_macos.sh install
```

Default run args are:

```bash
run-webcam --model yolo-pose --camera-id 0 --screen-blocker
```

Install with custom `run-webcam` args:

```bash
scripts/autostart_macos.sh install -- --camera-id 1 --model yolo-pose --screen-blocker
```

Control lifecycle:

```bash
scripts/autostart_macos.sh status
scripts/autostart_macos.sh stop       # stop now (manual close)
scripts/autostart_macos.sh start      # start again
scripts/autostart_macos.sh uninstall  # remove startup entry
```

Logs are written to:

```text
outputs/startup/stdout.log
outputs/startup/stderr.log
```

Note: startup runs through a hidden Terminal-hosted runner so macOS camera permissions are honored.

## Benchmark Metrics Tracked

- Average inference latency (ms)
- P95 inference latency (ms)
- Detection rate (robustness proxy)
- Slouch ratio
- Upright ratio

## Posture Rule (YOLO-Pose)

`yolo-pose` currently classifies slouch by a normalized geometric rule:

- Designed for side-view webcam setups.
- Uses only body-side landmarks (ear, shoulder, hip) and picks the more visible side.
- Requires only upper-body landmarks to classify:
  one ear + one shoulder on the same side (hips are optional).
- Computes a weighted slouch score from:
  - head-forward offset (`|ear_x - shoulder_x| / torso_len`)
  - torso lean angle (shoulder-hip line vs vertical)
  - neck drop (`max(0, ear_y - shoulder_y) / torso_len`)
- Classifies as `slouch` when `slouch_score >= slouch_threshold`.

Default threshold is `0.12` (strict and highly sensitive to slouch).  
Tune in `/Users/prateek/Downloads/_Projects/Personal/codex/NoSlouchBench/configs/models.yaml` under `models.yolo-pose.slouch_threshold`.

## Implementation Notes

- YOLO-Pose is now the primary detector for webcam benchmarking.
- Posture logic uses side-view geometry from ear/shoulder/hip landmarks with configurable slouch thresholding.
