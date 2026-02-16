from __future__ import annotations

import re
import shutil
import subprocess

import cv2


def resolve_camera_id(camera_id: int, camera_name: str | None) -> int:
    if not camera_name:
        return camera_id

    wanted = camera_name.strip().lower()
    resolved = _resolve_with_ffmpeg_avfoundation(wanted)
    if resolved is not None:
        return resolved

    # Fallback heuristic: if external camera is plugged in, it is often index 1.
    if wanted in {"logitech", "logitech c270", "c270"}:
        for idx in (1, 2, 3, 0):
            cap = cv2.VideoCapture(idx)
            ok = cap.isOpened()
            cap.release()
            if ok:
                return idx

    return camera_id


def _resolve_with_ffmpeg_avfoundation(wanted: str) -> int | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    try:
        proc = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    # ffmpeg writes device list to stderr for avfoundation.
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    lines = text.splitlines()
    in_video_section = False
    for line in lines:
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if in_video_section and "AVFoundation audio devices" in line:
            break
        if not in_video_section:
            continue

        m = re.search(r"\[(\d+)\]\s+(.*)$", line)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip().lower()
        if wanted in name:
            return idx
    return None

