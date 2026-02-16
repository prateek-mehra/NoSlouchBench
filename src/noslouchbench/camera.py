from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class CameraDevice:
    idx: int
    name: str


def resolve_camera_id(camera_id: int, camera_name: str | None) -> int:
    if not camera_name:
        return camera_id

    wanted = camera_name.strip()
    devices = list_cameras_avfoundation()
    resolved = _pick_best_match(wanted, devices)
    if resolved is not None:
        return resolved

    available = ", ".join([f"[{d.idx}] {d.name}" for d in devices]) or "none found"
    raise RuntimeError(
        f'Camera name "{camera_name}" not found. Available video devices: {available}. '
        "Use --camera-id explicitly if needed."
    )


def list_cameras_avfoundation() -> list[CameraDevice]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []

    try:
        proc = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    # ffmpeg writes device list to stderr for avfoundation.
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    lines = text.splitlines()
    in_video_section = False
    devices: list[CameraDevice] = []
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
        devices.append(CameraDevice(idx=int(m.group(1)), name=m.group(2).strip()))
    return devices


def _pick_best_match(wanted: str, devices: list[CameraDevice]) -> int | None:
    if not devices:
        return None

    wanted_norm = _norm(wanted)
    wanted_tokens = set(wanted_norm.split())

    best: tuple[int, int] | None = None  # (score, idx)
    for d in devices:
        name_norm = _norm(d.name)
        name_tokens = set(name_norm.split())

        score = 0
        if wanted_norm == name_norm:
            score += 100
        if wanted_norm in name_norm or name_norm in wanted_norm:
            score += 50

        common = wanted_tokens.intersection(name_tokens)
        score += 10 * len(common)
        if "c270" in common:
            score += 30
        if "logitech" in wanted_tokens and "logitech" in name_tokens:
            score += 20

        # Avoid accidentally choosing built-in camera when searching for external.
        if "logitech" in wanted_tokens or "c270" in wanted_tokens:
            if "facetime" in name_tokens or "built" in name_tokens:
                score -= 40

        if score <= 0:
            continue
        candidate = (score, d.idx)
        if best is None or candidate > best:
            best = candidate

    return best[1] if best is not None else None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
