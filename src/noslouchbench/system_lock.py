from __future__ import annotations

import platform
import re
import subprocess


class TrackpadSwipeGuard:
    """Temporarily disable desktop/app swipe gestures on macOS and restore on exit."""

    _KEYS = [
        # Per-device trackpad domains.
        ("com.apple.AppleMultitouchTrackpad", "TrackpadThreeFingerHorizSwipeGesture", False),
        ("com.apple.driver.AppleBluetoothMultitouch.trackpad", "TrackpadThreeFingerHorizSwipeGesture", False),
        ("com.apple.AppleMultitouchTrackpad", "TrackpadFourFingerHorizSwipeGesture", False),
        ("com.apple.driver.AppleBluetoothMultitouch.trackpad", "TrackpadFourFingerHorizSwipeGesture", False),
        # Global currentHost keys often drive Mission Control swipe behavior.
        ("NSGlobalDomain", "com.apple.trackpad.threeFingerHorizSwipeGesture", True),
        ("NSGlobalDomain", "com.apple.trackpad.fourFingerHorizSwipeGesture", True),
        # Non-currentHost mirror keys on some macOS setups.
        ("NSGlobalDomain", "com.apple.trackpad.threeFingerHorizSwipeGesture", False),
        ("NSGlobalDomain", "com.apple.trackpad.fourFingerHorizSwipeGesture", False),
    ]

    def __init__(self) -> None:
        self._active = False
        self._supported = platform.system() == "Darwin"
        self._snapshot: dict[tuple[str, str, bool], str | None] = {}

    @property
    def supported(self) -> bool:
        return self._supported

    def activate(self) -> None:
        if not self._supported or self._active:
            return

        for domain, key, current_host in self._KEYS:
            self._snapshot[(domain, key, current_host)] = self._read_default(domain, key, current_host)
            self._write_default_int(domain, key, 0, current_host)

        self._restart_dock()
        self._active = True

    def restore(self) -> None:
        if not self._supported or not self._active:
            return

        for domain, key, current_host in self._KEYS:
            prev = self._snapshot.get((domain, key, current_host))
            if prev is None:
                self._delete_default(domain, key, current_host)
                continue

            if re.fullmatch(r"-?\d+", prev):
                self._write_default_int(domain, key, int(prev), current_host)
            else:
                self._write_default_string(domain, key, prev, current_host)

        self._restart_dock()
        self._active = False

    def _read_default(self, domain: str, key: str, current_host: bool) -> str | None:
        cmd = ["defaults"]
        if current_host:
            cmd.append("-currentHost")
        cmd.extend(["read", domain, key])
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        return (proc.stdout or "").strip() or None

    def _write_default_int(self, domain: str, key: str, value: int, current_host: bool) -> None:
        cmd = ["defaults"]
        if current_host:
            cmd.append("-currentHost")
        cmd.extend(["write", domain, key, "-int", str(value)])
        subprocess.run(cmd, check=False)

    def _write_default_string(self, domain: str, key: str, value: str, current_host: bool) -> None:
        cmd = ["defaults"]
        if current_host:
            cmd.append("-currentHost")
        cmd.extend(["write", domain, key, "-string", value])
        subprocess.run(cmd, check=False)

    def _delete_default(self, domain: str, key: str, current_host: bool) -> None:
        cmd = ["defaults"]
        if current_host:
            cmd.append("-currentHost")
        cmd.extend(["delete", domain, key])
        subprocess.run(cmd, check=False)

    def _restart_dock(self) -> None:
        subprocess.run(["killall", "cfprefsd"], check=False, capture_output=True)
        subprocess.run(["killall", "Dock"], check=False, capture_output=True)
        subprocess.run(["killall", "SystemUIServer"], check=False, capture_output=True)
