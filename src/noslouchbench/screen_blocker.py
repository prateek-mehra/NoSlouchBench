from __future__ import annotations

import argparse
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


class ScreenBlocker:
    """Process-backed screen blocker with smooth show/hide commands."""

    def __init__(self, opacity: float = 0.78, kill_switch: str = "Ctrl+Shift+K") -> None:
        self.opacity = min(max(float(opacity), 0.20), 0.95)
        self.kill_switch = kill_switch
        self._proc: subprocess.Popen | None = None
        self._closed = False
        self._available = True
        self._visible = False
        self._kill_marker = Path(tempfile.gettempdir()) / f"noslouchbench_blocker_killed_{Path.cwd().name}_{id(self)}.flag"
        try:
            self._kill_marker.unlink(missing_ok=True)
        except Exception:
            pass

    def start(self) -> None:
        if self._closed or not self._available or self.killed:
            return
        if not self._ensure_proc():
            return
        self._send_cmd("SHOW")
        self._visible = True

    def stop(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        self._send_cmd("HIDE")
        self._visible = False

    def close(self) -> None:
        self._closed = True
        if self._proc is None:
            return

        if self._proc.poll() is None:
            self._send_cmd("EXIT")
            try:
                self._proc.wait(timeout=1.0)
            except Exception:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except Exception:
                    self._proc.kill()

        self._proc = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def killed(self) -> bool:
        return self._kill_marker.exists()

    def _ensure_proc(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True

        for cmd in self._build_blocker_commands():
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
            except Exception:
                continue

            # If child exits immediately (e.g., swift toolchain/env issue), fallback.
            time.sleep(0.12)
            if proc.poll() is not None:
                continue

            self._proc = proc
            return True

        self._available = False
        self._proc = None
        return False

    def _send_cmd(self, cmd: str) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        if self._proc.stdin is None:
            return

        try:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except Exception:
            self._available = False

    def _build_blocker_commands(self) -> list[list[str]]:
        commands: list[list[str]] = []

        # On macOS, prefer a native AppKit overlay across all Spaces so swipe
        # gestures cannot escape to another desktop while blocker is active.
        if platform.system() == "Darwin":
            swift = shutil.which("swift")
            script = Path(__file__).resolve().parents[2] / "scripts" / "screen_blocker.swift"
            if swift and script.exists():
                commands.append([
                    swift,
                    str(script),
                    "--opacity",
                    str(self.opacity),
                    "--kill-switch",
                    self.kill_switch,
                    "--kill-marker",
                    str(self._kill_marker),
                ])

        # Fallback to python/tk implementation.
        commands.append([
            sys.executable,
            "-m",
            "noslouchbench.screen_blocker",
            "--opacity",
            str(self.opacity),
            "--kill-switch",
            self.kill_switch,
            "--kill-marker",
            str(self._kill_marker),
        ])
        return commands


def _to_tk_binding(kill_switch: str) -> str:
    token_map = {
        "ctrl": "Control",
        "control": "Control",
        "shift": "Shift",
        "alt": "Alt",
        "option": "Alt",
        "cmd": "Command",
        "command": "Command",
    }
    parts = [p.strip() for p in kill_switch.replace("+", " ").split() if p.strip()]
    if not parts:
        return "<Control-Shift-K>"

    *mods, key = parts
    mapped_mods = [token_map.get(m.lower(), m) for m in mods]
    key = key.lower()
    chain = "-".join([*mapped_mods, key])
    return f"<{chain}>"


def blocker_process_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NoSlouchBench screen blocker process")
    parser.add_argument("--opacity", type=float, default=0.78)
    parser.add_argument("--kill-switch", default="Ctrl+Shift+K")
    parser.add_argument("--kill-marker", required=True)
    args = parser.parse_args(argv)

    try:
        import tkinter as tk
    except Exception as exc:
        print(f"Screen blocker unavailable (tkinter import failed): {exc}")
        return 2

    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"Screen blocker unavailable (tk init failed): {exc}")
        return 2

    max_alpha = min(max(float(args.opacity), 0.20), 0.95)
    fade_step = 0.05
    tick_ms = 25
    current_alpha = 0.0
    target_alpha = 0.0

    root.title("NoSlouchBench Alert")
    root.configure(bg="black")
    root.attributes("-topmost", True)
    root.overrideredirect(True)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")
    root.attributes("-alpha", 0.0)
    root.withdraw()

    label = tk.Label(
        root,
        text=f"Sit straight to continue\n\nKill switch: {args.kill_switch}",
        fg="white",
        bg="black",
        font=("Helvetica", 42, "bold"),
        justify="center",
    )
    label.place(relx=0.5, rely=0.5, anchor="center")

    cmd_queue: queue.Queue[str] = queue.Queue()

    def stdin_reader() -> None:
        try:
            for line in sys.stdin:
                cmd = line.strip().upper()
                if cmd:
                    cmd_queue.put(cmd)
        except Exception:
            cmd_queue.put("EXIT")

    threading.Thread(target=stdin_reader, daemon=True).start()

    def swallow_event(_event):
        if current_alpha > 0.02:
            return "break"
        return None

    def on_kill_switch(_event):
        try:
            Path(args.kill_marker).write_text("killed\n", encoding="utf-8")
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        return "break"

    root.bind_all("<Button>", swallow_event)
    root.bind_all("<ButtonRelease>", swallow_event)
    root.bind_all("<Motion>", swallow_event)
    root.bind_all("<MouseWheel>", swallow_event)
    root.bind_all("<Key>", swallow_event)
    root.bind_all(_to_tk_binding(args.kill_switch), on_kill_switch)

    def tick() -> None:
        nonlocal current_alpha, target_alpha

        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "SHOW":
                target_alpha = max_alpha
                root.deiconify()
                root.lift()
            elif cmd == "HIDE":
                target_alpha = 0.0
            elif cmd == "EXIT":
                try:
                    root.destroy()
                except Exception:
                    pass
                return

        if current_alpha < target_alpha:
            current_alpha = min(current_alpha + fade_step, target_alpha)
            try:
                root.attributes("-alpha", current_alpha)
            except Exception:
                pass
        elif current_alpha > target_alpha:
            current_alpha = max(current_alpha - fade_step, target_alpha)
            try:
                root.attributes("-alpha", current_alpha)
            except Exception:
                pass
            if current_alpha <= 0.001:
                try:
                    root.withdraw()
                except Exception:
                    pass

        root.after(tick_ms, tick)

    root.after(tick_ms, tick)

    try:
        root.mainloop()
    except Exception:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(blocker_process_main())
