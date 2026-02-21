from __future__ import annotations

import threading
import time

import numpy as np


class SlouchBeeper:
    """Continuous beep while slouching; silence when upright."""

    def __init__(self, frequency_hz: float = 880.0, sample_rate: int = 44100) -> None:
        self.frequency_hz = float(frequency_hz)
        self.sample_rate = int(sample_rate)
        self._enabled = False
        self._closed = False
        self._lock = threading.Lock()

        self._backend = "none"
        self._stream = None
        self._phase = 0
        self._fallback_thread = None
        self._fallback_stop = threading.Event()

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import sounddevice as sd
        except Exception:
            self._backend = "terminal_bell"
            self._start_terminal_bell_thread()
            return

        def callback(outdata, frames, _time, _status):
            with self._lock:
                enabled = self._enabled and not self._closed
            if not enabled:
                outdata[:] = 0
                return

            t = (np.arange(frames, dtype=np.float32) + self._phase) / self.sample_rate
            wave = 0.2 * np.sin(2 * np.pi * self.frequency_hz * t, dtype=np.float32)
            self._phase += frames
            outdata[:, 0] = wave
            if outdata.shape[1] > 1:
                outdata[:, 1:] = wave[:, None]

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=0,
        )
        self._stream.start()
        self._backend = "sounddevice"

    def _start_terminal_bell_thread(self) -> None:
        def loop():
            while not self._fallback_stop.is_set():
                with self._lock:
                    enabled = self._enabled and not self._closed
                if enabled:
                    print("\a", end="", flush=True)
                    time.sleep(0.35)
                else:
                    time.sleep(0.05)

        self._fallback_thread = threading.Thread(target=loop, daemon=True)
        self._fallback_thread.start()

    def start(self) -> None:
        with self._lock:
            self._enabled = True

    def stop(self) -> None:
        with self._lock:
            self._enabled = False

    def close(self) -> None:
        with self._lock:
            self._enabled = False
            self._closed = True

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._fallback_thread is not None:
            self._fallback_stop.set()
            self._fallback_thread.join(timeout=1.0)

