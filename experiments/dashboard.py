"""HiveBench terminal dashboard + keep-awake helper.

``TermDashboard`` renders the benchmark's live progression, predicted time to
completion, rolling stats, and a real-time turn feed inside the terminal using
ANSI cursor/line escapes (no external dependencies). It runs on the hot path's
own thread (update calls are cheap) and is a no-op when stdout is not a TTY.

``KeepAwake`` prevents the OS from sleeping while a long live run is active via
Windows ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)``. The
display state is intentionally left alone (nothing user-facing on screen during
a benchmark). No-op on non-Windows.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Optional


class KeepAwake:
    """Prevent OS sleep for the lifetime of the object (Windows only)."""

    _ES_CONTINUOUS = 0x80000000
    _ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self.active = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fn = None
        try:
            import ctypes

            self._fn = ctypes.windll.kernel32.SetThreadExecutionState
            self._assert()
            self._thread = threading.Thread(
                target=self._run, name="splinter-keep-awake", daemon=True
            )
            self._thread.start()
            self.active = True
        except Exception:  # pragma: no cover - platform dependent
            self._fn = None

    def _assert(self) -> None:
        if self._fn is not None:
            self._fn(self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED)

    def _run(self) -> None:
        while not self._stop.wait(30):
            self._assert()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._fn is not None:
            self._fn(self._ES_CONTINUOUS)  # release the system-required flag


class TermDashboard:
    """Terminal (ANSI) live dashboard.

    Renders phase, progress, ETA, rolling stats, and a recent-turn feed into a
    fixed region of the terminal using ANSI cursor/line escapes (no external
    dependencies). Only active when the flag is requested *and* stdout is a TTY;
    otherwise every method is a no-op.
    """

    def __init__(self, enabled: bool = True, refresh_interval: float = 1.0) -> None:
        self.enabled = False
        self._height = 0
        self._last = 0.0
        self._refresh_interval = refresh_interval
        self._totals: deque = deque(maxlen=1000)
        self._gens: deque = deque(maxlen=1000)
        self._pes: deque = deque(maxlen=1000)
        self.phase = "starting"
        self.done = 0
        self.total = 0
        self.elapsed = 0.0
        self.eta = 0.0
        self.conv = 0
        self.conv_total = 0
        self.turn = 0
        self.cap = 0
        self.lines: list = []
        self.stats = "awaiting first turn"
        if not enabled or not sys.stdout.isatty():
            return
        self._enable_vt()
        self.enabled = True

    # ------------------------------------------------------------------
    # Main-thread API (cheap no-ops when disabled)
    # ------------------------------------------------------------------
    def set_phase(self, text: str) -> None:
        self.phase = text
        self._draw()

    def update_progress(self, done, total, elapsed_s, eta_s, conv, conv_total, turn, cap) -> None:
        self.done, self.total = done, total
        self.elapsed, self.eta = elapsed_s, eta_s
        self.conv, self.conv_total = conv, conv_total
        self.turn, self.cap = turn, cap
        self._draw()

    def add_turn(self, query, reply, pes, gen_ms, total_ms) -> None:
        self._totals.append(total_ms)
        self._gens.append(gen_ms)
        self._pes.append(pes)
        import statistics as st

        q = (query or "").replace("\n", " ").strip()[:90]
        r = (reply or "").replace("\n", " ").strip()[:60]
        self.lines.append(f"PES {pes:5.1f}  gen {gen_ms:6.0f}ms  total {total_ms:6.0f}ms  | {q} -> {r}")
        self.lines = self.lines[-200:]
        n = len(self._pes)
        self.stats = (
            f"turns {n}  avg total {st.mean(self._totals):.0f}ms  "
            f"avg gen {st.mean(self._gens):.0f}ms  p50 {st.median(self._totals):.0f}ms  "
            f"PES min/mean {min(self._pes):.1f}/{st.mean(self._pes):.1f}"
        )
        self._draw()

    def add_line(self, text: str) -> None:
        self.lines.append(text)
        self.lines = self.lines[-200:]
        self._draw()

    def close(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\n")  # leave the shell prompt on a clean line
        sys.stdout.flush()
        self.enabled = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _enable_vt() -> None:
        """Best-effort enable of ANSI VT processing on the Windows console."""
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            handle = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(handle, ctypes.byref(mode)):
                k32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VT_PROCESSING
        except Exception:  # pragma: no cover
            pass

    def _render(self) -> list:
        pct = (self.done / self.total * 100.0) if self.total > 0 else 0.0
        width = 24
        filled = int(pct / 100.0 * width)
        bar = "#" * filled + "." * (width - filled)
        head = f"HiveBench  {self.phase:<24} elapsed {self._fmt(self.elapsed)}  ETA {self._fmt(self.eta)}"
        prog = f"conv {self.conv}/{self.conv_total}  turn {self.turn}/{self.cap}  done {self.done}/{self.total}  [{bar}] {pct:5.1f}%"
        block = [head, prog, self.stats]
        recent = self.lines[-8:]
        if recent:
            block.append("--- recent turns ---")
            block.extend(("  " + line)[:110] for line in recent)
        return block

    def _draw(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last < self._refresh_interval:
            return
        self._last = now
        block = self._render()
        out = sys.stdout
        try:
            if self._height:
                out.write(f"\x1b[{self._height}A")
            out.write("\r\x1b[2K" + "\r\n\x1b[2K".join(block) + "\x1b[J")
            self._height = len(block)
            out.flush()
        except Exception:  # pragma: no cover - terminal closed
            self.enabled = False

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"