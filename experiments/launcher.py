"""HiveBench launcher — a small Tkinter app to configure and run the benchmark
without memorizing CLI flags.

Builds the exact ``experiments.generate_data`` command from form controls, shows
it, and runs it in a subprocess with live output streamed into the window (so the
``--term`` dashboard is not needed here — the window *is* the display). The
command-building logic is a pure function (``build_argv``) so it is testable and
so the GUI never diverges from what the CLI can express.

Usage::

    python -m experiments.launcher
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

MODELS_BASE = "http://localhost:1234"
MODELS_URL = MODELS_BASE + "/v1/models"

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def run_python() -> str:
    """The interpreter to run the benchmark with: the project venv python when it
    exists (so deps resolve regardless of how the launcher was started), else the
    interpreter that launched this app."""
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


MODES = ("live", "mock")
CONFIDENCE_MODES = ("mcdropout", "single", "off")


class ToolTip:
    """Simple hover tooltip for a widget (bound to the widget and its children)."""

    def __init__(self, widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._tip = None
        targets = [widget]
        if hasattr(widget, "winfo_children"):
            targets += list(widget.winfo_children())
        for w in targets:
            w.bind("<Enter>", self._show, add="+")
            w.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self._tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + 22
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self.text, justify="left",
                 background="#ffffe0", foreground="#222",
                 relief="solid", borderwidth=1, font=("Segoe UI", 9),
                 wraplength=400, padx=6, pady=4).pack()

    def _hide(self, _event=None) -> None:
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def build_argv(v: dict) -> list[str]:
    """Map launcher field values to a ``generate_data`` argv list.

    ``v`` keys: mode, model, conversations, max_convs, max_turns, max_tokens,
    confidence, checkpoint_every, protocol, baselines, resume. Empty strings are
    skipped so defaults apply.
    """
    argv = ["--live" if v["mode"] == "live" else "--mock"]
    if v.get("model", "").strip():
        argv += ["--model", v["model"].strip()]
    if v.get("no_thinking"):
        argv.append("--no-thinking")
    if v.get("conversations", "").strip():
        argv += ["--conversations", v["conversations"].strip()]
    if v.get("max_convs", "").strip():
        argv += ["--max-convs", v["max_convs"].strip()]
    if v.get("max_turns", "").strip():
        argv += ["--max-turns", v["max_turns"].strip()]
    if v.get("max_tokens", "").strip():
        argv += ["--max-tokens", v["max_tokens"].strip()]
    if v.get("confidence", "").strip():
        argv += ["--confidence", v["confidence"].strip()]
    if v.get("checkpoint_every", "").strip():
        argv += ["--checkpoint-every", v["checkpoint_every"].strip()]
    if v.get("protocol"):
        argv.append("--protocol")
    if v.get("baselines"):
        argv.append("--baselines")
    if v.get("resume", "").strip():
        argv += ["--resume", v["resume"].strip()]
    return argv


class LauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HiveBench Launcher")
        self.root.geometry("760x640")
        self.proc: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        # Closing the window (X) must not leave the benchmark running in the
        # background: kill the child process before the app exits.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.vars = {
            "mode": tk.StringVar(value="live"),
            "model": tk.StringVar(value=""),
            "conversations": tk.StringVar(value="tests/fixtures/generated"),
            "max_convs": tk.StringVar(value="3"),
            "max_turns": tk.StringVar(value="10"),
            "max_tokens": tk.StringVar(value=""),
            "confidence": tk.StringVar(value="off"),
            "checkpoint_every": tk.StringVar(value="5"),
            "protocol": tk.BooleanVar(value=False),
            "baselines": tk.BooleanVar(value=False),
            "no_thinking": tk.BooleanVar(value=True),
            "resume": tk.StringVar(value=""),
        }
        self._build_form()
        self._build_output()
        self._load_models()
        self._detect_loaded_model()
        self._suggest_resume()
        for var in self.vars.values():
            var.trace_add("write", lambda *_: self._refresh_command())

    # ------------------------------------------------------------------
    def _row(self, parent, label, widget):
        ttk.Label(parent, text=label, anchor="w").grid(row=parent.grid_size()[1],
                                                       column=0, sticky="w", padx=6, pady=3)
        widget.grid(row=parent.grid_size()[1] - 1, column=1, sticky="ew", padx=6, pady=3)
        return widget

    def _build_form(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Run configuration")
        frame.pack(fill="x", padx=8, pady=(8, 4))
        frame.columnconfigure(1, weight=1)

        def tip(widget, text):
            ToolTip(widget, text)

        mode_row = ttk.Frame(frame)
        for m in MODES:
            ttk.Radiobutton(mode_row, text=m, value=m, variable=self.vars["mode"]).pack(side="left", padx=4)
        tip(self._row(frame, "Mode", mode_row),
            "live = real LM Studio generation (real evidence). "
            "mock = offline fake drones (validates the harness only, no real science).")

        self.model_combo = ttk.Combobox(frame, textvariable=self.vars["model"])
        tip(self._row(frame, "Model (live)", self.model_combo),
            "Model id loaded in LM Studio to generate with (auto-detected at "
            "startup). On reasoning models, chain-of-thought consumes the output "
            "budget before the visible answer, so a small reply cap yields empty "
            "replies. Disable thinking via LM Studio's 'thinking' toggle, or tick "
            "'Disable thinking' below.")

        tip(self._row(frame, "Conversations dir", ttk.Entry(frame, textvariable=self.vars["conversations"])),
            "Directory of conversation JSON files to run. Defaults to the synthetic "
            "corpus (tests/fixtures/generated).")

        tip(self._row(frame, "Max conversations", ttk.Entry(frame, textvariable=self.vars["max_convs"], width=8)),
            "How many conversations to process. Lower = faster iteration; raise it "
            "for the final evidence run.")

        tip(self._row(frame, "Max turns / conv", ttk.Entry(frame, textvariable=self.vars["max_turns"], width=8)),
            "Cap on user turns per conversation. Lower = faster iteration.")

        tip(self._row(frame, "Max tokens (blank = uncapped)", ttk.Entry(frame, textvariable=self.vars["max_tokens"], width=10)),
            "Cap on reply length. On reasoning models a small cap yields empty "
            "replies unless thinking is disabled (see 'Disable thinking'); with "
            "thinking off, a cap speeds up iteration.")

        conf_combo = ttk.Combobox(frame, textvariable=self.vars["confidence"], values=CONFIDENCE_MODES, state="readonly")
        tip(self._row(frame, "Confidence mode", conf_combo),
            "Drone confidence scoring. 'off' (default) skips the MC-dropout "
            "passes - the stock embedding model disables dropout at inference, so "
            "variance is always zero and confidence is always 1.0; those passes "
            "would only add time. 'single' is 1 pass (same 1.0 result as off). "
            "'mcdropout' (3 passes) is meaningful only with a dropout-active "
            "encoder (custom drone / P6 escalation).")

        no_thinking = ttk.Checkbutton(frame, text="Disable thinking (--no-thinking)",
                                      variable=self.vars["no_thinking"])
        tip(self._row(frame, "Generation", no_thinking),
            "Send enable_thinking=false with every request. On reasoning models "
            "this skips chain-of-thought, which speeds up turns and makes reply "
            "caps yield real (non-empty) output. Pair with LM Studio's 'thinking' "
            "toggle for models that only honor the GUI setting.")

        tip(self._row(frame, "Checkpoint every N", ttk.Entry(frame, textvariable=self.vars["checkpoint_every"], width=8)),
            "Write a resume checkpoint every N turns. If the run is interrupted, "
            "--resume continues from the last checkpoint instead of restarting at zero.")

        proto = ttk.Checkbutton(frame, text="Run P1-P10 protocol", variable=self.vars["protocol"])
        tip(self._row(frame, "Tests", proto),
            "Also run the white paper's P1-P10 predictions against live data. "
            "Adds significant time (full generations over the longest conversation).")
        base = ttk.Checkbutton(frame, text="Run baselines (LM Studio + FIFO)", variable=self.vars["baselines"])
        tip(self._row(frame, "", base),
            "Also run the no-splinter baselines for comparison (rolling + FIFO). "
            "Roughly doubles run time; required for the final evidence run.")

        tip(self._row(frame, "Resume run dir (optional)", ttk.Entry(frame, textvariable=self.vars["resume"])),
            "A previous run directory to continue from its checkpoint.json. "
            "Auto-suggested from the most recent checkpointed run.")

        ttk.Label(frame, text="Note: a small --max-tokens cap on a reasoning model yields "
                              "empty replies unless thinking is disabled (LM Studio "
                              "'thinking' toggle or --no-thinking).", foreground="#666").grid(
            row=frame.grid_size()[1], column=0, columnspan=2, sticky="w", padx=6, pady=(4, 4))

        ttk.Label(frame, text="Command:").grid(row=frame.grid_size()[1], column=0,
                                               sticky="nw", padx=6, pady=(6, 2))
        self.cmd_text = tk.Text(frame, height=3, font=("Consolas", 9), wrap="word",
                                state="disabled", bg="#f4f4f4")
        self.cmd_text.grid(row=frame.grid_size()[1], column=1, sticky="ew", padx=6, pady=(6, 6))
        self._refresh_command()

    def _refresh_command(self) -> None:
        v = {k: x.get() for k, x in self.vars.items()}
        argv = build_argv(v)
        cmd = " ".join([run_python(), "-m", "experiments.generate_data", *argv])
        self.cmd_text.config(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", cmd)
        self.cmd_text.config(state="disabled")

    def _build_output(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=4)
        self.run_btn = ttk.Button(bar, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.status = ttk.Label(bar, text="ready", anchor="e")
        self.status.pack(side="right", fill="x", expand=True)

        out = ttk.LabelFrame(self.root, text="Output")
        out.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.output = tk.Text(out, font=("Consolas", 9), wrap="none", state="disabled")
        scroll = ttk.Scrollbar(out, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)

    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        try:
            import requests

            r = requests.get(MODELS_URL, timeout=3)
            ids = [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
            if ids:
                self.model_combo["values"] = ids
        except Exception:  # noqa: BLE001 - LM Studio not running
            pass

    def _detect_loaded_model(self) -> None:
        """Probe the default endpoint with a 1-token request to learn which model
        LM Studio currently has loaded, and pre-fill the Model field with it."""
        if self.vars["model"].get().strip():
            return
        try:
            import requests

            r = requests.post(
                MODELS_BASE + "/v1/chat/completions",
                json={"model": "",
                      "messages": [{"role": "system", "content": "You are a connectivity probe. Reply with 'ok'."},
                                   {"role": "user", "content": "ping"}],
                      "max_tokens": 1},
                timeout=30,
            )
            loaded = r.json().get("model") or ""
            if loaded and not self.vars["model"].get().strip():
                self.vars["model"].set(loaded)
        except Exception:  # noqa: BLE001 - LM Studio off / unreachable
            pass

    def _suggest_resume(self) -> None:
        """Pre-fill Resume with the most recent run dir that has a checkpoint."""
        if self.vars["resume"].get().strip():
            return
        try:
            ckpts = sorted(Path(__file__).resolve().parents[1].glob("runs/*/checkpoint.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                self.vars["resume"].set(str(ckpts[0].parent))
        except Exception:  # pragma: no cover
            pass

    def _append(self, text: str) -> None:
        self.output.config(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.config(state="disabled")

    def _run(self) -> None:
        if self.proc is not None:
            return
        argv = build_argv({k: x.get() for k, x in self.vars.items()})
        cmd = [run_python(), "-m", "experiments.generate_data", *argv]
        self.output.delete("1.0", "end")
        self._append("$ " + " ".join(cmd) + "\n\n")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="running…")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            creationflags=flags,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        self.root.after(100, self._drain)

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self._queue.put(line)
        code = self.proc.wait()
        self._queue.put(f"\n[exit code {code}]\n")

    def _drain(self) -> None:
        try:
            while True:
                self._append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if self.proc is None:
            return  # stopped / closed — stop the poll loop
        if self.proc.poll() is not None:
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status.config(text="done")
            self.proc = None
        else:
            self.root.after(100, self._drain)

    def _kill_proc(self) -> None:
        """Forcefully terminate the benchmark subprocess and its whole tree.

        ``terminate()`` is ``TerminateProcess`` on Windows, which kills the direct
        child; ``/T`` additionally kills any descendants so no generation keeps
        running after the launcher is stopped or closed.
        """
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:  # pragma: no cover - already gone
            pass
        if sys.platform == "win32" and proc.pid:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                )
            except Exception:  # pragma: no cover
                pass

    def _stop(self) -> None:
        if self.proc is not None:
            self._kill_proc()
            self.status.config(text="stopped")
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def _on_close(self) -> None:
        self._kill_proc()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())