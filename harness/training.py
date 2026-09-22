"""Training engines for the Studio Training tab.

Two engines, one form:

- ``unsloth-core``  external, Apache-2.0 ``unsloth`` package — we depend on it
  and never vendor it (its Studio UI is AGPL-3.0 and must stay out of this repo).
- ``hive-ternary``  the native toolkit implementing the TBR contract
  (rotation, STE, KD, GGUF pack, eval).

This module is deliberately stdlib-only: the engines and the run-bundle bridge
must be unit-testable without the splinter stack or a GPU.

Run bundles mirror ``POST /v1/protocol/run``: a directory under ``runs_root``
holding ``training.json`` (config), ``run_stdout.log`` (streamed log) and, once
bridged, ``run_report.json`` for the /runs registry.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

ENGINE_UNSLOTH = "unsloth-core"
ENGINE_HIVE_TERNARY = "hive-ternary"

# Module-level so tests can intercept without spawning a process.
_popen = subprocess.Popen


# --------------------------------------------------------------------------- #
# catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Method:
    id: str
    label: str
    flags: tuple[str, ...] = ()
    note: str = ""
    implemented: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "flags": list(self.flags),
            "note": self.note,
            "implemented": self.implemented,
        }


@dataclass(frozen=True)
class Engine:
    id: str
    label: str
    description: str
    license: str
    methods: tuple[Method, ...]
    external: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "license": self.license,
            "external": self.external,
            "note": self.note,
            "methods": [m.to_dict() for m in self.methods],
        }


HIVE_TERNARY_METHODS: tuple[Method, ...] = (
    Method("ste", "Ternary STE (plain)", ("--ste",),
           "ternarize the forward; deployed-model metric, no rotation"),
    Method("ste-rotate", "Rotation + STE", ("--ste", "--rotate"),
           "spec-basis rotation in the training loop (the first real win)"),
    Method("ste-rotate-lsq", "Rotation + STE + LSQ", ("--ste", "--rotate", "--learn-scale"),
           "learnable per-group scales on top of rotation + STE"),
    Method("ste-rotate-kd", "Rotation + STE + KD", ("--ste", "--rotate", "--lam", "0.2"),
           "adds the weight-potential regulariser term"),
    Method("pack", "Pack to GGUF (PQ2_0/TQ2_0)", (), "pack a trained student into a GGUF artifact",
           implemented=False),
    Method("eval", "Evaluate (KLD/PPL + retention)", (), "same-binary KLD/PPL and region retention",
           implemented=False),
)

UNSLOTH_METHODS: tuple[Method, ...] = (
    Method("lora", "LoRA", (), "Apache-2.0 unsloth core via pinned dependency", implemented=False),
    Method("qlora", "QLoRA", (), "4-bit base + LoRA adapters", implemented=False),
    Method("full", "Full fine-tune", (), "all parameters", implemented=False),
    Method("rl", "RL (GRPO/DPO)", (), "reinforcement fine-tuning", implemented=False),
    Method("fp8", "FP8", (), "fp8 training/export", implemented=False),
)

ENGINES: tuple[Engine, ...] = (
    Engine(
        id=ENGINE_UNSLOTH,
        label="Unsloth Core",
        description="External trainer (LoRA/QLoRA/full/RL/FP8), consumed as a pinned "
                    "Apache-2.0 dependency.",
        license="Apache-2.0 (core); AGPL-3.0 Studio UI is NOT used",
        methods=UNSLOTH_METHODS,
        external=True,
        note="Launch path lands in slice 2; catalogue-only for now.",
    ),
    Engine(
        id=ENGINE_HIVE_TERNARY,
        label="Hive-Ternary",
        description="Native ternary toolkit implementing the TBR contract: rotation, "
                    "STE, KD, GGUF pack, evaluation.",
        license="project-owned",
        methods=HIVE_TERNARY_METHODS,
        external=False,
        note="One 512-token window per step; batch size is not a training control.",
    ),
)

DEFAULT_TERNARY_PYTHON = "~/.unsloth/studio/unsloth_studio/bin/python"
LEGACY_RUNNER = "~/Desktop/work/bonsai-ternary-forensics/scripts/pilot/rmd_kd.py"


def engine_catalog() -> list[dict]:
    return [e.to_dict() for e in ENGINES]


def get_engine(engine_id: str) -> Engine:
    for engine in ENGINES:
        if engine.id == engine_id:
            return engine
    raise ValueError(f"unknown engine {engine_id!r}")
    
def get_method(engine: Engine, method_id: str) -> Method:
    for method in engine.methods:
        if method.id == method_id:
            return method
    raise ValueError(f"unknown method {method_id!r} for engine {engine.id!r}")


# --------------------------------------------------------------------------- #
# runner / interpreter resolution (Hive-Ternary)
# --------------------------------------------------------------------------- #
def ternary_python() -> str:
    """Interpreter for Hive-Ternary runs (ROCm torch lives in the Unsloth env)."""
    env = os.environ.get("HIVE_TERNARY_PYTHON", "").strip()
    if env:
        return env
    candidate = Path(DEFAULT_TERNARY_PYTHON).expanduser()
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def ternary_runner() -> list[str]:
    """Argv prefix (python + target) for the Hive-Ternary runner.

    Resolution order: ``$HIVE_TERNARY_RUNNER`` (``.py`` path or module), then an
    in-repo ``hive_ternary/train.py``, then the current forensics harness as a
    documented temporary bridge. Empty when nothing resolves.
    """
    python = ternary_python()
    env = os.environ.get("HIVE_TERNARY_RUNNER", "").strip()
    if env:
        if env.endswith(".py"):
            return [python, str(Path(env).expanduser())]
        return [python, "-m", env]
    local = REPO_ROOT / "hive_ternary" / "train.py"
    if local.is_file():
        return [python, str(local)]
    legacy = Path(LEGACY_RUNNER).expanduser()
    if legacy.is_file():
        return [python, str(legacy)]
    return []


# --------------------------------------------------------------------------- #
# command construction
# --------------------------------------------------------------------------- #
# Fields forwarded for Hive-Ternary when present. The method supplies the
# recipe flags; these are the per-run parameters the form owns.
_TERNARY_INT_FIELDS = {
    "steps": "--steps",
    "seq": "--seq",
    "seed": "--seed",
    "rot_seed": "--rot-seed",
    "eval_regions": "--eval-regions",
    "eval_windows": "--eval-windows",
    "save_every": "--save-every",
    "log_every": "--log-every",
    "project_every": "--project-every",
    "q": "--q",
}
_TERNARY_STR_FIELDS = {
    "model_dir": "--model-dir",
    "corpus": "--corpus",
    "out": "--out",
    "device": "--device",
    "teacher_device": "--teacher-device",
    "update": "--update",
    "pot": "--pot",
}
_TERNARY_FLOAT_FIELDS = {"lr": "--lr", "temp": "--temp", "lam": "--lam"}

_TERNARY_REQUIRED = ("model_dir", "corpus", "out")


def build_command(engine_id: str, method_id: str, fields: dict) -> list[str]:
    """Build the argv for one run. Raises ValueError on bad input.

    Missing optional fields simply drop their flag; missing required fields for
    Hive-Ternary are an error because the runner rejects them anyway.
    """
    engine = get_engine(engine_id)
    method = get_method(engine, method_id)
    if not method.implemented:
        raise ValueError(f"method {engine_id}/{method_id} is not implemented yet")

    if engine_id == ENGINE_UNSLOTH:
        raise ValueError("unsloth-core launch is not implemented yet")

    runner = ternary_runner()
    if not runner:
        raise RuntimeError(
            "no Hive-Ternary runner resolved; set $HIVE_TERNARY_RUNNER to a .py "
            "path or module (e.g. hive_ternary.train)")
    missing = [k for k in _TERNARY_REQUIRED if not fields.get(k)]
    if missing:
        raise ValueError("missing required field(s): " + ", ".join(missing))

    argv = list(runner) + list(method.flags)
    for key, flag in _TERNARY_STR_FIELDS.items():
        value = fields.get(key)
        if value not in (None, ""):
            argv += [flag, str(value)]
    for key, flag in _TERNARY_INT_FIELDS.items():
        value = fields.get(key)
        if value not in (None, ""):
            argv += [flag, str(int(value))]
    for key, flag in _TERNARY_FLOAT_FIELDS.items():
        value = fields.get(key)
        if value not in (None, ""):
            argv += [flag, str(float(value))]
    return argv


# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, limit: int = 40) -> str:
    slug = _SLUG_RE.sub("-", str(text or "")).strip("-")
    return slug[:limit] or "run"


def _redact(argv: list[str]) -> list[str]:
    """Command preview without obviously sensitive values."""
    out = list(argv)
    for i, token in enumerate(out):
        if token in ("--api-key", "--token", "--hf-token") and i + 1 < len(out):
            out[i + 1] = "***"
    return out


def launch(runs_root, engine_id: str, method_id: str, fields: dict,
           name: Optional[str] = None) -> dict:
    """Create a run bundle and spawn one training job; returns its descriptor."""
    if engine_id != ENGINE_HIVE_TERNARY:
        raise ValueError(f"launch is not implemented for engine {engine_id!r}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{stamp}_{_slug(name or method_id)}"
    run_dir = Path(runs_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(fields)
    resolved["out"] = str(run_dir)
    argv = build_command(engine_id, method_id, resolved)

    config = _public_config(resolved, run_dir)
    manifest = {
        "engine": engine_id,
        "method": method_id,
        "argv": _redact(argv),
        "config": config,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cwd": str(REPO_ROOT),
    }
    (run_dir / "training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log_handle = open(run_dir / "run_stdout.log", "ab")  # noqa: SIM115 - child owns it
    proc = _popen(
        argv,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    (run_dir / "pid").write_text(str(proc.pid), encoding="utf-8")
    return {"run_dir": run_name, "pid": proc.pid, "argv": _redact(argv), "method": method_id}


def _public_config(fields: dict, run_dir: Path) -> dict:
    """Config block persisted to training.json (paths made display-friendly)."""
    config = {}
    for key, value in fields.items():
        if value in (None, ""):
            continue
        config[key] = str(value) if key in ("model_dir", "corpus", "out") else value
    config["out"] = str(run_dir)
    return config


# --------------------------------------------------------------------------- #
# progress + status
# --------------------------------------------------------------------------- #
_STEP_RE = re.compile(r"\[rmd\] step (\d+)/(\d+) loss ([\d.]+) (\d+)s")
_RATIO_RE = re.compile(r"deployed_ratio ([\d.]+)")
_INIT_RE = re.compile(r"init step 0: deployed_ratio ([\d.]+)")


def parse_progress(log_text: str) -> dict:
    """Parse the last `[rmd] step …` line and the latest deployed ratio."""
    progress: dict = {}
    init = _INIT_RE.search(log_text)
    if init:
        progress["init_ratio"] = float(init.group(1))
    steps = _STEP_RE.findall(log_text)
    if steps:
        step, total, loss, elapsed = steps[-1]
        progress.update({
            "step": int(step),
            "total_steps": int(total),
            "loss": float(loss),
            "elapsed_s": int(elapsed),
        })
        if int(step) and int(total):
            progress["percent"] = round(100.0 * int(step) / int(total), 1)
    ratios = _RATIO_RE.findall(log_text)
    if ratios:
        progress["deployed_ratio"] = float(ratios[-1])
    return progress


def _read_tail(path: Path, max_bytes: int = 65536) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_status(run_dir) -> dict:
    """Status of one run bundle: state, progress, elapsed, pid."""
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "training.json") or {}
    log_text = _read_tail(run_dir / "run_stdout.log")
    progress = parse_progress(log_text)

    eval_result = _read_json(run_dir / "eval-regions.json")
    pid = _read_int(run_dir / "pid")
    if eval_result is not None:
        state = "finished"
    elif pid and _pid_alive(pid):
        state = "running"
    elif log_text.strip():
        state = "stopped" if progress else "preparing"
    else:
        state = "starting"

    out = {
        "run_dir": run_dir.name,
        "state": state,
        "engine": manifest.get("engine"),
        "method": manifest.get("method"),
        "pid": pid,
        "progress": progress,
        "argv": manifest.get("argv", []),
    }
    if eval_result is not None:
        out["aggregate"] = eval_result.get("aggregate", {})
    # Persist a bridge so /runs can render the run without a separate call.
    write_run_report(run_dir)
    return out


# --------------------------------------------------------------------------- #
# run_report bridge (/runs registry data contract)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def build_run_report(run_dir) -> dict:
    """Bridge a training bundle into the shape `render_report_page` consumes."""
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "training.json") or {}
    eval_result = _read_json(run_dir / "eval-regions.json")
    rmd = _read_json(run_dir / "rmd-report.json")

    aggregate = (eval_result or {}).get("aggregate", {}) if eval_result else {}
    regions = (eval_result or {}).get("regions", []) if eval_result else []
    mean_ratio = aggregate.get("mean_ratio")

    report = {
        "kind": "training",
        "run_id": run_dir.name,
        "engine": manifest.get("engine"),
        "method": manifest.get("method"),
        "created": manifest.get("created"),
        "config": manifest.get("config", {}),
        "argv": manifest.get("argv", []),
        "aggregate": aggregate,
        "regions": regions,
        "progress": parse_progress(_read_tail(run_dir / "run_stdout.log")),
        "status": "finished" if eval_result is not None else "running",
    }
    if isinstance(mean_ratio, (int, float)) and mean_ratio:
        report["retention"] = round(100.0 / float(mean_ratio), 1)
    if isinstance(rmd, dict):
        report["training"] = rmd
    return report


def write_run_report(run_dir) -> Optional[Path]:
    """Persist the bridged run_report.json; returns its path (or None)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    report = build_run_report(run_dir)
    path = run_dir / "run_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
