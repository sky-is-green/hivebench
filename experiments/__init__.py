"""HiveBench experiments: data generation, probes, A/B batteries, finetune runs."""

# --- splinter system path bootstrap -------------------------------------------
# Non-pytest entry points (console scripts, `python -m ...`) need the sibling
# splinter-memory checkout on sys.path: flat names (cortex, retention, backend,
# ...) resolve from <SPLINTER>/splinter and `import splinter` from <SPLINTER> itself.
# pytest runs get the same paths from conftest.py; this is idempotent there.
# Override the checkout location with $SPLINTER_HOME.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]


def _resolve_system() -> _Path:
    env = _os.environ.get("SPLINTER_HOME") or _os.environ.get("STRATA_HOME")
    if env:
        return _Path(env)
    for _name in ("splinter-memory", "strata-memory"):
        _cand = _REPO_ROOT.parent / _name
        if _cand.is_dir():
            return _cand
    return _REPO_ROOT.parent / "splinter-memory"


_SPLINTER = _resolve_system()
for _p in (_SPLINTER, _SPLINTER / "splinter"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
