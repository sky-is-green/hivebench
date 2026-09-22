"""HiveBench Studio harness — the FastAPI sidecar over the splinter research stack.

Exposes the Splinter context-curation pipeline and the HiveBench measurement layer
as a local HTTP service (default 127.0.0.1:8765) for the dsh shell (M2) and the
web UI. See HARNESS-SPEC.md §3.3 for the endpoint contract.

Run::

    python -m harness                 # live (LM Studio / provider backends)
    python -m harness --mock          # offline (fake drone + mock backend)
"""

# --- splinter system path bootstrap -------------------------------------------
# Non-pytest entry points (console scripts, `python -m ...`) need the sibling
# splinter-memory checkout on sys.path: flat names (cortex, retention, backend,
# ...) resolve from <SPLINTER>/splinter and `import splinter` from <SPLINTER> itself.
# pytest runs get the same paths from conftest.py; this is idempotent there.
# Override the checkout location with $SPLINTER_HOME.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]


def _resolve_system() -> _Path:
    env = _os.environ.get("SPLINTER_HOME") or _os.environ.get("STRATA_HOME")
    if env:
        return _Path(env)
    for _name in ("splinter-memory", "strata-memory"):
        _cand = _REPO_ROOT / _name
        if _cand.is_dir():
            return _cand
    return _REPO_ROOT / "splinter-memory"


_SPLINTER = _resolve_system()

for _p in (_SPLINTER, _SPLINTER / "splinter"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from harness.app import create_app

__all__ = ["create_app"]

