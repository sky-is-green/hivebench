"""HiveBench ablation / AB test tooling."""

# --- strata system path bootstrap -------------------------------------------
# Non-pytest entry points (console scripts, `python -m ...`) need the sibling
# strata-memory checkout on sys.path: flat names (cortex, retention, backend,
# ...) resolve from <STRATA>/strata and `import strata` from <STRATA> itself.
# pytest runs get the same paths from conftest.py; this is idempotent there.
# Override the checkout location with $STRATA_HOME.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_STRATA = _Path(_os.environ.get("STRATA_HOME", _REPO_ROOT.parent / "strata-memory"))
for _p in (_STRATA, _STRATA / "strata"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
