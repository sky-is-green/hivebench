"""Make the suite (`experiments/`, `testing/`, `tests/`), sidecar (`harness/`)
and vendored dsh SDK (`vendor/deepseek_harness`) importable, plus the system
under test from the sibling splinter-memory checkout, so flat names like
`cortex`, `sieve`, `experiments`, `harness` resolve regardless of how pytest
is invoked (editable install makes this unnecessary, but from-source runs
stay supported)."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# experiments/, testing/, tests/ and the sidecar package harness/ live at the
# repo root (the old double-nested harness/harness/ was flattened 2026-09-12).
sys.path.insert(0, ROOT)
# The vendored dsh Python SDK (deepseek_harness) — the agent bridge imports it.
sys.path.insert(0, os.path.join(ROOT, "vendor"))
# The system under test — sibling checkout. Resolve $SPLINTER_HOME (the legacy
# alias $STRATA_HOME is still accepted), else the first of the two sibling
# names that exists, so both a fresh `splinter-memory` clone and a pre-rename
# `strata-memory` checkout resolve. Override with $SPLINTER_HOME.
SPLINTER = os.environ.get("SPLINTER_HOME") or os.environ.get("STRATA_HOME")
if not SPLINTER:
    for _name in ("splinter-memory", "strata-memory"):
        _cand = os.path.normpath(os.path.join(ROOT, "..", _name))
        if os.path.isdir(_cand):
            SPLINTER = _cand
            break
if not SPLINTER:
    SPLINTER = os.path.normpath(os.path.join(ROOT, "..", "splinter-memory"))
# The checkout root (so `import splinter` works) and the package root (so flat
# names like `cortex`, `sieve` work).
sys.path.insert(0, SPLINTER)
sys.path.insert(0, os.path.join(SPLINTER, "splinter"))
