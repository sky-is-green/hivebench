"""Make the suite (`experiments/`, `testing/`, `tests/`), sidecar (`harness/`)
and vendored dsh SDK (`vendor/deepseek_harness`) importable, plus the system
under test from the sibling strata-memory checkout, so flat names like
`cortex`, `sieve`, `experiments`, `harness` resolve regardless of how pytest
is invoked (editable install makes this unnecessary, but from-source runs
stay supported)."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# experiments/, testing/, tests/ live at the repo root.
sys.path.insert(0, ROOT)
# The sidecar package is double-nested (harness/harness/) — put the PARENT of
# the real package on the path so flat `import harness` resolves to it, not to
# the empty outer namespace dir (regression from the hivebench move: the old
# strata-memory conftest inserted this root and the moved one dropped it).
sys.path.insert(0, os.path.join(ROOT, "harness"))
# The vendored dsh Python SDK (deepseek_harness) — the agent bridge imports it.
sys.path.insert(0, os.path.join(ROOT, "vendor"))
# The system under test — sibling checkout (override with $STRATA_HOME).
STRATA = os.environ.get("STRATA_HOME", os.path.normpath(os.path.join(ROOT, "..", "strata-memory")))
# The checkout root (so `import strata` works) and the package root (so flat
# names like `cortex`, `sieve` work).
sys.path.insert(0, STRATA)
sys.path.insert(0, os.path.join(STRATA, "strata"))
