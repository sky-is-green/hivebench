"""Make the suite (`experiments/`, `testing/`, `tests/`), sidecar (`harness/`)
and vendored dsh SDK (`vendor/deepseek_harness`) importable, plus the system
under test from the sibling strata-memory checkout, so flat names like
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
# The system under test — sibling checkout (override with $STRATA_HOME).
STRATA = os.environ.get("STRATA_HOME", os.path.normpath(os.path.join(ROOT, "..", "strata-memory")))
# The checkout root (so `import strata` works) and the package root (so flat
# names like `cortex`, `sieve` work).
sys.path.insert(0, STRATA)
sys.path.insert(0, os.path.join(STRATA, "strata"))
