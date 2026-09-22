"""Import sweep: every module in both trees (splinter/ + hivebench/) must import.

Guards the flat-import contract (`cortex`, `sieve`, `tests`, `experiments`, ...)
and catches modules the rest of the suite never imports (e.g.
`experiments.encoder_probe`, `experiments.contrastive_finetune`) or any
path/rename fallout in the two-tree layout.
"""

import importlib
import pkgutil

import backend
import cortex
import experiments
import focal
import logs
import membrane
import auditor
import retention
import sieve
import testing
import tests

_PACKAGES = [
    cortex,
    sieve,
    membrane,
    retention,
    focal,
    backend,
    auditor,
    logs,
    testing,
    experiments,
    tests,
]


def test_every_module_imports_cleanly():
    failures = []
    for pkg in _PACKAGES:
        for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                importlib.import_module(mod.name)
            except Exception as exc:  # noqa: BLE001 - report any import failure
                failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert failures == [], "modules failed to import:\n" + "\n".join(failures)