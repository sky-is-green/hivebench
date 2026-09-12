"""Guards for the repo layout.

This repo (hivebench) holds the evaluation suite (`experiments/`, `testing/`,
`tests/`) and the HiveBench Studio sidecar (`harness/`) with flat top-level
import names; the system under test lives in the sibling checkout
(`../strata-memory`, overridable via $STRATA_HOME). These tests pin the
invariants the split depends on: no stray package dirs at the root, pyproject
declaring all trees, the vocab data present in the sibling `strata` tree, and
the test runner's path constants resolving.
"""

import os
import tomllib
from pathlib import Path

from setuptools import find_packages

ROOT = Path(__file__).resolve().parents[2]
STRATA_HOME = Path(os.environ.get("STRATA_HOME", str(ROOT.parent / "strata-memory")))
HIVE = STRATA_HOME / "strata"

PACKAGE_ROOTS = ("experiments", "testing", "tests", "harness")

EXPECTED_PACKAGES = {
    "experiments",
    "harness",
    "testing",
    "tests",
}


# Standalone root-level entrypoint/utility scripts that are not importable
# package modules (conftest + developer tools). The guard still fails on any
# *new* root module, so package roots cannot silently leak into the top level.
KNOWN_ROOT_SCRIPTS = {
    "conftest.py",
    "install_cpu_torch.py",
    "launch_detached.py",
    "longrun.py",
    "longtest.py",
}


def test_root_python_files_are_only_known_scripts():
    assert {p.name for p in ROOT.glob("*.py")} <= KNOWN_ROOT_SCRIPTS


# Directories that hold plain scripts (not importable packages) — allowed at
# root even though they contain .py files.
NONPKG_SCRIPT_DIRS = {"scripts"}

def test_no_stray_package_dirs_at_root():
    stray = []
    for entry in sorted(ROOT.iterdir()):
        if entry.is_dir() and entry.name not in (*PACKAGE_ROOTS, *NONPKG_SCRIPT_DIRS):
            if list(entry.glob("*.py")):
                stray.append(entry.name)
    assert stray == []


def test_pyproject_declares_all_trees():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    where = cfg["tool"]["setuptools"]["packages"]["find"]["where"]
    assert where == ["."]


def test_find_packages_resolves_flat_names():
    found = set(find_packages(where=str(ROOT)))
    assert EXPECTED_PACKAGES <= found
    # flat names, not nested: nothing may reintroduce a hivebench.* shadow
    assert not any(p.startswith("hivebench") for p in found)


def test_vocab_data_present_and_loadable():
    from sieve.vocabulary import Vocabulary

    vocab = HIVE / "vocab"
    assert (vocab / "code.json").is_file()
    assert (vocab / "general.json").is_file()
    vocab_obj = Vocabulary.load("code", "general")
    assert vocab_obj.size > 0


def test_runner_path_constants_resolve():
    import tests.run_hive_tests as runner

    assert runner.ROOT == ROOT
    assert runner.HIVEBENCH == ROOT
    assert runner.TESTS == ROOT / "tests"
    assert runner.HIVE == HIVE
    for rel in runner.INTELLIGENCE:
        assert (ROOT / rel).is_file(), f"intelligence file missing: {rel}"
