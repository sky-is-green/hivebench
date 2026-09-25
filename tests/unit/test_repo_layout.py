"""Guards for the repo layout.

This repo (hivebench) holds the evaluation suite (`experiments/`, `testing/`,
`tests/`) and the HiveBench Studio sidecar (`harness/`) with flat top-level
import names; the system under test lives in the sibling checkout. That
checkout is resolved (env -> walk-up from the worktree -> primary checkout)
by `conftest.resolve_splinter_home`, which also fails loudly rather than
returning a non-existent path. These tests pin the invariants the split
depends on: no stray package dirs at the root, pyproject declaring all trees,
the vocab data present in the sibling `splinter` tree, the test runner's path
constants resolving, and the sibling resolution order itself.
"""

import tomllib
from pathlib import Path

import pytest
from setuptools import find_packages

import conftest

ROOT = Path(__file__).resolve().parents[2]
# Resolve the sibling through the same order conftest uses (env -> walk-up ->
# primary checkout) so the guard and the runner cannot disagree.
SPLINTER_HOME = Path(conftest.resolve_splinter_home(str(ROOT)))
HIVE = SPLINTER_HOME / "splinter"

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
            # Nested git checkouts (e.g. CI's splinter-sys sibling) are
            # infrastructure, not package layout — exempt from the guard.
            if (entry / ".git").exists():
                continue
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


# --- sibling-checkout resolution (worktree layout) -------------------------
#
# HIVE-OPS puts worktrees at <project>/worktrees/<repo>/hivebench-<Task_ID>,
# where ROOT/../splinter-memory does not exist. These pin the resolution order
# SPLINTER_HOME -> walk-up -> primary checkout, and the loud failure when none
# resolve (never a silent run against the wrong tree).


def test_resolve_splinter_prefers_env_override(tmp_path):
    override = tmp_path / "custom-splinter"
    override.mkdir()
    root = tmp_path / "worktrees" / "hivebench" / "hivebench-T46"
    root.mkdir(parents=True)

    resolved = conftest.resolve_splinter_home(
        str(root), env={"SPLINTER_HOME": str(override)}
    )

    assert Path(resolved) == override


def test_resolve_splinter_walks_up_from_worktree(tmp_path):
    root = tmp_path / "worktrees" / "hivebench" / "hivebench-T46"
    root.mkdir(parents=True)
    sibling = tmp_path / "splinter-memory"
    sibling.mkdir()

    resolved = conftest.resolve_splinter_home(str(root), env={})

    assert Path(resolved) == sibling


def test_resolve_splinter_walks_up_to_legacy_strata(tmp_path):
    root = tmp_path / "worktrees" / "hivebench" / "hivebench-T46"
    root.mkdir(parents=True)
    legacy = tmp_path / "strata-memory"
    legacy.mkdir()

    resolved = conftest.resolve_splinter_home(str(root), env={})

    assert Path(resolved) == legacy


def test_resolve_splinter_falls_back_to_primary_checkout(tmp_path):
    # A worktree whose primary checkout lives outside its ancestor chain.
    primary = tmp_path / "primary" / "hivebench"
    (primary / ".git" / "worktrees" / "wt").mkdir(parents=True)
    sibling = tmp_path / "primary" / "splinter-memory"
    sibling.mkdir()

    root = tmp_path / "elsewhere" / "hivebench-T46"
    root.mkdir(parents=True)
    (root / ".git").write_text(
        f"gitdir: {primary / '.git' / 'worktrees' / 'wt'}\n", encoding="utf-8"
    )

    resolved = conftest.resolve_splinter_home(str(root), env={})

    assert Path(resolved) == sibling


def test_resolve_splinter_rejects_missing_env_override(tmp_path):
    missing = tmp_path / "does-not-exist"

    with pytest.raises(conftest.SplinterHomeNotFound) as excinfo:
        conftest.resolve_splinter_home(
            str(tmp_path), env={"SPLINTER_HOME": str(missing)}
        )

    assert str(missing) in str(excinfo.value)


def test_resolve_splinter_fails_loudly_when_nothing_resolves(tmp_path, monkeypatch):
    root = tmp_path / "worktrees" / "hivebench" / "hivebench-T46"
    root.mkdir(parents=True)
    # Force every filesystem probe to miss, including the primary-checkout path.
    monkeypatch.setattr(conftest, "_is_dir", lambda path: False)

    with pytest.raises(conftest.SplinterHomeNotFound) as excinfo:
        conftest.resolve_splinter_home(str(root), env={})

    message = str(excinfo.value)
    # Loud failure prints the paths it resolved to, rather than continuing.
    assert str(root) in message
    assert "splinter-memory" in message
    assert "SPLINTER_HOME" in message
