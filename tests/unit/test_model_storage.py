"""Unit tests for ``scripts/check_model_storage.py`` — where the models live.

The harness's models are read in ways that make the *device* matter far more
than the size: the Qwen3.8-Flash-Next PLE table is 320,001,536 rows of 90
bytes read randomly, one row per token per layer, so every row is a seek.
On a FUSE mount of a rotational disk that is 7.5 ms; on the NVMe it is
0.19 ms.  Measured on 2026-10-21, moving the model's shards to the NVMe took
the decode rate from 3.39 tok/s to 20.52 tok/s.

So: the resolution rule is the harness's own, the device classification is
right, and a model that is present locally is not left on a slow device.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_model_storage", REPO / "scripts" / "check_model_storage.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


csm = _load()


# ---------------------------------------------------------------------------
# the resolution rule
# ---------------------------------------------------------------------------
def test_resolve_prefers_an_exact_file(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"x")
    assert csm._resolve("model.gguf", tmp_path) == target.resolve()


def test_resolve_finds_a_file_by_stem(tmp_path):
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "Qwen3.8-Flash-Next.gguf"
    target.write_bytes(b"x")
    assert csm._resolve("Qwen3.8-Flash-Next", tmp_path) == target.resolve()


def test_resolve_returns_none_when_absent(tmp_path):
    assert csm._resolve("not-here", tmp_path) is None


def test_resolve_prefers_the_library_over_a_stem_match(tmp_path):
    exact = tmp_path / "thing.gguf"
    exact.write_bytes(b"x")
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "thing.gguf").write_bytes(b"x")
    # rglob order is not guaranteed, but an exact hit in the library must win.
    assert csm._resolve("thing.gguf", tmp_path) == exact.resolve()


# ---------------------------------------------------------------------------
# the device classification
# ---------------------------------------------------------------------------
def test_the_repo_itself_is_not_on_a_slow_device():
    """The repo lives on the fast device; the check is only meaningful there."""
    kind = csm._device_kind(REPO)
    assert kind not in ("fuse", "rotational"), f"the repo is on a {kind} device"


def test_a_fuse_mount_is_detected():
    """A FUSE filesystem is 'fuse' even when its st_dev looks like a disk."""
    fuse = [m for m in csm._mounts() if m[1].startswith("fuse")]
    if not fuse:
        pytest.skip("no FUSE mount on this machine")
    point, fs_type, _source = fuse[0]
    assert csm._device_kind(point) == "fuse"


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------
def test_no_stack_leaves_a_local_model_on_a_slow_device():
    """Every stack whose model is present locally must be on a fast device.

    Tiers whose model is not in the library are skipped: they are fetched
    from Hugging Face into the library, which is on the fast device.
    """
    rows = csm.collect()
    assert rows, "no stack tiers found - the stacks directory moved?"
    slow = [
        f"{r['stack']}/{r['role']} -> {r['path']} ({r['kind']})"
        for r in rows
        if r.get("slow")
    ]
    assert not slow, "models on a slow device:\n  " + "\n  ".join(slow)
