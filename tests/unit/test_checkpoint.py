"""Unit tests for cortex.checkpoint (S5.3)."""

import pytest

from cortex.checkpoint import StrataCheckpoint


def test_save_restore_roundtrip(tmp_path):
    ck = StrataCheckpoint(tmp_path)
    state = {"decay_multiplier": 1.8, "store": {"a": 1}, "params": {"threshold": 2}}
    path = ck.save(state, "v1")
    assert path.exists()
    assert ck.restore("v1") == state


def test_restore_unknown_tag_raises(tmp_path):
    ck = StrataCheckpoint(tmp_path)
    import pytest

    with pytest.raises(FileNotFoundError):
        ck.restore("does_not_exist")


def test_auto_checkpoint_only_at_high_pes(tmp_path):
    ck = StrataCheckpoint(tmp_path)
    assert ck.auto_checkpoint({"x": 1}, pes=70) is None
    path = ck.auto_checkpoint({"x": 1}, pes=90)
    assert path is not None
    # restore by stripping the checkpoint_ prefix from the filename stem
    tag = path.stem.replace("checkpoint_", "")
    assert ck.restore(tag) == {"x": 1}


def test_list_checkpoints(tmp_path):
    ck = StrataCheckpoint(tmp_path)
    ck.save({"a": 1}, "one")
    ck.save({"a": 2}, "two")
    names = [p.name for p in ck.list_checkpoints()]
    assert names == ["checkpoint_one.json", "checkpoint_two.json"]


def test_path_traversal_rejected(tmp_path):
    ck = StrataCheckpoint(tmp_path)
    with pytest.raises(ValueError):
        ck.save({"x": 1}, "../evil")
    with pytest.raises(ValueError):
        ck.restore("../../etc/passwd")
    with pytest.raises(ValueError):
        ck.save({"x": 1}, "")
    # no file escaped the checkpoint directory
    assert not (tmp_path.parent / "evil.json").exists()