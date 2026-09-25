"""Unit tests for harness.stack.schema — the §B3 stack data model.

Covers the T35 acceptance: both authored default stacks round-trip through the
data model and validate, and a stack with a missing or duplicate tier role is
rejected.  Stack files are written to ``tmp_path`` except where the test is
specifically about the repo's own ``stacks/`` directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harness.stack.schema import (
    DEFAULT_ROUTING,
    ROLE_FACE,
    ROLES,
    STACK_VERSION,
    Stack,
    Tier,
    delete_stack,
    list_stacks,
    load_stack,
    save_stack,
    stack_path,
    stacks_dir,
    validate_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STACKS = REPO_ROOT / "stacks"


def _tier(role, **over):
    base = {
        "role": role,
        "repo": "acme/model-GGUF",
        "file": "model-Q6_K.gguf",
        "ctx": 8192,
    }
    base.update(over)
    return Tier(**base)


# --- the authored default stacks -------------------------------------------


@pytest.mark.parametrize("name", ["peer-2tier", "peer-3tier"])
def test_authored_stacks_validate(name):
    """Both default stacks are on disk, parse, and pass shape validation."""
    assert (STACKS / f"{name}.json").is_file(), f"stacks/{name}.json is missing"

    stack = load_stack(name)
    assert stack.name == name
    assert stack.version == STACK_VERSION
    assert stack.roles == list(stack.roles) and stack.tiers
    assert stack.routing == DEFAULT_ROUTING
    assert validate_shape(stack) == []


def test_authored_stacks_round_trip_through_the_data_model():
    """to_dict is byte-faithful: reloading a re-serialised stack is identical."""
    for name in ("peer-2tier", "peer-3tier"):
        raw = json.loads((STACKS / f"{name}.json").read_text(encoding="utf-8"))
        assert Stack.from_dict(raw).to_dict() == raw, f"{name} is not canonical"


def test_peer_2tier_is_face_plus_worker():
    stack = load_stack("peer-2tier")
    assert stack.roles == ["face", "worker"]
    face, worker = stack.tiers
    # the face tier owns the full 256K window, the MTP draft and the projector
    assert face.ctx == 262144
    assert face.spec == {"type": "draft-mtp", "n_max": 3}
    assert face.mmproj == "mmproj-F16.gguf"
    assert face.ts == "1,1" and face.pin == "HIP_VISIBLE_DEVICES=0,1"
    # a 4B worker has no speculative block, no projector and no tensor split
    assert worker.ctx == 131072
    assert (worker.spec, worker.mmproj, worker.ts) == (None, None, None)
    assert worker.pin == "HIP_VISIBLE_DEVICES=1"
    for tier in stack.tiers:
        assert tier.backend == "vulkan"
        assert (tier.cache_k, tier.cache_v) == ("q8_0", "q8_0")


def test_peer_3tier_is_face_plus_agency_plus_mechanics():
    stack = load_stack("peer-3tier")
    assert stack.roles == ["face", "agency", "mechanics"]
    # LOCAL-STACKS §8: 128K + 32K + 16K, the trade against the 2-tier's 256K face
    assert [tier.ctx for tier in stack.tiers] == [131072, 32768, 16384]
    assert stack.tiers[0].file == "Qwen3.8-27B-UD-Q6_K.gguf"
    assert "Ornith-1.5-9B" in stack.tiers[1].file
    assert "2B" in stack.tiers[2].file


def test_saving_the_authored_stacks_reproduces_the_files(tmp_path):
    """save(load(authored)) == authored, so the files stay hand-editable."""
    for name in ("peer-2tier", "peer-3tier"):
        original = (STACKS / f"{name}.json").read_text(encoding="utf-8")
        path = save_stack(load_stack(name), root=tmp_path)
        assert path == tmp_path / "stacks" / f"{name}.json"
        assert path.read_text(encoding="utf-8") == original


# --- role validation: missing / duplicate -----------------------------------


def test_valid_stack_reports_no_errors():
    stack = Stack(name="ok", tiers=[_tier(ROLE_FACE), _tier("worker")])
    assert validate_shape(stack) == []


def test_duplicate_tier_role_is_rejected():
    stack = Stack(name="dup", tiers=[_tier("face"), _tier("worker"), _tier("worker")])
    errors = validate_shape(stack)
    assert len(errors) == 1
    assert "duplicate tier role 'worker'" in errors[0]


def test_duplicate_face_role_is_rejected():
    stack = Stack(name="dup", tiers=[_tier("face"), _tier("face")])
    errors = validate_shape(stack)
    assert any("duplicate tier role 'face'" in err for err in errors)


def test_missing_tier_role_is_rejected():
    stack = Stack(name="norole", tiers=[_tier("face"), _tier("")])
    errors = validate_shape(stack)
    assert errors == ["tier 2: missing tier role"]


def test_unknown_tier_role_is_rejected():
    stack = Stack(name="bogus", tiers=[_tier("face"), _tier("oracle")])
    errors = validate_shape(stack)
    assert any("'oracle' is not one of" in err for err in errors)


def test_empty_stack_is_rejected():
    errors = validate_shape(Stack(name="empty", tiers=[]))
    assert errors and "no tiers" in errors[0]


def test_tier_missing_repo_or_file_is_rejected():
    stack = Stack(name="nofile", tiers=[_tier("face", repo="", file="")])
    errors = validate_shape(stack)
    assert any("missing repo" in err for err in errors)
    assert any("missing file" in err for err in errors)


def test_non_positive_ctx_is_rejected():
    errors = validate_shape(Stack(name="ctx", tiers=[_tier("face", ctx=0)]))
    assert any("ctx must be a positive integer" in err for err in errors)


def test_all_four_roles_are_allowed_in_one_stack():
    stack = Stack(name="full", tiers=[_tier(role) for role in ROLES])
    assert validate_shape(stack) == []


def test_load_stack_rejects_a_duplicate_role_file(tmp_path):
    _write(tmp_path, "dup", {
        "name": "dup", "version": 1,
        "tiers": [
            {"role": "face", "repo": "a/b", "file": "b.gguf"},
            {"role": "face", "repo": "a/c", "file": "c.gguf"},
        ],
        "routing": {"workers_as": "subagent"},
    })
    with pytest.raises(ValueError, match="invalid shape"):
        load_stack("dup", root=tmp_path)


def test_load_stack_rejects_a_missing_role_file(tmp_path):
    _write(tmp_path, "norole", {
        "name": "norole", "version": 1,
        "tiers": [{"repo": "a/b", "file": "b.gguf"}],
    })
    with pytest.raises(ValueError, match="missing required field"):
        load_stack("norole", root=tmp_path)


# --- parsing ----------------------------------------------------------------


def test_tier_to_dict_omits_unset_optional_fields():
    out = _tier("worker").to_dict()
    assert set(out) == {
        "role", "repo", "file", "ctx", "ngl", "backend", "cache_k", "cache_v"}
    for key in ("spec", "mmproj", "pin", "ts"):
        assert key not in out


def test_tier_from_dict_applies_defaults():
    tier = Tier.from_dict({"role": "worker", "repo": "a/b", "file": "b.gguf"})
    assert (tier.ctx, tier.ngl, tier.backend) == (8192, 99, "vulkan")
    assert (tier.cache_k, tier.cache_v) == ("q8_0", "q8_0")
    assert (tier.spec, tier.mmproj, tier.pin, tier.ts) == (None, None, None, None)


def test_tier_from_dict_coerces_numeric_strings():
    tier = Tier.from_dict({
        "role": "face", "repo": "a/b", "file": "b.gguf", "ctx": "4096", "ngl": "99"})
    assert (tier.ctx, tier.ngl) == (4096, 99)


@pytest.mark.parametrize("tier_data, match", [
    ({"repo": "a/b", "file": "b.gguf"}, "missing required field"),
    ({"role": "", "repo": "a/b", "file": "b.gguf"}, "must not be empty"),
    ({"role": "face", "repo": "a/b", "file": "b.gguf", "ctx": 0}, "must be >= 1"),
    ({"role": "face", "repo": "a/b", "file": "b.gguf", "ngl": -1}, "must be >= 0"),
    ({"role": "face", "repo": "a/b", "file": "b.gguf", "ctx": "wide"}, "must be an integer"),
    ({"role": "face", "repo": "a/b", "file": "b.gguf", "spec": []}, "spec must be an object"),
    ({"role": "face", "repo": "a/b", "file": "b.gguf", "cxt": 4096}, "unknown field"),
])
def test_tier_from_dict_rejects_malformed_fields(tier_data, match):
    with pytest.raises(ValueError, match=match):
        Tier.from_dict(tier_data)


def test_stack_from_dict_requires_known_version():
    raw = {"name": "s", "version": 99, "tiers": []}
    with pytest.raises(ValueError, match="unsupported stack version 99"):
        Stack.from_dict(raw)


def test_stack_from_dict_defaults_routing_to_subagents():
    stack = Stack.from_dict({
        "name": "s", "tiers": [{"role": "face", "repo": "a/b", "file": "b.gguf"}]})
    assert stack.routing == DEFAULT_ROUTING
    assert stack.version == STACK_VERSION


def test_stack_from_dict_rejects_unknown_top_level_fields():
    with pytest.raises(ValueError, match="unknown field"):
        Stack.from_dict({"name": "s", "tiers": [], "routng": {}})


def test_stack_from_dict_rejects_non_list_tiers():
    with pytest.raises(ValueError, match="tiers must be a list"):
        Stack.from_dict({"name": "s", "tiers": {"role": "face"}})


# --- paths ------------------------------------------------------------------


def test_stacks_dir_defaults_to_the_repo_stacks_directory():
    assert stacks_dir() == STACKS


def test_stacks_dir_honours_a_root_override(tmp_path):
    assert stacks_dir(tmp_path) == tmp_path / "stacks"


def test_stack_path_is_under_stacks(tmp_path):
    assert stack_path("peer-2tier", root=tmp_path) == tmp_path / "stacks" / "peer-2tier.json"


def test_stack_path_tolerates_a_json_suffix(tmp_path):
    assert stack_path("peer-2tier.json", root=tmp_path) == stack_path("peer-2tier", root=tmp_path)


@pytest.mark.parametrize("name", [
    "", "   ", "..", "../evil", "a/b", "sub\\win", "/abs/evil", ".hidden", "a\x00b",
])
def test_stack_path_rejects_traversal(name, tmp_path):
    with pytest.raises(ValueError):
        stack_path(name, root=tmp_path)


# --- file io ----------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    stack = Stack(name="rt", tiers=[_tier("face", ctx=131072, ts="1,1")])
    path = save_stack(stack, root=tmp_path)
    assert path.is_file()
    assert path.read_text(encoding="utf-8").endswith("\n")

    reloaded = load_stack("rt", root=tmp_path)
    assert reloaded.to_dict() == stack.to_dict()
    assert reloaded.tiers[0].ts == "1,1"


def test_save_creates_the_stacks_directory(tmp_path):
    root = tmp_path / "fresh"
    save_stack(Stack(name="s", tiers=[_tier("face")]), root=root)
    assert (root / "stacks" / "s.json").is_file()


def test_save_overwrites_in_place(tmp_path):
    for ctx in (8192, 4096):
        save_stack(Stack(name="s", tiers=[_tier("face", ctx=ctx)]), root=tmp_path)
    assert load_stack("s", root=tmp_path).tiers[0].ctx == 4096
    assert len(list((tmp_path / "stacks").iterdir())) == 1


def test_save_leaves_no_temp_files_behind(tmp_path):
    save_stack(Stack(name="s", tiers=[_tier("face")]), root=tmp_path)
    assert [p.name for p in (tmp_path / "stacks").iterdir()] == ["s.json"]


def test_load_missing_stack_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stack("nope", root=tmp_path)


def test_load_rejects_invalid_json(tmp_path):
    _write_raw(tmp_path, "broken", "{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_stack("broken", root=tmp_path)


def test_list_stacks_is_empty_without_a_directory(tmp_path):
    assert list_stacks(root=tmp_path) == []


def test_list_stacks_summarises_each_file_sorted(tmp_path):
    save_stack(Stack(name="peer-3tier", tiers=[_tier(r) for r in ("face", "agency")]),
               root=tmp_path)
    save_stack(Stack(name="peer-2tier", tiers=[_tier(r) for r in ("face", "worker")]),
               root=tmp_path)

    assert list_stacks(root=tmp_path) == [
        {"name": "peer-2tier", "tiers": 2, "roles": ["face", "worker"]},
        {"name": "peer-3tier", "tiers": 2, "roles": ["face", "agency"]},
    ]


def test_list_stacks_lists_the_repo_defaults():
    rows = {row["name"]: row for row in list_stacks()}
    assert rows["peer-2tier"]["roles"] == ["face", "worker"]
    assert rows["peer-3tier"]["roles"] == ["face", "agency", "mechanics"]


def test_list_stacks_skips_an_unreadable_file(tmp_path):
    save_stack(Stack(name="good", tiers=[_tier("face")]), root=tmp_path)
    _write_raw(tmp_path, "bad", "{not json")
    assert [row["name"] for row in list_stacks(root=tmp_path)] == ["good"]


def test_delete_stack_reports_whether_a_file_went_away(tmp_path):
    save_stack(Stack(name="gone", tiers=[_tier("face")]), root=tmp_path)
    assert delete_stack("gone", root=tmp_path) is True
    assert delete_stack("gone", root=tmp_path) is False
    assert not (tmp_path / "stacks" / "gone.json").exists()


def test_delete_stack_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        delete_stack("../evil", root=tmp_path)


def test_save_stack_rejects_a_non_stack(tmp_path):
    with pytest.raises(ValueError, match="expected a Stack"):
        save_stack({"name": "s"}, root=tmp_path)


# --- helpers ----------------------------------------------------------------


def _write(root, name, payload):
    path = stack_path(name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_raw(root, name, text):
    path = stack_path(name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
