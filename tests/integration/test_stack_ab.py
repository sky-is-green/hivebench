"""Integration: stack A/B — two stacks, the same turns, one report.

Exercises the whole path offline: both arms go through
``experiments.paired_ab.run_paired`` (the reused quality engine), the throughput
half goes through an injected stand-in for the harness engines A/B
(``POST /v1/engines/ab/bench``), and the report is written to ``results.json``
with every printed number traceable to a JSON location.
"""

import importlib
import json
import sys
import types

import numpy as np
import pytest

from cortex.baselines.runner import FIFO_WINDOW_TOKENS
from experiments.stack_ab import (
    ENGINES_AB_ROUTE,
    STACKS_STATUS_ROUTE,
    applied_face_url,
    bench_stacks,
    engine_profile,
    face_tier,
    format_report,
    main,
    per_turn_rows,
    quality_verdict,
    run_stack_ab,
    stack_doc,
    stack_summary,
    synthetic_stack,
    turn_alignment,
)


# --------------------------------------------------------------------------
# offline stand-ins
# --------------------------------------------------------------------------

class _DistinctEmbedUltra:
    """FakeUltraSmall with distinct embeddings: the stock fake returns a constant
    vector, which makes the dedup pass collapse query and reply chunks as
    identical. One-hot term embeddings keep them distinct (mirrors
    tests/unit/test_paired_ab.py)."""

    def score(self, query, chunks):
        from cortex.e2e import FakeUltraSmall

        return FakeUltraSmall().score(query, chunks)

    def embed(self, text):
        from experiments.retrieval_diagnostic import _content_terms

        v = np.zeros(16)
        for i, w in enumerate(sorted(_content_terms(text))[:16]):
            v[i] = 1.0
        return v


def _ultra():
    from sieve.medium import MediumDrone

    return _DistinctEmbedUltra(), MediumDrone(score_pair_fn=lambda q, c: 0.5)


def _conversations():
    """Two user turns per conversation; the later turn repeats the earlier one, so
    its facts are retrievable from history."""
    return [{
        "conversation_id": "c1",
        "profile": "code",
        "turns": [
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
            {"role": "assistant",
             "content": "Use JWT tokens with rotation and short expiry."},
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
        ],
    }]


def _context_aware_backend(tag: str, states_facts: bool):
    """Stub backend: names the fixture facts only when the assembled context
    already carries them — the behaviour a real model should show. ``tag`` keys
    the replies per arm so a test can prove the two arms really differed."""
    from experiments.retrieval_diagnostic import _answer_fact_terms, _content_terms

    answers = {"Which tokens do I use for auth expiry?":
               "Use JWT tokens with rotation and short expiry."}

    class _Backend:
        def __init__(self):
            self.calls = 0

        def generate(self, context, query, sampling=None):
            self.calls += 1
            answer = answers.get(query, "")
            facts = _answer_fact_terms(query, answer) if answer else set()
            if states_facts and facts and facts <= _content_terms(context or ""):
                return f"[{tag}] The answer is: " + " ".join(sorted(facts))
            return f"[{tag}] I am not sure."

    return _Backend()


def _stacks():
    return (synthetic_stack("peer-2tier", ["face", "worker"]),
            synthetic_stack("peer-3tier", ["face", "worker", "mechanics"]))


def _bench_response(a_tps: float, b_tps: float,
                    a_name: str = "A-FACE", b_name: str = "B-FACE") -> dict:
    """A response in the exact shape ``harness/app.py``'s ``_ab_handle`` returns."""
    return {
        "winner": "A" if a_tps > b_tps else ("B" if b_tps > a_tps else "tie"),
        "a_tok_per_sec": a_tps,
        "b_tok_per_sec": b_tps,
        "a": {"profile": a_name, "port": 8100, "tok_per_sec": a_tps,
              "total_tokens": 100, "total_seconds": 1.0, "results": []},
        "b": {"profile": b_name, "port": 8101, "tok_per_sec": b_tps,
              "total_tokens": 100, "total_seconds": 1.0, "results": []},
        "port_a": 8100, "port_b": 8101,
    }


# --------------------------------------------------------------------------
# stack documents
# --------------------------------------------------------------------------

def test_face_tier_is_the_face_role():
    doc = synthetic_stack("two", ["face", "worker"])
    assert face_tier(doc)["role"] == "face"
    assert face_tier(doc)["ctx"] == 8192


def test_face_tier_falls_back_to_first_tier():
    doc = synthetic_stack("no-face", ["worker", "mechanics"])
    assert face_tier(doc)["role"] == "worker"
    summary = stack_summary(doc)
    assert summary["roles"] == ["worker", "mechanics"]
    assert summary["face"]["role"] == "worker"


def test_stack_summary_reports_roles_and_launch_provenance():
    a, b = _stacks()
    sa, sb = stack_summary(a), stack_summary(b)
    assert sa["name"] == "peer-2tier" and sa["tier_count"] == 2
    assert sa["roles"] == ["face", "worker"]
    assert sb["tier_count"] == 3 and sb["roles"] == ["face", "worker", "mechanics"]
    assert sb["routing"] == {"workers_as": "subagent"}
    # The launch block names its source; it never carries a second copy of the
    # tier -> load_options mapping (ADR-L7). Three legitimate states, all
    # recorded: T37 landed, T34's stub is present, or the module is absent.
    for summary in (sa, sb):
        source = summary["launch"]["source"]
        assert source == "harness.stack.manager.tier_load_options" or \
            source.startswith("unavailable (")
        assert ("load_options" in summary["launch"]) == (
            source == "harness.stack.manager.tier_load_options")


def _fake_t37(monkeypatch, fn):
    """Install a stand-in T37 `tier_load_options`, creating the frozen module
    when T34 has not been merged into this branch yet."""
    try:
        manager = importlib.import_module("harness.stack.manager")
    except ImportError:
        manager = types.ModuleType("harness.stack.manager")
        pkg = sys.modules.get("harness.stack")
        if pkg is None:
            pkg = types.ModuleType("harness.stack")
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, "harness.stack", pkg)
        monkeypatch.setitem(sys.modules, "harness.stack.manager", manager)
        monkeypatch.setattr(pkg, "manager", manager, raising=False)
    monkeypatch.setattr(manager, "tier_load_options", fn, raising=False)
    return manager


def test_launch_block_survives_the_frozen_t34_stub(monkeypatch):
    """T34 froze `tier_load_options`'s signature; its body is T37's. T43 runs in
    the same wave as the stub, so `NotImplementedError` must degrade to a named
    `unavailable` block, not crash the A/B."""
    _fake_t37(monkeypatch, lambda tier: (_ for _ in ()).throw(NotImplementedError))
    summary = stack_summary(synthetic_stack("stubbed", ["face", "worker"]))
    assert summary["launch"] == {
        "source": "unavailable (tier_load_options not implemented yet)"}
    assert "load_options" not in summary["launch"]


def test_launch_block_uses_t37_when_it_has_landed(monkeypatch):
    _fake_t37(monkeypatch,
              lambda tier: {"context": tier["ctx"], "gpu_layers": tier["ngl"]})
    summary = stack_summary(synthetic_stack("live", ["face", "worker"]))
    assert summary["launch"]["source"] == "harness.stack.manager.tier_load_options"
    assert summary["launch"]["load_options"] == {"context": 8192, "gpu_layers": 99}
    # and the profile carries them into the engines-A/B request
    assert engine_profile(summary, "http://h:1")["load_options"] == \
        {"context": 8192, "gpu_layers": 99}


def test_stack_doc_accepts_a_to_dict_object():
    class _Stack:
        def to_dict(self):
            return synthetic_stack("obj", ["face", "worker"])

    assert stack_doc(_Stack())["name"] == "obj"
    with pytest.raises(ValueError):
        stack_doc(object())


def test_engine_profile_name_is_the_face_gguf_stem():
    summary = stack_summary(synthetic_stack("p", ["face", "worker"]))
    profile = engine_profile(summary, "http://127.0.0.1:8100")
    assert profile["name"] == "p-face-0"
    assert profile["kind"] == "llama_cpp"
    assert profile["base_url"] == "http://127.0.0.1:8100"


def test_caveats_flag_a_colliding_engine_profile():
    """Two stacks whose face tiers are the same GGUF cannot be benched against
    each other — the report must say so instead of printing a 1.0 ratio."""
    shared = synthetic_stack("twin-a", ["face", "worker"])
    shared_b = synthetic_stack("twin-b", ["face", "mechanics"])
    for tier in shared_b["tiers"]:
        tier["file"] = "Qwen3.8-27B-UD-Q6_K.gguf"
    for tier in shared["tiers"]:
        tier["file"] = "Qwen3.8-27B-UD-Q6_K.gguf"
    ultra, medium = _ultra()
    report = run_stack_ab(
        shared, shared_b, _conversations(),
        lambda stack: _context_aware_backend("x", True),
        ultra=ultra, medium=medium, fifo_budget=1)
    assert report["arms"]["a"]["engine_profile"]["name"] == \
        report["arms"]["b"]["engine_profile"]["name"] == "Qwen3.8-27B-UD-Q6_K"
    assert any("same engines-A/B profile" in c for c in report["caveats"])
    assert any("launch config unavailable" in c for c in report["caveats"])
    assert any("throughput not measured" in c for c in report["caveats"])


# --------------------------------------------------------------------------
# the run: same turns, two stacks, one report
# --------------------------------------------------------------------------

def test_same_turns_two_stacks_one_report():
    convs = _conversations()
    stack_a, stack_b = _stacks()
    ultra, medium = _ultra()

    def backend_for(stack):
        # arm A states the facts it was given, arm B never does
        return _context_aware_backend(str(stack["name"]),
                                      states_facts=stack["name"] == "peer-2tier")

    report = run_stack_ab(stack_a, stack_b, convs, backend_for,
                          ultra=ultra, medium=medium, fifo_budget=1)

    # same turns — proven in the artifact, not asserted in prose
    assert report["turns"]["identical"] is True
    assert report["turns"]["count"] == 1
    assert report["turns"]["keys"] == [["c1", 2]]
    assert report["turns"]["only_a"] == [] and report["turns"]["only_b"] == []
    assert report["quality"]["a"]["turns_compared"] == 1
    assert report["quality"]["b"]["turns_compared"] == 1

    # one report, two stacks, distinguishable arms
    assert report["arms"]["a"]["name"] == "peer-2tier"
    assert report["arms"]["b"]["name"] == "peer-3tier"
    assert report["quality"]["a"]["hive_answer_recall"] == 100.0
    assert report["quality"]["b"]["hive_answer_recall"] == 0.0
    assert report["quality_verdict"]["winner"] == "a"
    assert report["quality_verdict"]["a_wins"] > report["quality_verdict"]["b_wins"]

    # per-turn rows pair the arms turn for turn
    assert len(report["per_turn"]) == 1
    row = report["per_turn"][0]
    assert (row["conversation_id"], row["turn"]) == ("c1", 2)
    assert row["a"]["answer_hive_hit_ratio"] == 1.0
    assert row["b"]["answer_hive_hit_ratio"] == 0.0
    assert row["a"]["hive_ctx_tokens"] > 0 and row["b"]["fifo_ctx_tokens"] >= 0

    # counts are reported but excluded from the verdict
    counts = [m for m in report["quality_verdict"]["metrics"]
              if m["direction"] is None]
    assert [m["name"] for m in counts] == ["turns_compared"]
    assert counts[0]["better"] is None


def test_turn_alignment_flags_a_mispaired_run():
    left = {"turns": [{"conversation_id": "c1", "turn": 2}]}
    right = {"turns": [{"conversation_id": "c1", "turn": 2},
                       {"conversation_id": "c2", "turn": 3}]}
    align = turn_alignment(left, right)
    assert align["identical"] is False
    assert align["only_b"] == [["c2", 3]] and align["only_a"] == []
    assert per_turn_rows(left, right) == [
        {"conversation_id": "c1", "turn": 2,
         "a": {f: None for f in ("answer_hive_hit_ratio", "answer_fifo_hit_ratio",
                                "hive_ctx_tokens", "fifo_ctx_tokens")},
         "b": {f: None for f in ("answer_hive_hit_ratio", "answer_fifo_hit_ratio",
                                 "hive_ctx_tokens", "fifo_ctx_tokens")}}]


def test_quality_verdict_ties_on_identical_arms():
    convs = _conversations()
    stack_a, _ = _stacks()
    ultra, medium = _ultra()
    report = run_stack_ab(
        stack_a, stack_a, convs,
        lambda stack: _context_aware_backend("same", states_facts=True),
        ultra=ultra, medium=medium, fifo_budget=1)
    assert report["turns"]["identical"] is True
    assert report["quality_verdict"]["winner"] == "tie"
    assert report["quality_verdict"]["a_wins"] == 0
    assert report["quality_verdict"]["b_wins"] == 0
    assert all(m["better"] == "tie" for m in report["quality_verdict"]["metrics"]
               if m["direction"] is not None)


def test_empty_corpus_arms_still_agree():
    ultra, medium = _ultra()
    stack_a, stack_b = _stacks()
    report = run_stack_ab(stack_a, stack_b, [],
                          lambda stack: _context_aware_backend("x", True),
                          ultra=ultra, medium=medium)
    assert report["turns"]["identical"] is True
    assert report["turns"]["count"] == 0
    assert report["quality_verdict"]["metrics"][0]["a"] is None
    assert report["quality_verdict"]["winner"] == "tie"


# --------------------------------------------------------------------------
# the engines A/B seam
# --------------------------------------------------------------------------

def test_bench_stacks_returns_the_harness_response_verbatim():
    seen = {}
    expected = _bench_response(40.0, 20.0, "peer-2tier-face-0", "peer-3tier-face-0")

    def fake_post(url, body, **kwargs):
        seen["url"] = url
        seen["body"] = body
        return expected

    stack_a, stack_b = _stacks()
    bench = bench_stacks(engine_profile(stack_summary(stack_a), "http://h:8100"),
                         engine_profile(stack_summary(stack_b), "http://h:8101"),
                         harness_url="http://h:9000/", base_port=8100,
                         prompts=["hi", "there"], post=fake_post)
    assert seen["url"] == f"http://h:9000{ENGINES_AB_ROUTE}"
    # only the keys _ab_handle reads; names, not profile blobs
    assert seen["body"] == {"profile_a": "peer-2tier-face-0",
                            "profile_b": "peer-3tier-face-0",
                            "basePort": 8100, "prompts": ["hi", "there"]}
    assert bench["ok"] is True
    # the winner is the harness's, not recomputed here
    assert bench["response"] == expected
    assert bench["response"]["winner"] == "A"
    assert bench["engines_known"] == {"a": True, "b": True}


def test_bench_stacks_flags_a_profile_the_harness_answered_with_another():
    bench = bench_stacks({"name": "a"}, {"name": "b"},
                         post=lambda url, body, **kw: _bench_response(1.0, 2.0, "c", "d"))
    assert bench["ok"] is True
    assert bench["engines_known"] == {"a": False, "b": False}


def test_bench_stacks_survives_an_unreachable_harness():
    def boom(url, body, **kwargs):
        raise ConnectionError("connection refused")

    bench = bench_stacks({"name": "a"}, {"name": "b"}, post=boom)
    assert bench["ok"] is False
    assert "ConnectionError" in bench["error"]
    assert bench["route"] == ENGINES_AB_ROUTE


def test_bench_stacks_flags_an_unknown_engine():
    bench = bench_stacks({"name": "a"}, {"name": "b"},
                         post=lambda url, body, **kw: {"detail": "unknown engine 'a'"})
    assert bench["ok"] is False
    assert bench["engines_known"] == {"a": False, "b": False}
    assert bench["response"] == {"detail": "unknown engine 'a'"}


def test_applied_face_url_reads_the_status_route():
    seen = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        return {"ok": True, "stack": "peer-2tier",
                "tiers": [{"role": "worker", "port": 8101},
                          {"role": "face", "port": 8100}]}

    assert applied_face_url("http://h:9000/", "peer-2tier", get=fake_get) == \
        "http://127.0.0.1:8100"
    assert seen["url"] == f"http://h:9000{STACKS_STATUS_ROUTE}"


def test_applied_face_url_refuses_a_different_applied_stack():
    with pytest.raises(LookupError, match="applied"):
        applied_face_url("http://h:9000", "peer-3tier",
                         get=lambda url, **kw: {"stack": "peer-2tier", "tiers": []})


# --------------------------------------------------------------------------
# the CLI + traceability
# --------------------------------------------------------------------------

def test_main_mock_writes_results_json(tmp_path, capsys):
    convs = _conversations()
    corpus = tmp_path / "convs"
    corpus.mkdir()
    (corpus / "c1.json").write_text(json.dumps(convs[0]), encoding="utf-8")
    out = tmp_path / "results.json"

    assert main(["--mock", "--conversations", str(corpus), "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Stack A/B" in printed
    assert str(out.resolve()) in printed

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["kind"] == "stack_ab"
    assert doc["turns"]["identical"] is True
    assert doc["arms"]["a"]["name"] == "mock-2tier"
    assert doc["arms"]["b"]["name"] == "mock-3tier"
    assert doc["throughput"]["ok"] is False  # offline: no live harness
    # every figure the console printed is locatable in results.json
    assert "arms.a.tier_count" in printed and str(doc["arms"]["a"]["tier_count"]) in printed
    assert "turns.identical" in printed
    assert "quality_verdict.winner" in printed
    assert "quality_verdict.metrics.turns_compared" in printed
    assert "settings.fifo_budget_tokens" in printed
    assert doc["settings"]["fifo_budget_tokens"] == FIFO_WINDOW_TOKENS
    assert doc["quality"]["a"]["hive_answer_recall"] == 0.0  # mock replies carry no facts


def test_format_report_renders_only_from_the_dict():
    stack_a, stack_b = _stacks()
    convs = _conversations()
    ultra, medium = _ultra()
    report = run_stack_ab(
        stack_a, stack_b, convs,
        lambda stack: _context_aware_backend(str(stack["name"]), states_facts=True),
        ultra=ultra, medium=medium, fifo_budget=1,
        bench=bench_stacks({"name": "x"}, {"name": "y"},
                           post=lambda url, body, **kw: _bench_response(40.0, 20.0)))
    text = format_report(report)
    assert report["quality_verdict"]["winner"] in text
    assert "[turns.identical]" in text
    assert "[throughput.response.winner]" in text
    assert "peer-2tier" in text and "peer-3tier" in text
    for caveat in report["caveats"]:
        assert f"caveat[0]" in text and caveat in text
    # the rendered text survives a JSON round-trip: it is read off the artifact
    assert format_report(json.loads(json.dumps(report))) == text


def test_format_report_surfaces_a_turn_mismatch():
    left = {"turns": [{"conversation_id": "c1", "turn": 2}]}
    right = {"turns": []}
    text = format_report({"arms": {}, "turns": turn_alignment(left, right),
                          "quality_verdict": quality_verdict({}, {}),
                          "throughput": {"ok": False, "error": "down"}})
    assert "turn sets differ" in text
    assert "[turns.only_a]" in text


def test_main_missing_corpus(tmp_path):
    assert main(["--mock", "--conversations", str(tmp_path / "none")]) == 2


def test_main_live_needs_two_stacks(tmp_path):
    corpus = tmp_path / "convs"
    corpus.mkdir()
    (corpus / "c1.json").write_text(json.dumps(_conversations()[0]), encoding="utf-8")
    assert main(["--live", "--conversations", str(corpus)]) == 2


def test_main_live_reports_a_missing_stack(tmp_path, capsys):
    corpus = tmp_path / "convs"
    corpus.mkdir()
    (corpus / "c1.json").write_text(json.dumps(_conversations()[0]), encoding="utf-8")
    out = tmp_path / "results.json"
    # T35 has not landed on this branch and there is no stacks/peer-2tier.json
    # here either, so the stack cannot be read at all.
    assert main(["--live", "--stack-a", "peer-2tier", "--stack-b", "peer-3tier",
                 "--conversations", str(corpus), "--output", str(out)]) == 2
    assert "error:" in capsys.readouterr().out
