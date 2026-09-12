"""paired_ab — paired strata-vs-FIFO answer-quality A/B (offline)."""

import json

import numpy as np

from experiments.paired_ab import run_paired
from experiments.retrieval_diagnostic import _answer_fact_terms, _content_terms


class _DistinctEmbedUltra:
    """FakeUltraSmall with distinct embeddings: the stock fake returns a
    constant vector, which makes the dedup pass collapse query and reply
    chunks as identical. One-hot term embeddings keep them distinct."""

    def score(self, query, chunks):
        from cortex.e2e import FakeUltraSmall

        return FakeUltraSmall().score(query, chunks)

    def embed(self, text):
        v = np.zeros(16)
        for i, w in enumerate(sorted(_content_terms(text))[:16]):
            v[i] = 1.0
        return v


class _ContextAwareBackend:
    """Stub backend: answers with the fixture facts only when the context
    already contains them — the behavior a real model should show when the
    needed facts are present. Deterministic and network-free."""

    def __init__(self, conversations):
        from experiments.retrieval_diagnostic import _fixture_answer_map

        per_conv = _fixture_answer_map(conversations)
        self.answers = {
            q: a
            for conv_answers in per_conv.values()
            for q, a in conv_answers.items()
        }

    def generate(self, context, query, sampling=None):
        answer = self.answers.get(query, "")
        facts = _answer_fact_terms(query, answer) if answer else set()
        if facts and facts <= _content_terms(context or ""):
            return "The answer is: " + " ".join(sorted(facts))
        return "I am not sure."


class _DumbBackend:
    """Stub backend that never states the facts."""

    def generate(self, context, query, sampling=None):
        return "I am not sure."


def _conv():
    """Two user turns; the second repeats the first, so its facts are
    retrievable from history (>=2 shared query terms; 'handle' is a diagnostic
    stopword, so the query needs distinct content terms)."""
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


def _ultra():
    from sieve.medium import MediumDrone

    return _DistinctEmbedUltra(), MediumDrone(score_pair_fn=lambda q, c: 0.5)


def test_strict_hive_win_when_fifo_drops_the_fact():
    convs = _conv()
    ultra, medium = _ultra()
    report = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                        fifo_budget=1)  # FIFO window fits only the current turn
    m = report["metrics"]
    assert m["turns_compared"] == 1
    assert m["hive_only"] == 1
    assert m["fifo_only"] == 0
    assert m["hive_answer_recall"] == 100.0
    assert m["fifo_answer_recall"] == 0.0
    assert m["hive_ge_fifo_ratio"] == 100.0
    assert m["strict_hive_only_ratio"] == 100.0
    assert m["hive_avg_fact_hit_ratio"] == 1.0
    assert m["fifo_avg_fact_hit_ratio"] == 0.0
    # Context fidelity: the strata answer's terms came from its own context;
    # the FIFO arm had nothing to draw on.
    assert m["hive_avg_context_fidelity"] > 0.0
    assert m["fifo_avg_context_fidelity"] == 0.0
    assert m["fidelity_hive_gt_fifo_ratio"] == 100.0


def test_fifo_catches_up_with_full_window():
    convs = _conv()
    ultra, medium = _ultra()
    # A full FIFO window includes the earlier assistant answer, so both arms
    # carry the facts -> both bucket, strata >= FIFO ratio still 100%.
    report = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                        fifo_budget=100000)
    m = report["metrics"]
    assert m["both_sufficient"] == 1
    assert m["hive_only"] == 0
    assert m["hive_answer_recall"] == 100.0
    assert m["fifo_answer_recall"] == 100.0


def test_zero_recall_when_backend_never_states_facts():
    convs = _conv()
    ultra, medium = _ultra()
    report = run_paired(convs, _DumbBackend(), ultra, medium, fifo_budget=1)
    m = report["metrics"]
    assert m["turns_compared"] == 1
    assert m["neither_sufficient"] == 1
    assert m["hive_answer_recall"] == 0.0
    assert m["fifo_answer_recall"] == 0.0
    assert m["hive_ge_fifo_ratio"] == 0.0
    assert m["strict_hive_only_ratio"] == 0.0


def test_first_mention_excluded():
    convs = [{
        "conversation_id": "c2",
        "profile": "code",
        "turns": [
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
            {"role": "assistant",
             "content": "Use JWT tokens with rotation and short expiry."},
        ],
    }]
    ultra, medium = _ultra()
    report = run_paired(convs, _DumbBackend(), ultra, medium)
    assert report["turns_compared"] == 0
    assert report["first_mention_excluded"] == 1
    assert report["turns"] == []


def test_main_mock_writes_report(tmp_path, capsys):
    convs = _conv()
    data_dir = tmp_path / "convs"
    data_dir.mkdir()
    (data_dir / "c1.json").write_text(json.dumps(convs[0]), encoding="utf-8")
    out = tmp_path / "report.json"
    from experiments.paired_ab import main

    code = main(["--mock", "--conversations", str(data_dir),
                 "--output", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["metrics"]["turns_compared"] == 1
    assert doc["metrics"]["hive_answer_recall"] == 0.0  # mock replies carry no facts
    assert "answer recall" in printed


def test_fixture_replay_store_is_symmetric():
    """The default (fixture-replay) store keeps both arms' histories
    identical, so the comparison isolates selection. live_store=True starves
    the strata whenever the model never stated the canonical facts — exactly
    the asymmetry found in the first prose-horizon evidence run."""
    convs = [{
        "conversation_id": "c3",
        "profile": "code",
        "turns": [
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
            {"role": "assistant",
             "content": "Access tokens rotate every fifteen minutes."},
            {"role": "user", "content": "How often do access tokens rotate?"},
            {"role": "assistant",
             "content": "Rotation happens every fifteen minutes without exception."},
            {"role": "user", "content": "How often do access tokens rotate?"},
        ],
    }]
    ultra, medium = _ultra()
    fair = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                      fifo_budget=100000)
    m = fair["metrics"]
    assert m["turns_compared"] == 2
    # turn 2: the canonical answer (a2) is not in any prior history -> neither
    # arm can carry the facts; turn 3: both arms have a2 available
    assert m["neither_sufficient"] == 1
    assert m["fifo_only"] == 0
    assert m["both_sufficient"] == 1
    assert all(r["ctx_hive_suff"] for r in fair["turns"][1:])

    live = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                      fifo_budget=100000, live_store=True)
    ml = live["metrics"]
    # asymmetric mode: the strata store never saw a2 (the model didn't state it),
    # so on turn 3 only FIFO's context carries the facts
    assert ml["fifo_only"] == 1
    assert ml["both_sufficient"] == 0


def test_main_missing_corpus(tmp_path):
    from experiments.paired_ab import main

    assert main(["--mock", "--conversations", str(tmp_path / "none")]) == 2


def test_checkpoint_and_resume(tmp_path):
    convs = _conv()
    ultra, medium = _ultra()
    ckpt = tmp_path / "ckpt.json"
    report = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                        fifo_budget=1, checkpoint_path=ckpt, checkpoint_every=1)
    assert report["metrics"]["turns_compared"] == 1
    saved = json.loads(ckpt.read_text(encoding="utf-8"))
    # fidelity_* is computed after the loop, so it is absent from checkpoints
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("fidelity")}
    assert [strip(r) for r in saved["rows"]] == [strip(r) for r in report["turns"]]

    # Resume with an empty-ish run: turn_index already covers the only turn,
    # so no new rows are produced and the prior rows are preserved.
    resumed = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                         fifo_budget=1, checkpoint_path=tmp_path / "ckpt2.json",
                         checkpoint_every=1, resume=saved)
    assert resumed["turns"] == report["turns"]
    assert resumed["metrics"]["turns_compared"] == 1

    # A resume mid-conversation skips completed turns without dropping rows.
    saved["conv_index"] = 0
    saved["turn_index"] = 2
    saved["prior"] = []
    saved["store"] = {"chunks": {}, "turn_counts": {}}
    saved["rows"] = []
    resumed2 = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                          fifo_budget=1, checkpoint_path=tmp_path / "ckpt3.json",
                          checkpoint_every=1, resume=saved)
    assert resumed2["turns"] == []
    assert resumed2["turns_compared"] == 0