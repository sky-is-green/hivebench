"""Integration: P1-P10 protocol driver runs (mock mode)."""

import json

from backend.lmstudio import LMStudioBackend
from cortex.baselines.runner import load_conversations
from cortex.e2e import FakeUltraSmall, MockTransport
from experiments.run_p1_p10 import PredictionSuite, _load_labels
from auditor.auditor import Auditor
from sieve.medium import MediumDrone


def _suite():
    convs = load_conversations("hivebench/tests/fixtures/generated")
    labels = _load_labels(convs)
    backend = LMStudioBackend(base_url="localhost", model="m", transport=MockTransport())
    auditor = Auditor(
        generate_fn=lambda p: json.dumps({"sufficient": True, "used_pieces": [], "missing": [], "score": 4})
    )
    return PredictionSuite(
        backend, FakeUltraSmall(), MediumDrone(score_pair_fn=lambda q, c: 0.5),
        convs, labels, auditor, live=False,
    )


def test_protocol_runs_all_ten_predictions():
    results = _suite().run()
    assert len(results) == 11
    assert [r.id for r in results] == [f"P{i}" for i in range(1, 12)]
    for r in results:
        assert r.title
        assert r.status in ("PASS", "FAIL", "SKIP", "REPORT")


def test_protocol_predictions_have_evidence_or_note():
    for r in _suite().run():
        assert r.evidence or r.note


def test_protocol_stable_ids():
    ids = {r.id for r in _suite().run()}
    assert ids == {f"P{i}" for i in range(1, 12)}


def test_p11_comb_return_protocol_pass():
    """P11 must report PASS on the return corpus with the real drone: the comb
    resurrects topic-return facts under budget pressure (100% vs 20% no-comb),
    does not regress the full replay (100% vs 100%), does not crowd non-return
    turns, and beats keep-last-N.

    Regression-locks the 2026-08-24 comb work: selection-as-curation (the
    assembler marks selected chunks, so comb_relevant_only archives what the
    strata once judged relevant), the query-echo gate (template-sibling question
    chunks must not keep the gate closed), and the boost-shifted 0.85 gate
    calibration."""
    import json

    from backend.lmstudio import LMStudioBackend
    from cortex.baselines.runner import load_conversations
    from cortex.e2e import MockTransport
    from experiments.run_p1_p10 import PredictionSuite, _load_labels
    from auditor.auditor import Auditor
    from sieve.medium import MediumDrone
    from sieve.ultra_small import UltraSmallDrone

    from tests.fixtures.synthetic_conversations.generate import (
        RETURN_SEED,
        generate_return,
    )

    if not load_conversations("hivebench/tests/fixtures/generated_return"):
        generate_return("hivebench/tests/fixtures/generated_return", seed=RETURN_SEED)

    ultra = UltraSmallDrone(confidence_mode="off")
    convs = load_conversations("hivebench/tests/fixtures/generated")
    labels = _load_labels(convs)
    suite = PredictionSuite(
        LMStudioBackend(base_url="localhost", model="m", transport=MockTransport()),
        ultra, MediumDrone(score_pair_fn=lambda q, c: 0.5),
        [], labels, Auditor(generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": [], "missing": [], "score": 4})),
        live=False,
    )
    r = suite.p11()
    assert r.status == "PASS", f"P11 not PASS: {r.status} — {r.note}"
    e = r.evidence
    assert e["pressure_replay"]["comb_on_return_recall"] >= 90.0
    assert e["pressure_replay"]["comb_on_return_recall"] >= e["pressure_replay"]["comb_off_return_recall"]
    assert e["full_replay"]["comb_on_return_recall"] >= e["full_replay"]["comb_off_return_recall"] - 5.0
    assert e["full_replay"]["comb_on_non_return_recall"] == e["full_replay"]["comb_off_non_return_recall"]
    assert e["clauses"]["beats_keep_last_n"]


def test_p3_long_conversations_close_sufficiency():
    """P3 must PASS on full-length long conversations with the real drone: on
    turns where the answer's facts were actually in history, strata >= FIFO on
    >=80% of turns (paper's paired-A/B criterion, ties counted, first-mention
    excluded)."""
    import json

    from backend.lmstudio import LMStudioBackend
    from cortex.baselines.runner import load_conversations
    from cortex.e2e import MockTransport
    from experiments.run_p1_p10 import PredictionSuite, _load_labels
    from auditor.auditor import Auditor
    from sieve.medium import MediumDrone
    from sieve.ultra_small import UltraSmallDrone

    ultra = UltraSmallDrone(confidence_mode="off")
    convs = [c for c in load_conversations("hivebench/tests/fixtures/generated")
             if c.get("profile") == "long"]
    labels = _load_labels(convs)
    suite = PredictionSuite(
        LMStudioBackend(base_url="localhost", model="m", transport=MockTransport()),
        ultra, MediumDrone(score_pair_fn=lambda q, c: 0.5),
        convs, labels,
        Auditor(generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": [], "missing": [], "score": 4})),
        live=False,
    )
    r = suite.p3()
    assert r.status == "PASS", f"P3 should PASS on long conversations: {r.evidence}"
    assert r.evidence["hive_ge_fifo_ratio"] >= 0.80
    assert r.evidence["hive_only"] > r.evidence["fifo_only"]


def test_p4_horizon_corpus_separates_domains():
    """P4 must now report the long-horizon sweep (code vs prose) with the m90
    verdict metric: code tolerates more decay than prose (gap > 0.2 band).

    Regression-locks the 2026-08-23 horizon-corpus work: the old corpora
    measured FLAT curves (relevant facts were always recent); the horizon
    corpus ages the facts (establish -> recap at age == E) so the multiplier
    governs retrieval, and the fixed 1000-token budget isolates decay from the
    adaptive budget's high-relevance feedback. The verdict PASS here is the
    corpus's designed, reproducible outcome."""
    import json

    from backend.lmstudio import LMStudioBackend
    from cortex.baselines.runner import load_conversations
    from cortex.e2e import MockTransport
    from experiments.run_p1_p10 import PredictionSuite, _load_labels
    from auditor.auditor import Auditor
    from sieve.medium import MediumDrone
    from sieve.ultra_small import UltraSmallDrone

    from tests.fixtures.synthetic_conversations.generate import generate_horizon

    for horizon_dir, seed, domain in (
        ("hivebench/tests/fixtures/generated_horizon", 4041, "code"),
        ("hivebench/tests/fixtures/generated_prose_horizon", 5051, "prose"),
    ):
        if not load_conversations(horizon_dir):
            generate_horizon(horizon_dir, seed=seed, domain=domain)

    ultra = UltraSmallDrone(confidence_mode="off")
    convs = load_conversations("hivebench/tests/fixtures/generated")
    labels = _load_labels(convs)
    suite = PredictionSuite(
        LMStudioBackend(base_url="localhost", model="m", transport=MockTransport()),
        ultra, MediumDrone(score_pair_fn=lambda q, c: 0.5),
        convs, labels,
        Auditor(generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": [], "missing": [], "score": 4})),
        live=False,
    )
    r = suite.p4()
    assert r.status == "PASS", f"P4 should PASS on the horizon corpus: {r.note}"
    assert "code" in r.evidence and "prose" in r.evidence, r.evidence.keys()
    for dom in ("code", "prose"):
        assert r.evidence[dom]["flat_across_multipliers"] is False
        assert r.evidence[dom]["m90"] is not None
        assert r.evidence[dom]["best_recall"] > 0.0
    assert r.evidence["m90_gap"] > 0.2
    assert r.evidence["code_vs_prose_differ"] is True
    # predicted direction: code (clustered references, young facts) tolerates
    # more decay than prose (long-range references, stale facts)
    assert r.evidence["code"]["m90"] > r.evidence["prose"]["m90"]


def test_horizon_corpus_age_structure():
    """The long-horizon corpus must age its facts exactly: every recap query
    re-asks an established fact at age == establish length E (youngest prior
    copy), and the recap answers' fact terms must be present in prior history
    (retrievable). Per-domain E ranges encode the P4 premise (code 10/20, no
    stale; prose 24/32, stale)."""
    from cortex.baselines.runner import load_conversations
    from experiments.retrieval_diagnostic import _answer_fact_terms, _content_terms

    expected_e = {"code": {10, 20}, "prose": {24, 32}}
    for domain, path in (
        ("code", "hivebench/tests/fixtures/generated_horizon"),
        ("prose", "hivebench/tests/fixtures/generated_prose_horizon"),
    ):
        convs = load_conversations(path)
        assert convs, f"{domain} horizon corpus missing (run --horizon)"
        ages = set()
        for c in convs:
            assert c.get("profile") == "longhorizon"
            assert c.get("domain") == domain
            E = c["establish_turns"]
            assert E in expected_e[domain], (domain, E)
            turns = c["turns"]
            ans_map = {}
            for i, t in enumerate(turns):
                if t["role"] == "user" and i + 1 < len(turns):
                    ans_map.setdefault(t["content"], turns[i + 1]["content"])
            user_turn = 0
            recap_count = 0
            for i, t in enumerate(turns):
                if t["role"] != "user":
                    continue
                user_turn += 1
                if user_turn <= E:
                    continue
                recap_count += 1
                q = t["content"]
                facts = _answer_fact_terms(q, ans_map[q])
                prior = " ".join(x["content"] for x in turns[:i])
                assert facts, f"{c['conversation_id']} turn {user_turn}: empty facts"
                assert facts <= _content_terms(prior), (
                    f"{c['conversation_id']} turn {user_turn}: facts not in history")
                est_turn = None
                u = 0
                for j, x in enumerate(turns[:i]):
                    if x["role"] == "user":
                        u += 1
                    if x["role"] == "assistant" and facts <= _content_terms(x["content"]):
                        est_turn = u
                assert est_turn is not None
                assert user_turn - est_turn == E, (
                    f"{c['conversation_id']} turn {user_turn}: age "
                    f"{user_turn - est_turn} != E={E}")
                ages.add(user_turn - est_turn)
            assert recap_count == c["recap_turns"]
        assert ages <= expected_e[domain]


def test_p9_duplicate_pairs_merge():
    """Every engineered dense<->verbose pair in the P9 corpus must exceed the
    dedup threshold (cosine > 0.92) with the default drone — otherwise the A/B
    measures nothing (the policies would keep the same copies). Regression-
    locks the corpus construction (2026-08-23: measured 24/24 pairs, min
    0.9417)."""
    import numpy as np

    from cortex.baselines.runner import load_conversations
    from sieve.ultra_small import UltraSmallDrone
    from tests.fixtures.synthetic_conversations.generate import P9_SEED, generate_p9

    from experiments.p9_densest_duplicate import DUPLICATE_THRESHOLD

    convs = load_conversations("hivebench/tests/fixtures/generated_p9")
    if not convs:
        generate_p9("hivebench/tests/fixtures/generated_p9", seed=P9_SEED)
        convs = load_conversations("hivebench/tests/fixtures/generated_p9")

    ultra = UltraSmallDrone(confidence_mode="off")
    pairs = 0
    below = []
    for conv in convs:
        turns = conv["turns"]
        for i in range(0, 4 * conv["aspects_per_conv"], 4):
            a1 = turns[i + 1]["content"]
            a2 = turns[i + 3]["content"]
            v1 = np.asarray(ultra.embed(a1), dtype=float)
            v2 = np.asarray(ultra.embed(a2), dtype=float)
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            pairs += 1
            if cos <= DUPLICATE_THRESHOLD:
                below.append((conv["conversation_id"], round(cos, 4)))
    assert pairs == 24, pairs
    assert not below, f"pairs below the 0.92 dedup threshold: {below}"


def test_p9_densest_beats_recency():
    """The P9 A/B must PASS with the locked corpus: on the informative
    (recency_favors_verbose) turns, densest retention beats recency retention
    on sufficiency-per-1k-tokens, and the control shows no effect."""
    import json

    from backend.lmstudio import LMStudioBackend
    from cortex.baselines.runner import load_conversations
    from cortex.e2e import MockTransport
    from experiments.run_p1_p10 import PredictionSuite, _load_labels
    from auditor.auditor import Auditor
    from sieve.medium import MediumDrone
    from sieve.ultra_small import UltraSmallDrone
    from tests.fixtures.synthetic_conversations.generate import P9_SEED, generate_p9

    from experiments.p9_densest_duplicate import DUPLICATE_THRESHOLD

    convs = load_conversations("hivebench/tests/fixtures/generated_p9")
    if not convs:
        generate_p9("hivebench/tests/fixtures/generated_p9", seed=P9_SEED)
        convs = load_conversations("hivebench/tests/fixtures/generated_p9")

    ultra = UltraSmallDrone(confidence_mode="off")
    base = load_conversations("hivebench/tests/fixtures/generated")
    labels = _load_labels(base)
    suite = PredictionSuite(
        LMStudioBackend(base_url="localhost", model="m", transport=MockTransport()),
        ultra, MediumDrone(score_pair_fn=lambda q, c: 0.5),
        base, labels,
        Auditor(generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": [], "missing": [], "score": 4})),
        live=False,
    )
    r = suite.p9()
    assert r.status == "PASS", f"P9 should PASS on the engineered-duplicate corpus: {r.note}"
    assert r.evidence["informative_turns"] == 12
    assert r.evidence["recency_wins"] == 0
    assert r.evidence["densest_per_1k"] > r.evidence["recency_per_1k"]
    # control: both policies keep the dense copy -> no effect
    assert r.evidence["control_densest_per_1k"] == r.evidence["control_recency_per_1k"]


def test_horizon_decision_terms_are_distinctive():
    """Fact terms (decision words) must be absent from every OTHER aspect's
    chunks — aspect names, feature names, query templates, replies — or an
    unrelated selected chunk could false-hit the retrieval check (this was
    measured: "warehouse" leaked via "warehouse replication", "team" via the
    "team communication" feature name, etc.)."""
    from experiments.retrieval_diagnostic import _answer_fact_terms, _content_terms
    from tests.fixtures.synthetic_conversations.generate import (
        HORIZON_TOPICS, PROSE_HORIZON_TOPICS,
        HORIZON_ESTABLISH_USER, HORIZON_ESTABLISH_ASST, HORIZON_ESTABLISH_PROSE_FILLER,
        HORIZON_RECAP_USER_TPL, HORIZON_RECAP_ASST,
    )

    for name, topics in (("code", HORIZON_TOPICS), ("prose", PROSE_HORIZON_TOPICS)):
        facts_of = {}
        all_chunks = {}
        for topic, data in topics.items():
            for aspect, decision in data["decisions"].items():
                q = HORIZON_RECAP_USER_TPL[0].format(
                    feature=data["feature"], aspect=aspect)
                a = HORIZON_RECAP_ASST.format(
                    feature=data["feature"], aspect=aspect, decision=decision)
                facts = _answer_fact_terms(q, a)
                assert facts, f"{name}/{topic}/{aspect}: empty fact terms"
                asst = HORIZON_ESTABLISH_ASST.format(
                    feature=data["feature"], aspect=aspect, decision=decision)
                if name == "prose":
                    asst += HORIZON_ESTABLISH_PROSE_FILLER
                chunks = [asst]
                for tpl in (HORIZON_ESTABLISH_USER, HORIZON_RECAP_ASST):
                    chunks.append(tpl.format(
                        feature=data["feature"], aspect=aspect, decision=decision))
                for tpl in HORIZON_RECAP_USER_TPL:
                    chunks.append(tpl.format(feature=data["feature"], aspect=aspect))
                chunks.append(data["feature"])
                chunks.append(aspect)
                facts_of[(topic, aspect)] = facts
                all_chunks[(topic, aspect)] = chunks
        for (topic, aspect), facts in facts_of.items():
            for other, chunks in all_chunks.items():
                if other == (topic, aspect):
                    continue
                for ch in chunks:
                    assert not (facts & _content_terms(ch)), (
                        f"{name}/{topic}/{aspect} facts leak into "
                        f"{other[0]}/{other[1]} chunk: {ch[:60]!r}")