"""token_growth — per-turn context-size series from run bundles."""

import json

from experiments.token_growth import analyze, conversation_series


def _report(convs):
    return {"conversations": [{"conversation_id": f"c{i}", "turns": t}
                              for i, t in enumerate(convs)]}


def _conv(n_turns, reply_chars=2000):
    turns = []
    for turn in range(1, n_turns + 1):
        turns.append({
            "turn": turn,
            "query": f"question number {turn} about the api gateway?",
            "reply": "x" * reply_chars,
            "token_count": 1200 if turn > 1 else 0,  # strata: flat after warm-up
        })
    return turns


def test_raw_grows_while_hive_stays_flat(tmp_path):
    doc = _report([_conv(12)])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_report.json").write_text(json.dumps(doc), encoding="utf-8")

    analysis = analyze([run_dir], fifo_budget=4000)
    buckets = {b["turns"]: b for b in analysis["buckets"]}

    # raw history grows linearly; strata stays at its flat assembled size
    assert buckets["6-10"]["raw_median"] >= 3 * buckets["1"]["raw_median"]
    late = [b for b in analysis["buckets"] if b["turns"] == "6-10"][0]
    assert late["hive_median"] == 1200
    assert late["raw_median"] > late["hive_median"] * 1.5
    # FIFO caps at the window budget
    for b in analysis["buckets"]:
        if b["turns"] != "1":
            assert b["fifo_median"] <= 4000


def test_series_shape_and_first_turn(tmp_path):
    pts = conversation_series(_conv(3), fifo_budget=4000)
    assert [p["turn"] for p in pts] == [1, 2, 3]
    assert pts[0]["strata"] == 0  # empty store on the first turn
    # raw is non-decreasing across the session
    raws = [p["raw"] for p in pts]
    assert raws == sorted(raws)


def test_analyze_multiple_runs(tmp_path):
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "run_report.json").write_text(json.dumps(_report([_conv(4)])),
                                           encoding="utf-8")
    analysis = analyze([tmp_path / "a", tmp_path / "b"], fifo_budget=4000)
    assert analysis["turns_analyzed"] == 8
    assert len(analysis["runs"]) == 2