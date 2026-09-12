"""Unit tests for experiments.launcher (Tkinter run-configurator)."""

from experiments.launcher import build_argv


def _defaults(**over):
    v = {
        "mode": "live",
        "model": "",
        "no_thinking": True,
        "conversations": "hivebench/tests/fixtures/generated",
        "max_convs": "3",
        "max_turns": "10",
        "max_tokens": "",
        "confidence": "off",
        "checkpoint_every": "5",
        "protocol": False,
        "baselines": False,
        "resume": "",
    }
    v.update(over)
    return v


def test_build_argv_mock_small():
    v = _defaults(mode="mock", max_convs="2", max_turns="3")
    assert build_argv(v) == [
        "--mock",
        "--no-thinking",
        "--conversations", "hivebench/tests/fixtures/generated",
        "--max-convs", "2",
        "--max-turns", "3",
        "--confidence", "off",
        "--checkpoint-every", "5",
    ]


def test_build_argv_no_thinking_omitted_when_off():
    v = _defaults(no_thinking=False)
    assert "--no-thinking" not in build_argv(v)


def test_build_argv_live_full():
    v = _defaults(model="qwen3.6-35b-a3b-apex-mtp", max_tokens="256",
                  protocol=True, baselines=True, resume="runs/20260821_164839")
    argv = build_argv(v)
    assert argv[0] == "--live"
    assert "--model" in argv and "qwen3.6-35b-a3b-apex-mtp" in argv
    assert "--max-tokens" in argv and "256" in argv
    assert "--protocol" in argv
    assert "--baselines" in argv
    assert "--resume" in argv and "runs/20260821_164839" in argv


def test_build_argv_omits_blank_optionals():
    v = _defaults(max_tokens="", checkpoint_every="", resume="", model="")
    argv = build_argv(v)
    assert "--max-tokens" not in argv
    assert "--checkpoint-every" not in argv
    assert "--resume" not in argv
    assert "--model" not in argv
    assert "--max-convs" in argv  # non-blank values still included