"""``hive serve`` — the CLI owns the lifecycle and stays engine-blind."""

import json

import harness.serve as serve


class _Models:
    last_kwargs = None

    def __init__(self, **kwargs):
        _Models.last_kwargs = kwargs


class _Manager:
    instances = []

    def __init__(self, models, *, stacks_root=None):
        self.models = models
        self.stacks_root = stacks_root
        self.applied = None
        self.unloads = 0
        _Manager.instances.append(self)

    def apply(self, stack):
        self.applied = stack
        return {"ok": True, "tiers": [{"role": "face", "port": 8080, "key": "k"}]}

    def status(self):
        return {
            "ok": True,
            "stack": self.applied.name if self.applied else None,
            "tiers": [
                {"role": "face", "port": 8080, "base_url": None, "ctx": 8192, "backend": "rocm"},
                {"role": "worker", "port": 8081, "base_url": "http://127.0.0.1:9000/v1",
                 "ctx": 32768, "backend": "strata"},
            ],
            "warnings": [],
        }

    def unload(self):
        self.unloads += 1
        return {"ok": True, "unloaded": []}


class _RefusingManager(_Manager):
    def apply(self, stack):
        raise RuntimeError("stack 't' refused: over budget")


def _stack_file(tmp_path, name="t"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "name": name,
        "version": 1,
        "tiers": [{"role": "face", "repo": "org/model", "file": "m.gguf"}],
    }), encoding="utf-8")
    return path


def test_serve_once_applies_and_unloads(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serve, "LlamaServerManager", _Models)
    monkeypatch.setattr(serve, "StackManager", _Manager)
    _Manager.instances.clear()

    rc = serve.main([str(_stack_file(tmp_path)), "--once"])

    out = capsys.readouterr().out
    assert rc == 0
    manager = _Manager.instances[-1]
    assert manager.applied.name == "t"
    assert manager.unloads == 1
    # a spawned tier is addressed by its port, an endpoint tier by base_url
    assert "http://127.0.0.1:8080/v1" in out
    assert "http://127.0.0.1:9000/v1" in out


def test_serve_list(tmp_path, capsys):
    _stack_file(tmp_path)
    rc = serve.main(["--list", "--stacks-dir", str(tmp_path)])
    assert rc == 0
    assert "t" in capsys.readouterr().out


def test_serve_missing_stack_is_usage_error(tmp_path, capsys):
    rc = serve.main([str(tmp_path / "nope.json"), "--once"])
    assert rc == 2
    assert "no stack file" in capsys.readouterr().err


def test_serve_apply_failure_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serve, "LlamaServerManager", _Models)
    monkeypatch.setattr(serve, "StackManager", _RefusingManager)
    rc = serve.main([str(_stack_file(tmp_path)), "--once"])
    assert rc == 1
    assert "refused" in capsys.readouterr().err


_DEVICES = [
    {"index": 0, "bdf": "0000:03:00.0", "memory_gb": 20.0, "display": True},
    {"index": 1, "bdf": "0000:07:00.0", "memory_gb": 20.0, "display": False},
]


def test_model_two_cards_generates_pin_ts_backend(monkeypatch):
    monkeypatch.setattr(serve, "_detect_devices", lambda: _DEVICES)
    monkeypatch.setattr(serve, "backend_binary",
                        lambda name: "/fake/llama-server" if name == "rocm" else None)
    monkeypatch.setattr(serve, "LlamaServerManager", _Models)
    monkeypatch.setattr(serve, "StackManager", _Manager)
    _Manager.instances.clear()

    rc = serve.main(["--model", "org/m", "--file", "m.gguf", "--cards", "2", "--once"])

    assert rc == 0
    tier = _Manager.instances[-1].applied.tiers[0]
    assert tier.repo == "org/m" and tier.file == "m.gguf"
    assert tier.pin == "0,1"
    assert tier.ts == "1,1"          # equal cards -> equal split
    assert tier.backend == "rocm"    # the installed backend wins
    assert tier.ngl == 99


def test_model_repo_colon_file_form(monkeypatch):
    monkeypatch.setattr(serve, "_detect_devices", lambda: _DEVICES)
    monkeypatch.setattr(serve, "backend_binary", lambda name: None)
    monkeypatch.setattr(serve, "LlamaServerManager", _Models)
    monkeypatch.setattr(serve, "StackManager", _Manager)
    _Manager.instances.clear()

    rc = serve.main(["--model", "org/m:model.gguf", "--cards", "1", "--once"])

    assert rc == 0
    tier = _Manager.instances[-1].applied.tiers[0]
    assert tier.file == "model.gguf"
    assert tier.pin == "0"
    assert tier.ts is None           # one card: no tensor split


def test_model_more_cards_than_detected(monkeypatch, capsys):
    monkeypatch.setattr(serve, "_detect_devices", lambda: _DEVICES[:1])
    rc = serve.main(["--model", "org/m", "--file", "m.gguf", "--cards", "2", "--once"])
    assert rc == 2
    assert "only 1 GPU" in capsys.readouterr().err


def test_model_no_gpu_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(serve, "_detect_devices", lambda: [])
    monkeypatch.setattr(serve, "backend_binary",
                        lambda name: "/fake/llama-server" if name == "cpu" else None)
    monkeypatch.setattr(serve, "LlamaServerManager", _Models)
    monkeypatch.setattr(serve, "StackManager", _Manager)
    _Manager.instances.clear()

    rc = serve.main(["--model", "org/m", "--file", "m.gguf", "--once"])

    assert rc == 0
    tier = _Manager.instances[-1].applied.tiers[0]
    assert tier.backend == "cpu"
    assert tier.pin is None


def test_model_needs_a_file(capsys):
    rc = serve.main(["--model", "org/m", "--once"])
    assert rc == 2
    assert "needs the GGUF file" in capsys.readouterr().err
