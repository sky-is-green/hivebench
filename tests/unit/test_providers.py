"""Unit tests for the provider config layer (HARNESS-SPEC §4)."""

import argparse
import json

import pytest

from backend.openai_compat import OpenAICompatBackend
from backend.providers import (
    MASK,
    Provider,
    ProviderRegistry,
    apply_provider_overrides,
    backend_kwargs,
    load_registry,
    redact_provider,
    save_registry,
)


class CapturingTransport:
    def __init__(self, payload=None):
        self.payload = payload or {"choices": [{"message": {"content": "ok"}}]}
        self.posts = []
        self.gets = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json, headers))
        return _Resp(self.payload)

    def get(self, url, headers=None, timeout=None):
        self.gets.append((url, headers))
        return _Resp({"data": [{"id": "model-b"}, {"id": "model-a"}]})


class _Resp:
    ok = True
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


# ---------------------------------------------------------------------------
# Provider / registry
# ---------------------------------------------------------------------------
def test_provider_from_dict_validates():
    with pytest.raises(ValueError):
        Provider.from_dict({"base_url": "http://x"})
    with pytest.raises(ValueError):
        Provider.from_dict({"name": "n"})
    p = Provider.from_dict({
        "name": "ds", "base_url": "https://api.deepseek.com",
        "api_key": "sk-1", "model": "deepseek-chat",
        "headers": {"X-Title": 123},
    })
    assert p.extra_headers == {"X-Title": "123"}


def test_registry_roundtrip_preserves_fields(tmp_path):
    reg = ProviderRegistry(default="b")
    reg.providers = [
        Provider(name="a", base_url="http://localhost:1234", api_key="lm-studio"),
        Provider(name="b", base_url="https://api.deepseek.com", api_key="sk-2",
                 model="deepseek-chat", extra_headers={"X-T": "v"}),
    ]
    path = tmp_path / "providers.local.json"
    save_registry(reg, path)
    loaded = load_registry(path)
    assert loaded.default == "b"
    assert [p.name for p in loaded.providers] == ["a", "b"]
    assert loaded.providers[1].extra_headers == {"X-T": "v"}
    # the file itself holds the real key (it stays local + gitignored)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["providers"][1]["api_key"] == "sk-2"


def test_load_registry_missing_file_is_empty(tmp_path):
    reg = load_registry(tmp_path / "nope.json")
    assert reg.providers == []
    with pytest.raises(LookupError):
        reg.resolve(None)


def test_resolve_default_name_and_case_insensitive():
    reg = ProviderRegistry(providers=[
        Provider(name="LMStudio", base_url="http://localhost:1234"),
        Provider(name="deepseek", base_url="https://api.deepseek.com"),
    ])
    # resolve() deep-copies to apply resolve_auto_base_url(), so identity is
    # not preserved; equality is the contract.
    assert reg.resolve(None) == reg.providers[0]  # no default -> first
    reg.default = "DeepSeek"
    assert reg.resolve(None).name == "deepseek"
    assert reg.resolve("lmstudio").name == "LMStudio"
    with pytest.raises(LookupError) as exc:
        reg.resolve("groq")
    assert "groq" in str(exc.value) and "deepseek" in str(exc.value)


def test_redact_provider_masks_only_the_key():
    src = {"name": "ds", "base_url": "https://x", "api_key": "sk-secret"}
    out = redact_provider(src)
    assert out["api_key"] == MASK
    assert src["api_key"] == "sk-secret"  # untouched
    assert redact_provider({**src, "api_key": ""})["api_key"] == ""
    reg = ProviderRegistry(providers=[
        Provider(name="n", base_url="u"), Provider(name="k", base_url="v",
                                                   api_key="sk-9"),
    ])
    reds = reg.redacted()
    assert reds[0]["api_key"] == ""
    assert reds[1]["api_key"] == MASK
    assert reg.providers[1].api_key == "sk-9"  # registry itself keeps the key


def test_backend_kwargs_defaults_and_copies():
    kw = backend_kwargs(Provider(name="lm", base_url="localhost:1234"))
    assert kw["api_key"] == "lm-studio"
    assert kw["extra_headers"] == {}
    headers = {"X-A": "1"}
    kw2 = backend_kwargs(Provider(name="x", base_url="u", api_key="k",
                                  extra_headers=headers))
    headers["X-B"] = "2"
    assert kw2["extra_headers"] == {"X-A": "1"}


def test_apply_provider_overrides_respects_explicit_flags():
    defaults = {"base_url": "http://localhost:1234", "model": ""}
    prov = Provider(name="ds", base_url="https://api.deepseek.com",
                    model="deepseek-chat")
    args = argparse.Namespace(base_url="http://localhost:1234", model="")
    apply_provider_overrides(defaults, args, prov)
    assert args.base_url == "https://api.deepseek.com"
    assert args.model == "deepseek-chat"
    args2 = argparse.Namespace(base_url="http://explicit:1", model="")
    apply_provider_overrides(defaults, args2, prov)
    assert args2.base_url == "http://explicit:1"  # explicit flag wins
    assert args2.model == "deepseek-chat"


# ---------------------------------------------------------------------------
# OpenAICompatBackend integration
# ---------------------------------------------------------------------------
def test_extra_headers_merged_into_requests():
    t = CapturingTransport()
    backend = OpenAICompatBackend(
        base_url="localhost:1234", model="m", transport=t,
        api_key="sk-live", extra_headers={"X-Title": "strata"},
    )
    backend.generate("ctx", "q")
    _url, _payload, headers = t.posts[0]
    assert headers["Authorization"] == "Bearer sk-live"
    assert headers["X-Title"] == "strata"
    # the provider headers also ride along on GET /v1/models (health, listing)
    backend.health()
    _gurl, gheaders = t.gets[0]
    assert gheaders["X-Title"] == "strata"


def test_models_lists_sorted_ids():
    t = CapturingTransport()
    backend = OpenAICompatBackend(base_url="localhost", model="", transport=t)
    assert backend.models() == ["model-a", "model-b"]
    assert t.gets[0][0].endswith("/v1/models")
