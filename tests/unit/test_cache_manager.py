"""Unit tests for backend.cache_manager (S3.2)."""

from backend.cache_manager import KVCacheManager
from backend.lmstudio import LMStudioBackend
from backend.vllm import VLLMBackend


class _T:
    def post(self, url, json=None, headers=None, timeout=None):
        class R:
            ok = True

            def json(self):
                return {"choices": [{"message": {"content": "x"}}]}

            def raise_for_status(self):
                pass

        return R()

    def get(self, url, headers=None, timeout=None):
        class R:
            ok = True

            def json(self):
                return {"data": []}

        return R()


def test_vllm_surgical_mode():
    manager = KVCacheManager(VLLMBackend(transport=_T()))
    assert manager.supports_surgical_edits is True
    result = manager.update_cache("assembled context", persistent_prefix="PIN")
    assert result["mode"] == "surgical"
    assert result["plan"]["pinned"] == ["PIN"]
    assert result["plan"]["dynamic"] == ["assembled context"]


def test_lmstudio_prefix_caching_mode():
    backend = LMStudioBackend(transport=_T())
    manager = KVCacheManager(backend)
    assert manager.supports_surgical_edits is False

    result = manager.update_cache("assembled context", persistent_prefix="PIN")
    assert result["mode"] == "prefix_caching"
    assert result["prefix_stable"] is True
    assert result["cache_invalidated"] is False
    # the backend is configured to send the pinned prefix verbatim first
    assert backend.pinned_prefix == "PIN"


def test_lmstudio_prefix_change_invalidates_cache():
    backend = LMStudioBackend(transport=_T())
    manager = KVCacheManager(backend)

    first = manager.update_cache("ctx", persistent_prefix="RULES v1")
    assert first["prefix_stable"] is True

    second = manager.update_cache("ctx", persistent_prefix="RULES v2")
    assert second["cache_invalidated"] is True
    assert second["prefix_stable"] is False

    third = manager.update_cache("ctx", persistent_prefix="RULES v2")
    assert third["prefix_stable"] is True  # stable again after the change


def test_side_by_side_modes():
    managers = [
        KVCacheManager(VLLMBackend(transport=_T())),
        KVCacheManager(LMStudioBackend(transport=_T())),
    ]
    modes = {m.update_cache("ctx")["mode"] for m in managers}
    assert modes == {"surgical", "prefix_caching"}