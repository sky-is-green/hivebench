"""Local llama.cpp server management + Hugging Face model acquisition.

Owns the "load models in our own app" layer (HARNESS-SPEC M4), LM-Studio
style: **multiple models can be loaded at once**, each as its own
``llama-server`` process on its own port, independently unloadable, every
one registered as a provider so chat/agent/benchmarks can target it by name.

- Discovery is **live**: hub search and repo file listings hit the public
  Hugging Face API per request, so newly released models appear immediately —
  nothing model-specific is hardcoded here. Managed downloads run in
  background threads via ``huggingface_hub`` into ``models_dir``.
- ``llama-server --hf-repo/--hf-file`` is passed through verbatim when a
  requested repo/file is not yet local, so day-one releases work even before
  a managed download exists.

Injectable seams (``spawner``, ``prober``, module-level ``_hf_download``)
keep the whole manager unit-testable offline.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

# Quantization labels for GGUF general.file_type, mirroring llama.cpp's
# LLAMA_FTYPE_* values. Unknown values render as ftype-N.
_FILE_TYPE_NAMES: dict[int, str] = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 4: 'Q4_1_SOME_F16', 5: 'Q4_2', 6: 'Q4_3',
    7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M', 13: 'Q3_K_L',
    14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K', 19: 'IQ2_XXS',
    20: 'IQ2_XS', 21: 'Q2_K_S', 22: 'IQ3_XXS', 23: 'IQ1_S', 24: 'IQ4_NL', 25: 'IQ3_S',
    26: 'IQ2_S', 27: 'IQ4_XS', 28: 'IQ1_M', 29: 'BF16', 30: 'Q4_0_4_4', 31: 'Q4_0_4_8',
    32: 'Q4_0_8_8', 33: 'TQ1_0', 34: 'TQ2_0',
    # Prism fork: released Bonsai PQ2_0 files set general.file_type 141 (the
    # ggml tensor type is 142 / QK_PQ2_0); verified on the 1.7B/27B GGUFs.
    141: 'PQ2_0',
}

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_API = "https://huggingface.co/api"

# Header-parse guards. The GGUF reader is best-effort and runs on every
# library poll, so a corrupt or hostile header must cost us the parse, never
# an unbounded allocation or a hang.
_MAX_STRING_LEN = 1 << 20
_MAX_ARRAY_ELEMS = 1 << 24
_MAX_LAYER_TYPES = 4096  # no model has more than a few hundred layers

# LM Studio default models location (Windows) — checked first
def _lmstudio_models_dir() -> Optional[Path]:
    candidates = [
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
        Path(os.environ.get("LOCALAPPDATA", "")) / "LM Studio" / "models",
        Path(os.environ.get("APPDATA", "")) / "LM Studio" / "models",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None

def _default_models_dir() -> Path:
    lm = _lmstudio_models_dir()
    if lm and lm.is_dir():
        return lm
    # Fallback: repo's own models/gguf (created on demand, also used for HF downloads)
    fallback = REPO_ROOT / "models" / "gguf"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _default_binary() -> Path:
    """HARNESS_LLAMA_SERVER override > tools/llama.cpp/llama-server(.exe)."""
    env = os.environ.get("HARNESS_LLAMA_SERVER")
    if env:
        return Path(env)
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    return REPO_ROOT / "tools" / "llama.cpp" / exe


BACKENDS = ("vulkan", "rocm", "cuda", "cpu", "sycl")


def _binary_for_backend(backend: Optional[str]) -> Optional[Path]:
    """Per-backend llama-server binary under tools/backends/<backend>/, when
    present (fetched via tools/fetch_backend.ps1). None when the backend is
    unknown to the layout or its binary has not been fetched."""
    if not backend:
        return None
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    p = (REPO_ROOT / "tools" / "backends" / backend.strip().lower() / exe)
    return p if p.is_file() else None


class OutOfMemoryError(RuntimeError):
    """A model cannot load without exhausting VRAM/RAM.

    Raised before spawn when the estimate clearly exceeds host capacity, or
    when llama-server exits during startup with an allocation failure in its
    log. Actionable, non-fatal: callers report it instead of hanging or dying.
    """


# Substrings that mark an allocation failure in a llama-server log. Kept broad
# so CUDA, HIP, Vulkan and CPU/ggml allocator messages all match.
_OOM_SIGNATURES = (
    "out of memory", "outofmemory", "out_of_memory",
    "hiperroroutofmemory", "hip error: out of memory",
    "cuda_error_out_of_memory", "cuda error: out of memory",
    "failed to allocate", "unable to allocate", "cannot allocate",
    "not enough memory", "insufficient memory",
)

# Models below this size skip the preflight (unit tests use tiny stubs).
_MEMORY_GUARD_MIN_GB = 1.0


def _looks_like_oom(text: str) -> bool:
    low = (text or "").lower()
    return any(sig in low for sig in _OOM_SIGNATURES)


def _read_log_tail(path: Path, limit: int = 8192) -> str:
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def preflight_memory(model_path, *, hardware_probe=None) -> dict:
    """Estimate whether ``model_path`` fits on this host.

    Weights dominate; KV/scratch is estimated as a small fraction. Never
    raises: returns a dict with ``ok`` False only when the estimate clearly
    exceeds total VRAM + available RAM. ``warning`` marks a model that fits
    but must offload layers to RAM (slow).
    """
    path = Path(model_path)
    try:
        model_gb = path.stat().st_size / (1024 ** 3)
    except OSError:
        return {"ok": True, "checked": False, "reason": "unreadable model file"}
    if model_gb < _MEMORY_GUARD_MIN_GB:
        return {"ok": True, "checked": False, "reason": "small model",
                "model_gb": round(model_gb, 3)}
    if hardware_probe is None:
        try:
            from harness.app import _hardware_summary
            hw = _hardware_summary()
        except Exception:  # noqa: BLE001 - best-effort; never block a load
            return {"ok": True, "checked": False,
                    "reason": "hardware probe unavailable",
                    "model_gb": round(model_gb, 2)}
    else:
        hw = hardware_probe()
    devices = hw.get("devices") or []
    combined = float(hw.get("combined_vram_gb") or hw.get("vram_gb") or 0.0)
    visibility = (os.environ.get("HIP_VISIBLE_DEVICES")
                  or os.environ.get("CUDA_VISIBLE_DEVICES"))
    per_card = [float(d.get("memory_gb") or 0.0) for d in devices]
    per_card = [g for g in per_card if g > 0]
    override = os.environ.get("HARNESS_VRAM_GB")
    if override:
        # Authoritative: the caller knows which cards are visible.
        try:
            vram = float(override)
        except ValueError:
            vram = combined
    elif hw.get("vram_free_gb") is not None:
        # Visible free VRAM (other apps' usage already subtracted).
        vram = float(hw["vram_free_gb"])
    elif visibility is not None and per_card:
        # Visibility set but no free accounting: assume the largest single
        # visible card (conservative; HIP indices need not match sysfs order).
        vram = max(per_card)
    else:
        vram = combined
    ram = float(hw.get("available_ram_gb") or 0.0)
    # On AMD, CPU offload is bounded by GTT (GPU-accessible system memory),
    # not the whole of RAM; use the smaller budget.
    gtt_free = None
    if hw.get("gtt_total_gb") is not None:
        gtt_free = max(0.0, float(hw["gtt_total_gb"])
                       - float(hw.get("gtt_used_gb") or 0.0))
    offload_budget = min(ram, gtt_free) if gtt_free is not None else ram
    capacity = vram + offload_budget
    required = model_gb * 1.05 + 1.0  # weights + KV / scratch headroom
    result = {
        "ok": capacity <= 0 or required <= capacity * 0.98,
        "checked": True,
        "model_gb": round(model_gb, 2),
        "required_gb": round(required, 2),
        "capacity_gb": round(capacity, 2),
        "vram_gb": round(vram, 2),
        "ram_gb": round(ram, 2),
        "gtt_free_gb": round(gtt_free, 2) if gtt_free is not None else None,
        "offload": model_gb > vram,
        "visible": visibility,
        "vram_override": override or None,
    }
    if not result["ok"]:
        result["reason"] = (
            f"needs ~{required:.1f} GB but only {capacity:.1f} GB is available "
            f"(VRAM {vram:.1f} + RAM {ram:.1f})")
    elif result["offload"]:
        result["warning"] = (
            f"model is {model_gb:.1f} GB > VRAM {vram:.1f} GB; llama-server "
            "will offload layers to RAM and run slowly")
    return result


def _probe(base_url: str, timeout: float = 5.0):
    """Return the first model id advertised by the server, True if reachable
    but silent about ids, False when unreachable."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
    except requests.RequestException:
        return False
    if not resp.ok:
        return False
    try:
        ids = [m.get("id", "") for m in resp.json().get("data", [])]
    except ValueError:
        return True
    ids = [i for i in ids if i]
    return ids[0] if ids else True


def _hf_download(repo_id: str, filename: str, dest_dir: Path,
                 token: Optional[str] = None) -> Path:
    """Blocking single-file HF download (runs inside a worker thread)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=str(dest_dir),
        token=token or os.environ.get("HF_TOKEN") or None,
    ))


def _read_gguf_metadata(path: Path) -> dict:
    """Best-effort GGUF header parse for library hover metadata.

    Reads the GGUF magic + version + KV section and extracts
    general.architecture, general.file_type (→ quantization label),
    <arch>.context_length, the attention keys ``fit_gpu_layers`` and
    ``attention_kv_estimate`` need (block/head counts, key/value lengths,
    sliding window + pattern) and the ``<arch>.attention.layer_types`` array
    that marks each layer ``full_attention`` / ``sliding_attention`` /
    ``linear_attention``. Missing or unreadable files return {}.
    Mirrors the TypeScript gguf-metadata parser's FILE_TYPE_NAMES.
    """
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return {}
            ver = struct.unpack("<I", fh.read(4))[0]
            if ver not in (2, 3):
                return {}
            pos = fh.tell()
            try:
                tc = struct.unpack("<Q", fh.read(8))[0]
                mc = struct.unpack("<Q", fh.read(8))[0]
                if mc > 10000:
                    raise ValueError("implausible")
            except Exception:
                fh.seek(pos)
                tc = struct.unpack("<I", fh.read(4))[0]
                mc = struct.unpack("<I", fh.read(4))[0]
                if mc > 10000:
                    return {}
            out: dict = {}
            arch: str | None = None
            file_type: int | None = None
            context_candidates: dict[str, int] = {}
            for _ in range(int(mc)):
                try:
                    klen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                except Exception:
                    break
                if klen > 500:
                    break
                try:
                    key = fh.read(int(klen)).decode("utf-8", errors="ignore")
                except Exception:
                    break
                try:
                    ktype = struct.unpack("<I", fh.read(4))[0]
                except Exception:
                    break
                try:
                    if ktype == 0:
                        val = struct.unpack("<B", fh.read(1))[0]
                    elif ktype == 1:
                        val = struct.unpack("<b", fh.read(1))[0]
                    elif ktype == 2:
                        val = struct.unpack("<H", fh.read(2))[0]
                    elif ktype == 3:
                        val = struct.unpack("<h", fh.read(2))[0]
                    elif ktype == 4:
                        val = struct.unpack("<I", fh.read(4))[0]
                    elif ktype == 5:
                        val = struct.unpack("<i", fh.read(4))[0]
                    elif ktype == 6:
                        val = struct.unpack("<f", fh.read(4))[0]
                    elif ktype == 7:
                        val = bool(struct.unpack("<B", fh.read(1))[0])
                    elif ktype == 8:
                        slen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                        if slen > 64 * 1024 * 1024:
                            break
                        val = fh.read(int(slen)).decode("utf-8", errors="ignore")
                    elif ktype == 9:
                        atype = struct.unpack("<I", fh.read(4))[0]
                        alen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                        if alen > _MAX_ARRAY_ELEMS:
                            raise ValueError("implausible array length")
                        if atype == 8:
                            strings: list[str] = []
                            keep = key.endswith(".attention.layer_types")
                            for __ in range(int(alen)):
                                sl = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                                if sl > _MAX_STRING_LEN:
                                    raise ValueError("implausible string length")
                                raw = fh.read(int(sl))
                                if keep and len(strings) < _MAX_LAYER_TYPES:
                                    strings.append(raw.decode("utf-8", errors="ignore"))
                            # Only the layer-type labels carry meaning; every
                            # other array stays a marker so the header stays
                            # cheap to parse.
                            val = strings if keep else f"<array:{alen}>"
                        else:
                            # Non-string arrays carry raw elements, not a
                            # length prefix: skip by element size or the
                            # stream desyncs and every later key is garbage.
                            size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(atype, 1)
                            fh.read(int(alen) * size)
                            val = f"<array:{alen}>"
                    elif ktype == 10:
                        val = struct.unpack("<Q", fh.read(8))[0]
                    elif ktype == 11:
                        val = struct.unpack("<q", fh.read(8))[0]
                    elif ktype == 12:
                        val = struct.unpack("<d", fh.read(8))[0]
                    else:
                        break
                except Exception:
                    break
                if key == "general.architecture" and isinstance(val, str):
                    arch = val
                    out["general.architecture"] = val
                elif key == "general.file_type" and isinstance(val, int):
                    file_type = val
                    out["general.file_type"] = val
                elif key.endswith(".context_length") and isinstance(val, int):
                    prefix = key[: -len(".context_length")]
                    context_candidates[prefix] = int(val)
                    out[key] = val
                elif key in ("general.name", "general.parameter_count"):
                    out[key] = val
                elif isinstance(val, int) and (
                        key.endswith(".block_count")
                        or key.endswith(".embedding_length")
                        or key.endswith(".attention.head_count")
                        or key.endswith(".attention.head_count_kv")
                        or key.endswith(".attention.key_length")
                        or key.endswith(".attention.value_length")
                        or key.endswith(".attention.sliding_window")
                        or key.endswith(".attention.sliding_window_pattern")):
                    out[key] = int(val)
                elif isinstance(val, list) and key.endswith(".attention.layer_types"):
                    out[key] = val
                # Keep reading until the fit-relevant keys are present, not
                # just the hover ones (block_count/head counts often come
                # later). The attention plan needs one more: layer_types
                # (hybrid archs) or sliding_window (Gemma-3 style). A dense
                # arch ships neither, so that case would otherwise scan to the
                # end of the KV section and through the 100k-entry tokenizer
                # vocab arrays on every library poll — hence the tokenizer
                # stop below, which no fit key ever lives behind.
                has_fit = (any(k.endswith(".block_count") for k in out)
                           and any(k.endswith(".attention.head_count_kv") for k in out))
                # The plan needs layer_types (hybrid archs) or the sliding
                # window pair (Gemma-3 style) — the pattern key is what turns
                # a window into a per-layer count, so both must be in.
                has_attention = (
                    any(k.endswith(".attention.layer_types") for k in out)
                    or (any(k.endswith(".attention.sliding_window") for k in out)
                        and any(k.endswith(".attention.sliding_window_pattern") for k in out)))
                if arch is not None and file_type is not None and has_fit and (
                        has_attention or key.startswith("tokenizer.")):
                    break
            if arch is not None:
                out["architecture"] = arch
            if file_type is not None:
                out["quantization"] = _FILE_TYPE_NAMES.get(file_type, f"ftype-{file_type}")
                out["file_type"] = file_type
            if arch is not None and arch in context_candidates:
                out["context_length"] = context_candidates[arch]
            elif context_candidates:
                # fallback to any context_length when arch prefix unknown
                out["context_length"] = next(iter(context_candidates.values()))
            return out
    except Exception:
        return {}


# --- KV cache sizing (T33) -------------------------------------------------
#
# Not every layer carries a cache that grows with the context window. Hybrid
# architectures interleave a few full-attention layers with linear-attention
# (GatedDeltaNet/Mamba) layers, which hold a fixed-size recurrent state, and
# sliding-window (SWA) layers, which are bounded by the window. An estimator
# that prices every layer at the full context size over-counts a 27B-class
# hybrid by 4-8x and then refuses contexts that fit comfortably. See
# LOCAL-STACKS.md section 5 for the measured reference rates.
#
# kv_per_token = kv_layers x kv_heads x (key_len + val_len) x 2 (K and V) x
#                bytes_per_element
#
# Reference rates reproduced by this module (Qwen3.8-27B shape: 64 layers, 16
# of them full attention, 8 KV heads, 128 head dim):
#   f16  -> 64 KiB/token, q8_0 -> 34 KiB/token, 2.1 GiB at 64K context.

# Bytes per stored element for llama.cpp's KV cache types, derived from each
# type's block layout: (block bytes) / (block elements). K-quants use a 256-
# element super-block. Values that do not divide evenly still price correctly
# in aggregate, since llama.cpp stores whole blocks.
_KV_TYPE_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,        # 32 int8 + 1 fp16 scale per 32 elements
    "q5_1": 24.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q6_k": 210.0 / 256.0,
    "q5_k": 176.0 / 256.0,
    "q4_k": 144.0 / 256.0,
    "q3_k": 110.0 / 256.0,
    "q2_k": 84.0 / 256.0,
}

# Fallback for a KV type this build does not know: f16 is the llama.cpp default
# and the largest entry here, so an unknown type is priced conservatively
# (over-estimating memory) rather than silently freeing headroom that is not
# there.
_KV_TYPE_FALLBACK = "f16"

# Layer-type labels llama.cpp writes into <arch>.attention.layer_types.
_LINEAR_ATTENTION_LABELS = frozenset({
    "linear_attention", "linear", "mamba", "recurrent", "recurrent_attention",
    "ssm", "gated_delta_net", "gated_delta", "rwkv", "meson",
})
_SLIDING_ATTENTION_LABELS = frozenset({
    "sliding_attention", "sliding_window", "sliding", "swa",
})

# Architectures whose GGUF writers omit ``<arch>.attention.layer_types`` even
# though the backbone is hybrid.  The layout is structural — one full-attention
# layer every N, the rest GatedDeltaNet/linear — so a per-arch period is enough,
# and an explicit ``layer_types`` array always wins over this table.  Measured
# on the real Qwen3.8-27B GGUFs: 65 blocks (64 + a trailing MTP/nextn block),
# 16 full-attention layers, 64 KiB/token f16 / 34 KiB/token q8_0.
_ARCH_FULL_ATTENTION_PERIOD: dict[str, int] = {
    "qwen35": 4,
    "qwen3_5_moe": 4,
}


def _kv_bytes_per_element(kv_cache_type: Optional[str]) -> float:
    """Bytes per cached element for ``kv_cache_type`` (``f16`` when unknown)."""
    if not kv_cache_type:
        return _KV_TYPE_BYTES_PER_ELEMENT[_KV_TYPE_FALLBACK]
    return _KV_TYPE_BYTES_PER_ELEMENT.get(
        str(kv_cache_type).strip().lower(),
        _KV_TYPE_BYTES_PER_ELEMENT[_KV_TYPE_FALLBACK],
    )


def _meta_first_int(meta: dict, suffix: str) -> Optional[int]:
    """First ``<arch><suffix>`` value in ``meta`` that parses as an int."""
    for key, value in meta.items():
        if not key.endswith(suffix):
            continue
        if isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _meta_layer_types(meta: dict) -> list[str]:
    """The ``<arch>.attention.layer_types`` labels, normalised to lower case.

    A GGUF from an older build (or a truncated parse) may leave the placeholder
    ``"<array:64>"`` string behind, which is not a label list; that yields [].
    """
    for key, value in meta.items():
        if not key.endswith(".attention.layer_types"):
            continue
        if isinstance(value, str):
            return []
        if not isinstance(value, (list, tuple)):
            continue
        return [str(item).strip().lower() for item in value]
    return []


@dataclass(frozen=True)
class KVEstimate:
    """Attention-aware KV cache size for one GGUF at a given KV quantisation.

    Splits the block count into the three layer kinds that matter:

    - ``full_layers`` — carry a KV cache that grows with the context window.
    - ``sliding_layers`` — carry a KV cache bounded by ``sliding_window``.
    - ``linear_layers`` — hold a fixed-size recurrent state instead of a KV
      cache, so they cost the same at 1K and at 256K context.

    ``n_layers == full + sliding + linear``; the sum is what the naive
    ``2 * n_layers * ...`` formula in older revisions of
    ``fit_gpu_layers`` assumed carried a cache.
    """

    n_layers: int
    full_layers: int
    sliding_layers: int
    linear_layers: int
    sliding_window: int
    kv_heads: int
    key_len: int
    val_len: int
    architecture: Optional[str] = None
    layer_types: tuple = ()

    @property
    def elements_per_layer(self) -> int:
        """Cached scalars per token per layer (K and V together)."""
        return self.kv_heads * (self.key_len + self.val_len)

    def bytes_per_token(self, kv_cache_type: str = "f16") -> float:
        """Marginal bytes the cache grows by for one more token.

        Only full-attention layers grow with the window, so this is the rate
        that matters when sizing a context: 64 KiB/token f16 and 34 KiB/token
        q8_0 for the 27B hybrid, versus 256/136 KiB for the all-layers
        assumption.
        """
        return self.full_layers * self.elements_per_layer * _kv_bytes_per_element(kv_cache_type)

    def bytes_at(self, ctx_size: int, kv_cache_type: str = "f16") -> float:
        """Total KV bytes at a fully-filled ``ctx_size`` window.

        Sliding layers saturate at ``sliding_window``; linear layers are
        excluded entirely (constant state, not a per-token cache).
        """
        ctx = max(0, int(ctx_size))
        per_layer = self.elements_per_layer * _kv_bytes_per_element(kv_cache_type)
        growing = self.full_layers * ctx
        sliding = self.sliding_layers * min(ctx, max(0, self.sliding_window))
        return (growing + sliding) * per_layer

    def naive_bytes_at(self, ctx_size: int, kv_cache_type: str = "f16") -> float:
        """The pre-T33 estimate, verbatim: every layer charged at the full
        window, with K and V multiplied twice on top
        (``2 * n_layers * kv_heads * (key_len + val_len) * ctx * bytes``).

        Kept as the comparison baseline — it is the number the library used to
        refuse contexts that actually fit. On a 27B hybrid (64 layers, 16 of
        them full attention) that is 512 KiB/token against a real 64 KiB, an
        8x over-charge.
        """
        per_layer = self.kv_heads * (self.key_len + self.val_len)
        return (2 * self.n_layers * per_layer * int(ctx_size)
                * _kv_bytes_per_element(kv_cache_type))

    def overcount_factor(self, ctx_size: int, kv_cache_type: str = "f16") -> float:
        """How much the naive formula over-charged at ``ctx_size`` (1.0 = dense)."""
        truth = self.bytes_at(ctx_size, kv_cache_type)
        naive = self.naive_bytes_at(ctx_size, kv_cache_type)
        if truth <= 0:
            return float("inf") if naive > 0 else 1.0
        return naive / truth

    def to_dict(self, ctx_size: Optional[int] = None,
                kv_cache_type: str = "f16") -> dict:
        """JSON-safe view for the stack validator's per-card residency plan."""
        out = {
            "architecture": self.architecture,
            "n_layers": self.n_layers,
            "full_attention_layers": self.full_layers,
            "sliding_layers": self.sliding_layers,
            "linear_layers": self.linear_layers,
            "sliding_window": self.sliding_window,
            "kv_heads": self.kv_heads,
            "key_len": self.key_len,
            "val_len": self.val_len,
            "kv_cache_type": kv_cache_type,
            "bytes_per_token": int(self.bytes_per_token(kv_cache_type)),
            "kib_per_token": round(self.bytes_per_token(kv_cache_type) / 1024.0, 2),
        }
        if ctx_size is not None:
            ctx = int(ctx_size)
            out["ctx_size"] = ctx
            out["kv_bytes"] = int(self.bytes_at(ctx, kv_cache_type))
            out["kv_gib"] = round(self.bytes_at(ctx, kv_cache_type) / (1024 ** 3), 3)
            out["naive_kv_gib"] = round(
                self.naive_bytes_at(ctx, kv_cache_type) / (1024 ** 3), 3)
            out["overcount_factor"] = round(self.overcount_factor(ctx, kv_cache_type), 2)
        return out


def attention_kv_estimate(model_path=None, *, gguf_meta: Optional[dict] = None,
                           kv_cache_type: str = "f16") -> Optional[KVEstimate]:
    """Layer-type-aware KV estimate for a GGUF, or None if the header is thin.

    Reads ``<arch>.attention.layer_types`` when the arch ships it (hybrid
    GatedDeltaNet/Mamba backbones) and falls back to ``sliding_window`` +
    ``sliding_window_pattern`` when it does not (Gemma-3 style), then to the
    per-arch fixed-interval table when the writer omits both (``qwen35`` /
    ``qwen3_5_moe``: one full-attention layer every 4), and finally to
    all-layers attention. Never raises: a missing or unreadable header returns
    None so callers keep their previous behaviour.
    """
    if gguf_meta is None:
        try:
            meta = _read_gguf_metadata(Path(model_path))
        except Exception:  # noqa: BLE001 - best-effort metadata, never block
            return None
    else:
        meta = gguf_meta
    if not meta:
        return None

    n_layers = _meta_first_int(meta, ".block_count")
    head = _meta_first_int(meta, ".attention.head_count")
    emb = _meta_first_int(meta, ".embedding_length")
    kv_heads = _meta_first_int(meta, ".attention.head_count_kv")
    key_len = _meta_first_int(meta, ".attention.key_length")
    val_len = _meta_first_int(meta, ".attention.value_length")
    if not n_layers or not kv_heads:
        return None
    # key_length/value_length are optional in practice (llama.cpp defaults them
    # to embedding_length / head_count, i.e. the head dim). Fall back the same
    # way, independently for each: a GGUF carrying key_length but not
    # value_length is ordinary, and dropping the whole estimate over the
    # missing one would put a dense model back on the naive path.
    if not key_len and emb and head:
        key_len = emb // head
    if not val_len:
        val_len = key_len
    if not key_len or not val_len:
        # Nothing left to derive from: a truncated parse is not worth guessing
        # at, since pricing the cache wrong is worse than deferring to the
        # caller's fallback.
        return None

    window = _meta_first_int(meta, ".attention.sliding_window") or 0
    pattern = _meta_first_int(meta, ".attention.sliding_window_pattern") or 0
    labels = _meta_layer_types(meta)
    arch_key = str(
        meta.get("architecture") or meta.get("general.architecture") or ""
    ).lower()

    full = sliding = linear = 0
    if labels:
        for label in labels:
            if label in _LINEAR_ATTENTION_LABELS:
                linear += 1
            elif label in _SLIDING_ATTENTION_LABELS:
                sliding += 1
            else:
                full += 1
        # A short array (truncated header) still leaves layers unclassified;
        # charge them as full attention rather than under-counting.
        for _ in range(max(0, n_layers - len(labels))):
            full += 1
    elif window > 0:
        if pattern > 0:
            # llama.cpp: layer il slides when il % pattern == 0.
            sliding = len(range(0, n_layers, pattern))
        else:
            sliding = n_layers
        full = n_layers - sliding
    else:
        period = _ARCH_FULL_ATTENTION_PERIOD.get(arch_key)
        if period:
            # No layer_types and no sliding window, but the arch is known to be
            # hybrid at a fixed interval.  A block count one past the last whole
            # period is the appended MTP/nextn head, which mainline llama.cpp
            # ignores at serve time, so it is charged no growing cache.
            main_layers = n_layers - 1 if n_layers % period == 1 else n_layers
            full = max(1, main_layers // period)
            linear = n_layers - full
        else:
            # A plain dense transformer: every layer carries a cache that grows
            # with the window.
            full = n_layers

    return KVEstimate(
        n_layers=n_layers,
        full_layers=full,
        sliding_layers=sliding,
        linear_layers=linear,
        sliding_window=window,
        kv_heads=kv_heads,
        key_len=key_len,
        val_len=val_len,
        architecture=meta.get("architecture") or meta.get("general.architecture"),
        layer_types=tuple(labels),
    )


def fit_gpu_layers(model_path, *, ctx_size: int = 8192,
                   hardware: Optional[dict] = None,
                   gguf_meta: Optional[dict] = None,
                   kv_cache_type: str = "f16") -> Optional[int]:
    """Max transformer layers that fit on the visible GPU(s), unsloth-style.

    Leaves ~10% VRAM headroom and subtracts the KV cache for ``ctx_size``;
    llama.cpp offloads the remaining layers to CPU/GTT. Returns None when the
    GGUF lacks the metadata to estimate reliably (caller falls back to all).

    The KV charge comes from :func:`attention_kv_estimate`, so only layers that
    actually carry a growing (or sliding-window) cache are counted. The older
    formula charged all ``n_layers`` and multiplied K and V twice, which
    over-charged a 27B-class hybrid by 8x and made the library refuse contexts
    that fit. ``kv_cache_type`` should match ``-ctk``/``-ctv`` at serve time;
    it defaults to f16, the llama.cpp default.
    """
    path = Path(model_path)
    try:
        model_bytes = path.stat().st_size
    except OSError:
        return None

    n_layers = _meta_first_int(gguf_meta, ".block_count") if gguf_meta is not None \
        else _meta_first_int(_read_gguf_metadata(path), ".block_count")
    if not n_layers:
        return None
    if hardware is None:
        try:
            from harness.app import _hardware_summary
            hardware = _hardware_summary()
        except Exception:  # noqa: BLE001 - caller falls back
            return None
    vram_free = hardware.get("vram_free_gb")
    if vram_free is None:
        vram_free = hardware.get("combined_vram_gb") or hardware.get("vram_gb")
    if not vram_free:
        return None

    estimate = attention_kv_estimate(path, gguf_meta=gguf_meta,
                                     kv_cache_type=kv_cache_type)
    kv_bytes = estimate.bytes_at(ctx_size, kv_cache_type) if estimate else 0
    budget = float(vram_free) * (1024 ** 3) * 0.90 - kv_bytes
    per_layer = model_bytes / n_layers
    if per_layer <= 0:
        return None
    return int(max(0, min(n_layers, budget // per_layer)))


def _port_in_use(host: str, port: int, timeout: float = 1.0) -> bool:
    """True when something already accepts TCP connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pid_listening_on(port: int) -> Optional[int]:
    """Pid of whatever process LISTENs on this port (best effort)."""
    try:
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if (conn.status == psutil.CONN_LISTEN and conn.laddr
                    and conn.laddr.port == port and conn.pid):
                return conn.pid
    except Exception:  # noqa: BLE001 - psutil limits are environment-specific
        return None
    return None


def _process_name(pid: Optional[int]) -> str:
    if not pid:
        return ""
    try:
        import psutil

        return psutil.Process(pid).name() or ""
    except Exception:  # noqa: BLE001
        return ""


def _terminate_pid(pid: int) -> None:
    try:
        import psutil

        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
    except Exception:  # noqa: BLE001 - already gone
        pass


def gguf_identity(path) -> dict:
    """Best-effort GGUF provenance: name, architecture, quant, size, mtime."""
    p = Path(path)
    meta = _read_gguf_metadata(p)
    try:
        stat = p.stat()
        size_gb = round(stat.st_size / (1024 ** 3), 3)
        mtime = int(stat.st_mtime)
    except OSError:
        size_gb = None
        mtime = None
    return {
        "path": str(p),
        "name": meta.get("general.name"),
        "architecture": meta.get("general.architecture") or meta.get("architecture"),
        "quantization": meta.get("quantization"),
        "size_gb": size_gb,
        "mtime": mtime,
    }


def verify_reference(path, *, expect_name: Optional[str] = None,
                     expect_arch: Optional[str] = None,
                     forbid: Optional[list[str]] = None) -> dict:
    """Check a reference GGUF is the expected base, not a finetune.

    ``ok`` is False when an expectation is unmet: architecture mismatch, name
    missing ``expect_name``, or name containing any ``forbid`` marker (e.g.
    ``["dflash", "uncensored"]`` — substring matching alone cannot separate a
    finetune whose name merely contains the base name)."""
    identity = gguf_identity(path)
    name = str(identity.get("name") or "")
    problems: list[str] = []
    if expect_arch and identity.get("architecture") and identity["architecture"] != expect_arch:
        problems.append(
            f"architecture {identity['architecture']!r} != {expect_arch!r}")
    if expect_name and name and expect_name.lower() not in name.lower():
        problems.append(f"name {name!r} does not contain {expect_name!r}")
    for marker in (forbid or []):
        if marker and marker.lower() in name.lower():
            problems.append(f"name {name!r} contains forbidden marker {marker!r}")
    result: dict = {"ok": not problems, "identity": identity}
    if problems:
        result["reason"] = "; ".join(problems)
    return result


def stop_stale_server(host: str, port: int) -> dict:
    """Terminate a llama-server orphan holding ``host:port`` (by PID, never by
    pattern). Safe when the port is free or held by something else."""
    if not _port_in_use(host, port):
        return {"stopped": None, "reason": "port free"}
    pid = _pid_listening_on(port)
    name = _process_name(pid)
    if pid and "llama" in name.lower():
        _terminate_pid(pid)
        return {"stopped": pid, "name": name}
    return {"stopped": None, "reason": f"port held by {name or 'unknown'}"}


def competing_gpu_processes() -> list[dict]:
    """Other llama.cpp processes that would contend for the GPU/ROCm stack.

    Advisory: two independent ROCm contexts can hang this host, so callers
    should refuse to start a heavy load while this is non-empty."""
    out: list[dict] = []
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return out
    me = os.getpid()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if proc.info["pid"] == me:
                continue
            name = (proc.info.get("name") or "")
            low = name.lower()
            if low.startswith("llama-") or low == "llama-server":
                out.append({"pid": proc.info["pid"], "name": name})
        except Exception:  # noqa: BLE001 - process vanished
            continue
    return out


def launch_extra_args(load_options: dict) -> list[str]:
    """llama-server argv extras recorded in an engine profile's load_options
    (shared by the start endpoint, /model command, and CLI auto-start)."""
    args: list[str] = []
    if load_options.get("threads"):
        args += ["-t", str(load_options["threads"])]
    if load_options.get("flash_attn"):
        args += ["-fa", "on"]
    if load_options.get("parallel_slots"):
        args += ["-np", str(load_options["parallel_slots"])]
    if load_options.get("cache_type_k"):
        args += ["--cache-type-k", load_options["cache_type_k"]]
    if load_options.get("cache_type_v"):
        args += ["--cache-type-v", load_options["cache_type_v"]]
    if load_options.get("batch_size"):
        args += ["-b", str(load_options["batch_size"])]
    if load_options.get("ubatch_size"):
        args += ["-ub", str(load_options["ubatch_size"])]
    if load_options.get("alias"):
        args += ["--alias", load_options["alias"]]
    if load_options.get("mlock"):
        args += ["--mlock"]
    if load_options.get("no_mmap"):
        args += ["--no-mmap"]
    if load_options.get("mmproj"):
        args += ["--mmproj", str(load_options["mmproj"])]
    return args


@dataclass
class ServerInstance:
    """One loaded model: its llama-server process, port, and identity."""

    key: str
    port: int
    model: str
    pid: Optional[int] = None
    adopted: bool = False
    started_at: str = ""
    backend: Optional[str] = None
    binary: str = ""
    embedding: bool = False
    proc: object = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "running": True,
            "healthy": True,
            "port": self.port,
            "base_url": f"http://127.0.0.1:{self.port}",
            "model": self.model,
            "pid": self.pid,
            "adopted": self.adopted,
            "started_at": self.started_at,
            "backend": self.backend,
            "binary": self.binary,
            "embedding": self.embedding,
        }


class DownloadJob:
    """Status record for one managed HF file download."""

    def __init__(self, key: str, repo: str, filename: str) -> None:
        self.key = key
        self.repo = repo
        self.filename = filename
        self.state = "queued"  # queued | downloading | done | error
        self.error = ""
        self.path: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "key": self.key, "repo": self.repo, "filename": self.filename,
            "state": self.state, "error": self.error, "path": self.path,
            "elapsed_s": round((self.finished_at or time.time())
                               - self.started_at, 1),
        }


class LlamaServerManager:
    """Multi-model llama-server fleet, downloads, and the local library.

    Each ``load()`` spawns (or adopts) one llama-server on its own port and
    registers it under a unique key; ``unload(key)`` stops exactly that one.
    """

    def __init__(
        self,
        binary: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 1234,
        log_dir: Optional[Path] = None,
        spawner: Callable[..., object] = subprocess.Popen,
        prober: Optional[Callable[[str], object]] = None,
        startup_timeout: float = 300.0,
        hardware_probe: Optional[Callable[[], dict]] = None,
        memory_guard: Optional[bool] = None,
    ) -> None:
        self.binary = Path(binary) if binary else _default_binary()
        # Persisted override from previous directory selection
        persisted: Optional[Path] = None
        try:
            cfg = REPO_ROOT / "harness_state" / "models_dir.txt"
            if cfg.is_file():
                p = Path(cfg.read_text(encoding="utf-8").strip())
                if p.is_dir():
                    persisted = p
        except OSError:
            pass
        if models_dir is not None:
            self.models_dir = Path(models_dir)
        elif persisted is not None:
            self.models_dir = persisted
        else:
            self.models_dir = _default_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self.log_dir = Path(log_dir) if log_dir else REPO_ROOT / "logs"
        self.spawner = spawner
        self.prober = prober or _probe
        self.startup_timeout = startup_timeout
        self.hardware_probe = hardware_probe
        if memory_guard is None:
            memory_guard = os.environ.get(
                "HARNESS_MEMORY_GUARD", "1").strip().lower() not in (
                    "0", "false", "no", "off")
        self.memory_guard = bool(memory_guard)
        self._lock = threading.RLock()
        self._instances: dict[str, ServerInstance] = {}
        self._downloads: dict[str, DownloadJob] = {}
        self._local_cache: list[dict] | None = None
        self._local_cache_ts: float = 0.0

    # ------------------------------------------------------------------
    # local library
    # ------------------------------------------------------------------
    def server_log(self, tail: int = 120) -> list[str]:
        """Last lines of the llama-server output (startup errors land here).

        Logs are per-port (``llama_server_<port>.log``); the most recently
        written file wins."""
        candidates = sorted(self.log_dir.glob("llama_server*.log"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return []
        text = candidates[0].read_text(encoding="utf-8", errors="replace")
        return text.splitlines()[-max(1, tail):]

    def _raise_startup_exit(self, proc, log_path: Path) -> None:
        """Raise the right error when llama-server dies before becoming healthy.

        An allocation failure in the log becomes ``OutOfMemoryError`` with the
        log tail attached; anything else keeps the original RuntimeError."""
        code = proc.poll()
        tail = _read_log_tail(log_path)
        if _looks_like_oom(tail):
            raise OutOfMemoryError(
                f"llama-server ran out of memory during startup (exit {code}). "
                "Lower --ctx-size or -ngl, free VRAM/RAM, or set "
                "HARNESS_MEMORY_GUARD=0 to skip the preflight. "
                f"Log tail:\n{tail[-600:]}"
            )
        raise RuntimeError(
            f"llama-server exited during startup (code {code}); see {log_path}"
        )

    def list_local(self, use_cache: bool = True) -> list[dict]:
        # Cache for 30s to avoid re-reading GGUF headers on every poll (refresh every 15s + tab switch)
        if use_cache and self._local_cache is not None and (time.time() - self._local_cache_ts) < 30.0:
            return self._local_cache
        t0=time.time()
        entries = []
        for p in sorted(self.models_dir.rglob("*.gguf")):
            try:
                stat = p.stat()
            except OSError:
                continue
            try:
                rel = str(p.relative_to(self.models_dir))
            except ValueError:
                rel = p.name
            meta = _read_gguf_metadata(p)
            size_gb = round(stat.st_size / (1024 ** 3), 2)
            modified = time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(stat.st_mtime))
            entry: dict = {
                "name": p.stem,
                "file": rel,
                "size_gb": size_gb,
                "sizeGb": size_gb,
                "modified": modified,
                "lastModified": modified,
            }
            # gguf-metadata fields for the hover card (inline)
            arch = meta.get("architecture") or meta.get("general.architecture")
            quant = meta.get("quantization")
            ctx = meta.get("context_length")
            if arch is not None:
                entry["architecture"] = arch
            if quant is not None:
                entry["quantization"] = quant
            if ctx is not None:
                entry["context_length"] = int(ctx)
                entry["contextLength"] = int(ctx)
            # keep raw gguf_metadata for advanced consumers (Auto preset, etc.)
            if meta:
                entry["gguf_metadata"] = meta
                entry["ggufMetadata"] = meta
            entries.append(entry)
        out=sorted(entries, key=lambda e: e["modified"], reverse=True)
        self._local_cache=out
        self._local_cache_ts=time.time()
        print(f"list_local: {len(out)} models in {(self._local_cache_ts-t0):.2f}s from {self.models_dir}")
        return out

    def invalidate_cache(self) -> None:
        self._local_cache=None
        self._local_cache_ts=0.0

    def import_local_path(self, folder: str) -> dict:
        """Link an external folder into the library without copying (instant).

        Creates a directory junction/symlink at ``models_dir/<folder-name>``
        pointing at ``folder`` so ``list_local`` finds its .gguf files via
        the existing rglob. No file contents are copied.
        """
        src = Path(folder).expanduser().resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"folder not found: {src}")
        ggufs = list(src.rglob("*.gguf"))
        if not ggufs:
            raise FileNotFoundError(f"no .gguf files under {src}")
        # If folder is already inside models_dir, nothing to link
        try:
            src.relative_to(self.models_dir.resolve())
            return {"imported": [str(p.relative_to(self.models_dir)) for p in ggufs[:50]], "linked": False, "models_dir": str(self.models_dir)}
        except ValueError:
            pass
        link_name = src.name or "imported"
        dest = (self.models_dir / link_name).resolve()
        # Avoid collision
        if dest.exists():
            if dest.is_symlink() or dest.is_dir():
                # already linked
                try:
                    if dest.resolve() == src:
                        return {"imported": [str(p.relative_to(self.models_dir)) for p in self.list_local() if link_name in str(p)], "linked": False, "models_dir": str(self.models_dir)}
                except OSError:
                    pass
            # collision with different target — use unique name
            n = 2
            while dest.exists():
                dest = (self.models_dir / f"{link_name}-{n}").resolve()
                n += 1
        try:
            # Windows: directory junction/symlink; fallback to .pth-like shortcut
            if os.name == "nt":
                import ctypes
                # Try symlink first (requires admin/dev mode), fallback to junction via mklink /J
                try:
                    dest.symlink_to(src, target_is_directory=True)
                except OSError:
                    subprocess.run(["cmd", "/c", "mklink", "/J", str(dest), str(src)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                dest.symlink_to(src, target_is_directory=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to link {src} -> {dest}: {exc}")
        imported = [str(p.relative_to(self.models_dir)) for p in sorted(dest.rglob("*.gguf"))]
        return {"imported": imported, "linked": str(dest), "models_dir": str(self.models_dir)}

    def set_models_dir(self, folder: str) -> dict:
        """Point the library at a new folder (no copy, instant).

        Default is LM Studio's ``.lmstudio/models`` when it exists, else
        ``models/gguf``. This call switches the live ``models_dir`` and
        persists it to ``harness_state/models_dir.txt`` so Hugging Face
        downloads and ``list_local`` use the chosen folder.
        """
        p = Path(folder).expanduser().resolve()
        # Create if it doesn't exist (for fresh download target)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise FileNotFoundError(f"cannot create folder {p}: {exc}")
        if not p.is_dir():
            raise NotADirectoryError(f"not a directory: {p}")
        # Require read access
        if not os.access(p, os.R_OK):
            raise PermissionError(f"cannot read folder: {p}")
        self.models_dir = p
        try:
            cfg = REPO_ROOT / "harness_state" / "models_dir.txt"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(str(p), encoding="utf-8")
        except OSError:
            pass
        return {"models_dir": str(p), "models": self.list_local()}

    def resolve_model(self, name: str) -> Optional[Path]:
        """A model arg is either a local library name/file or a real path."""
        candidate = Path(name)
        if candidate.is_file():
            return candidate.resolve()
        direct = self.models_dir / name
        if direct.is_file():
            return direct.resolve()
        matches = [p for p in self.models_dir.rglob("*.gguf")
                   if name in p.stem or name == p.name]
        return matches[0].resolve() if matches else None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _instance_alive(self, inst: ServerInstance) -> bool:
        if inst.adopted:
            return True
        return inst.proc is not None and inst.proc.poll() is None

    def _probe_instance(self, inst: ServerInstance) -> object:
        try:
            return self.prober(f"http://{self.host}:{inst.port}")
        except Exception:  # noqa: BLE001 - a dead instance probes False
            return False

    def _prune_dead(self) -> None:
        for key in list(self._instances):
            inst = self._instances[key]
            if not self._instance_alive(inst):
                del self._instances[key]

    def _free_port(self, preferred: Optional[int] = None) -> int:
        """First free port at/below-or-after ``preferred`` (our own loaded
        instances' ports are busy by definition, so they are skipped)."""
        port = preferred if preferred is not None else self.port
        for _ in range(64):
            if not _port_in_use(self.host, port):
                return port
            port += 1
        raise RuntimeError("no free port found in the llama-server range")

    def _unique_key(self, base: str) -> str:
        key = base
        n = 2
        while key in self._instances:
            key = f"{base}-{n}"
            n += 1
        return key

    def status(self) -> dict:
        with self._lock:
            self._prune_dead()
            instances = [i.to_dict() for i in
                         sorted(self._instances.values(), key=lambda i: i.key)]
        for inst_dict in instances:
            inst = self._instances[inst_dict["key"]]
            probe = self._probe_instance(inst)
            inst_dict["healthy"] = bool(probe)
            if probe and probe is not True:
                inst_dict["model"] = probe
        primary = instances[0] if instances else None
        return {
            "running": bool(instances),
            "healthy": bool(instances) and all(i["healthy"] for i in instances),
            "binary": str(self.binary),
            "binary_found": self.binary.is_file(),
            "port": primary["port"] if primary else self.port,
            "base_url": primary["base_url"] if primary
            else f"http://{self.host}:{self.port}",
            "model": primary["model"] if primary else None,
            "pid": primary["pid"] if primary else None,
            "started_at": primary["started_at"] if primary else None,
            "instances": instances,
            "local_models": self.list_local(),
            "downloads": self.downloads_status(),
        }

    def load(
        self,
        model: Optional[str] = None,
        hf_repo: Optional[str] = None,
        hf_file: Optional[str] = None,
        key: Optional[str] = None,
        port: Optional[int] = None,
        ctx_size: int = 8192,
        ngl: "int | str | None" = 999,
        extra_args: Optional[list[str]] = None,
        env: Optional[dict] = None,
        backend: Optional[str] = None,
        embedding: bool = False,
        pooling: Optional[str] = None,
        mmproj: Optional[str] = None,
    ) -> dict:
        """Load one model as its own llama-server instance. Priority: local
        ``model`` > --hf-repo/--hf-file passthrough. Loading the same model
        twice is rejected; different models coexist on different ports.

        ``backend`` selects a per-backend llama-server binary from
        ``tools/backends/<backend>/`` (vulkan | rocm | cuda | cpu | sycl;
        fetched via tools/fetch_backend.ps1). Unset/unknown falls back to the
        default binary. ``embedding`` enables ``--embedding`` mode for
        embedding models (bge-m3, nomic-embed, etc. — uses ``--pooling``
        when provided)."""
        if pooling is not None:
            pv = pooling.strip().lower()
            if pv not in ("mean", "cls", "last"):
                raise RuntimeError(
                    f"unknown pooling '{pooling}'; known: mean, cls, last")
            pooling = pv
        if pooling is not None and not embedding:
            raise RuntimeError("pooling requires embedding=True")
        binary = self.binary
        backend_name = (backend or "").strip().lower() or None
        if backend_name and backend_name not in BACKENDS:
            raise RuntimeError(
                f"unknown backend '{backend_name}'; known: {', '.join(BACKENDS)}")
        if backend_name:
            override = _binary_for_backend(backend_name)
            if override is not None:
                binary = override
            elif backend_name != "vulkan":
                # the default binary IS the vulkan build; anything else must
                # be fetched explicitly
                raise RuntimeError(
                    f"no llama-server binary for backend '{backend_name}' under "
                    f"tools/backends/{backend_name}/ - fetch one with "
                    "tools/fetch_backend.ps1")
        if not binary.is_file():
            raise RuntimeError(
                f"llama-server not found at {self.binary}. Set HARNESS_LLAMA_SERVER "
                "or extract a llama.cpp Vulkan release into tools/llama.cpp/."
            )
        resolved = self.resolve_model(model) if model else None
        if model and resolved is None and not (hf_repo and hf_file):
            raise RuntimeError(
                f"model '{model}' is neither a local file nor paired with "
                "--hf-repo/--hf-file; see GET /v1/models/local"
            )
        with self._lock:
            self._prune_dead()
            if resolved is not None:
                for inst in self._instances.values():
                    if inst.model in (resolved.name, resolved.stem, str(resolved)):
                        raise RuntimeError(
                            f"'{resolved.name}' is already loaded "
                            f"(instance '{inst.key}' on port {inst.port})"
                        )

        model_name = (resolved.name if resolved else (hf_file or "model"))
        base_key = key or Path(model_name).stem or "model"
        use_port = int(port) if port else self._free_port(self.port)
        # Pre-flight: something is already on this port. If it is a healthy
        # llama-server, ADOPT it (sidecar restarts must not orphan or fight
        # their own servers); anything else is refused loudly.
        if _port_in_use(self.host, use_port):
            base_url = f"http://{self.host}:{use_port}"
            listener_pid = _pid_listening_on(use_port)
            listener_name = _process_name(listener_pid)
            try:
                probe = self.prober(base_url)
            except Exception:  # noqa: BLE001 - a broken squatter is not ours
                probe = False
            if probe and "llama" in listener_name.lower():
                inst = ServerInstance(
                    key=self._unique_key(base_key),
                    port=use_port,
                    model=(probe if probe is not True else model_name),
                    pid=listener_pid,
                    adopted=True,
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    backend=backend_name,
                    binary=str(binary),
                    embedding=embedding,
                )
                with self._lock:
                    self._instances[inst.key] = inst
                return inst.to_dict()
            raise RuntimeError(
                f"port {use_port} is already serving"
                + (f" (pid {listener_pid}: {listener_name})" if listener_pid else "")
                + "; stop that server or pick another port."
            )

        if isinstance(ngl, str):
            ngl = ngl.strip().lower()
            ngl = None if ngl in ("", "auto", "fit") else int(ngl)
        if ngl is None:
            computed = fit_gpu_layers(resolved, ctx_size=ctx_size) if resolved else None
            ngl = computed if computed is not None else 999

        memory_info: Optional[dict] = None
        if self.memory_guard and resolved is not None:
            memory_info = preflight_memory(resolved,
                                           hardware_probe=self.hardware_probe)
            if not memory_info.get("ok", True):
                raise OutOfMemoryError(
                    f"refusing to load '{model_name}': {memory_info.get('reason')}. "
                    "Lower --ctx-size/-ngl, free VRAM, or set HARNESS_MEMORY_GUARD=0 "
                    "to override.")
            if memory_info.get("warning"):
                print(f"[models] {memory_info['warning']}", file=sys.stderr)

        cmd = [str(binary)]
        if resolved is not None:
            cmd += ["-m", str(resolved)]
        elif hf_repo and hf_file:
            cmd += ["--hf-repo", hf_repo, "--hf-file", hf_file]
        if embedding:
            cmd += ["--embedding"]
            if pooling:
                cmd += ["--pooling", pooling]
        cmd += [
            "--host", self.host,
            "--port", str(use_port),
            "-ngl", str(ngl),
            "-c", str(ctx_size),
        ]
        if embedding:
            cmd += ["--embedding"]
            if pooling:
                cmd += ["--pooling", pooling]
        if mmproj:
            cmd += ["--mmproj", mmproj]
        cmd += [str(a) for a in (extra_args or [])]

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"llama_server_{use_port}.log"
        log_handle = open(log_path, "ab")
        try:
            sp_kwargs: dict = {"stdout": log_handle, "stderr": subprocess.STDOUT}
            if env:
                # Per-load card selection: e.g. {"HIP_VISIBLE_DEVICES": "1"}.
                sp_kwargs["env"] = {**os.environ, **env}
            if os.name == "nt":
                sp_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                proc = self.spawner(cmd, **sp_kwargs)
            except TypeError as e:
                if "creationflags" in str(e):
                    sp_kwargs.pop("creationflags", None)
                    proc = self.spawner(cmd, **sp_kwargs)
                else:
                    raise
        finally:
            log_handle.close()

        base_url = f"http://{self.host}:{use_port}"
        deadline = time.time() + self.startup_timeout
        model_id = None
        while time.time() < deadline:
            if proc.poll() is not None:
                self._raise_startup_exit(proc, log_path)
            probe = self.prober(base_url)
            if probe:
                # The probe must answer for OUR process: re-check liveness so a
                # foreign server that raced the bind is not misattributed.
                time.sleep(0.5)
                if proc.poll() is not None:
                    self._raise_startup_exit(proc, log_path)
                model_id = probe if probe is not True else None
                break
            time.sleep(0.5)
        else:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            raise RuntimeError(
                f"llama-server did not become healthy within "
                f"{self.startup_timeout:.0f}s; see {log_path}"
            )

        inst = ServerInstance(
            key=self._unique_key(base_key),
            port=use_port,
            model=model_id or model_name,
            pid=getattr(proc, "pid", None),
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            backend=backend_name,
            binary=str(binary),
            embedding=embedding,
            proc=proc,
        )
        with self._lock:
            self._instances[inst.key] = inst
        out = inst.to_dict()
        out["ngl"] = ngl
        if memory_info is not None:
            out["memory"] = memory_info
        return out

    def unload(self, key: str) -> dict:
        with self._lock:
            inst = self._instances.pop(key, None)
        if inst is None:
            raise RuntimeError(f"no such loaded instance: {key}")
        if inst.proc is not None:
            try:
                inst.proc.terminate()
                try:
                    inst.proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    inst.proc.kill()
            except Exception:  # noqa: BLE001 - already gone
                pass
        elif inst.pid:
            _terminate_pid(inst.pid)
        return {"ok": True, "unloaded": key}

    def stop_all(self) -> dict:
        with self._lock:
            keys = list(self._instances)
        for key in keys:
            try:
                self.unload(key)
            except RuntimeError:
                pass
        return {"ok": True, "stopped": len(keys)}

    # Backward-compatible single-instance vocabulary -----------------------
    def start(self, **kwargs) -> dict:
        return self.load(**kwargs)

    def stop(self) -> dict:
        return self.stop_all()

    # ------------------------------------------------------------------
    # Hugging Face (live hub API — nothing cached/hardcoded)
    # ------------------------------------------------------------------
    def hub_search(self, query: str = "", limit: int = 12) -> list[dict]:
        params = {"filter": "gguf", "sort": "downloads", "limit": limit}
        if query:
            params["search"] = query
        resp = requests.get(f"{HF_API}/models", params=params, timeout=15)
        resp.raise_for_status()
        return [
            {
                "repo": m.get("id", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "last_modified": (m.get("lastModified") or "")[:10],
            }
            for m in resp.json()
        ]

    def hub_files(self, repo: str) -> list[dict]:
        resp = requests.get(f"{HF_API}/models/{repo}/tree/main", timeout=20)
        resp.raise_for_status()
        return [
            {"file": e["path"], "size_gb": round(e.get("size", 0) / (1024 ** 3), 2)}
            for e in resp.json()
            if e.get("path", "").endswith(".gguf")
        ]

    def delete_local(self, file: str) -> dict:
        """Remove one GGUF from the local library (path-traversal safe)."""
        root = self.models_dir.resolve()
        target = (root / file).resolve()
        if root not in target.parents and target != root:
            raise ValueError("file must stay inside the models directory")
        if not target.is_file():
            raise FileNotFoundError(f"no such model file: {file}")
        target.unlink()
        return {"ok": True, "deleted": str(target)}

    def server_log(self, tail: int = 120) -> list[str]:
        """Last lines of the llama-server logs (startup errors live here)."""
        lines: list[str] = []
        for path in sorted(self.log_dir.glob("llama_server*.log")):
            lines += path.read_text(encoding="utf-8",
                                    errors="replace").splitlines()[-max(1, tail):]
        return lines

    def server_metrics(self, key: Optional[str] = None) -> dict:
        """Proxy llama-server's prometheus /metrics (kv usage, throughput)."""
        with self._lock:
            inst = self._instances.get(key) if key else None
            port = inst.port if inst else (
                next(iter(self._instances.values())).port
                if self._instances else self.port)
        base = f"http://{self.host}:{port}"
        try:
            resp = requests.get(f"{base}/metrics", timeout=3)
        except requests.RequestException:
            return {"available": False}
        if not resp.ok:
            return {"available": False}
        gauges = {}
        for line in resp.text.splitlines():
            if line.startswith("#") or " " not in line:
                continue
            k, _, v = line.partition(" ")
            try:
                gauges[k] = float(v)
            except ValueError:
                continue
        return {"available": True, "gauges": gauges}

    def downloads_status(self) -> list[dict]:
        self._prune_downloads()
        return [job.to_dict() for job in self._downloads.values()]

    def _prune_downloads(self, keep: int = 20) -> None:
        """Drop finished/failed jobs beyond the newest ``keep`` — the registry
        must not grow with every hub download a browser ever started."""
        with self._lock:
            finished = sorted(
                (j for j in self._downloads.values()
                 if j.state in ("done", "error")),
                key=lambda j: j.started_at,
            )
            excess = max(0, len(finished) - keep)
            for job in finished[:excess]:
                self._downloads.pop(job.key, None)

    def download(self, repo: str, filename: str) -> dict:
        key = f"{repo}:{filename}"
        with self._lock:
            existing = self._downloads.get(key)
            if existing and existing.state in ("queued", "downloading"):
                return existing.to_dict()
            job = DownloadJob(key, repo, filename)
            self._downloads[key] = job

        def run(job: DownloadJob) -> None:
            job.state = "downloading"
            try:
                path = _hf_download(repo, filename, self.models_dir)
                job.state = "done"
                job.path = str(path)
            except Exception as exc:  # noqa: BLE001 - surfaced via status
                job.state = "error"
                job.error = str(exc)
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, args=(job,), daemon=True).start()
        return job.to_dict()
