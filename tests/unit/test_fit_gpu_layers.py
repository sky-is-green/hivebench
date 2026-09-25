"""Unit tests for the attention-aware KV estimator (``harness/models.py``).

All offline and synthetic: no GPU, no network, no GGUF download. The reference
rates are the measured ones in ``LOCAL-STACKS.md`` section 5, which the
estimator must reproduce — a naive "every layer carries a KV cache that grows
with the window" model over-charges a hybrid architecture by 4-8x and makes the
library refuse contexts that fit.

Covers the three regression architectures the task calls for: hybrid
(linear-attention + full-attention), dense, and sliding-window, plus the
GGUF-header parse that feeds the estimator.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

import harness.models as mm


KIB = 1024.0
GIB = 1024.0 ** 3


def _sparse(path: Path, gb: float) -> Path:
    """A file of the right size without spending the disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(int(gb * GIB))
    return path


def _hybrid_meta(**overrides) -> dict:
    """Qwen3-Next-shaped 27B: 64 layers, every 4th full attention."""
    meta = {
        "general.architecture": "qwen3next",
        "qwen3next.block_count": 64,
        "qwen3next.embedding_length": 5120,
        "qwen3next.attention.head_count": 40,
        "qwen3next.attention.head_count_kv": 8,
        "qwen3next.attention.key_length": 128,
        "qwen3next.attention.value_length": 128,
        "qwen3next.attention.layer_types": [
            "linear_attention" if i % 4 else "full_attention" for i in range(64)
        ],
    }
    meta.update(overrides)
    return meta


def _within(actual: float, expected: float, tol: float = 0.10) -> bool:
    """True when ``actual`` is within ``tol`` (relative) of ``expected``."""
    return abs(actual - expected) <= abs(expected) * tol


# ---------------------------------------------------------------------------
# Reference rates (LOCAL-STACKS.md section 5) — the acceptance criterion
# ---------------------------------------------------------------------------
def test_hybrid_per_token_rate_matches_measured_64_and_34_kib():
    """27B hybrid, 64 layers, 16 full attention: 64 KiB/tok f16, 34 KiB/tok Q8_0.

    8 KV heads x (128 + 128) elements x 16 full-attention layers, at 2 bytes
    (f16) and 34/32 bytes (q8_0, the block's int8 payload + fp16 scale).
    """
    est = mm.attention_kv_estimate(gguf_meta=_hybrid_meta())

    assert est is not None
    assert est.full_layers == 16
    assert est.linear_layers == 48
    assert est.sliding_layers == 0
    assert est.kv_heads == 8
    assert est.key_len == 128 and est.val_len == 128

    f16_kib = est.bytes_per_token("f16") / KIB
    q8_kib = est.bytes_per_token("q8_0") / KIB
    assert _within(f16_kib, 64.0), f"f16 {f16_kib:.2f} KiB/tok != 64 KiB/tok"
    assert _within(q8_kib, 34.0), f"q8_0 {q8_kib:.2f} KiB/tok != 34 KiB/tok"
    # The two are independent formulas, so pin the absolute values too: a
    # tolerance-only assertion would pass a constant 34 KiB on every arch.
    assert est.bytes_per_token("f16") == 65536.0
    assert est.bytes_per_token("q8_0") == 34816.0


@pytest.mark.parametrize("ctx,expected_gib", [
    (65536, 2.1), (131072, 4.3), (262144, 8.5),
])
def test_hybrid_context_totals_match_measured_gib(ctx, expected_gib):
    """Q8_0 totals at 64K/128K/256K: 2.1 / 4.3 / 8.5 GiB."""
    est = mm.attention_kv_estimate(gguf_meta=_hybrid_meta())

    got = est.bytes_at(ctx, "q8_0") / GIB
    assert _within(got, expected_gib), f"{ctx}: {got:.2f} GiB != {expected_gib} GiB"


def test_reference_table_rows_two_and_three():
    """The other two measured shapes stay within 10% as well.

    9B/4B, 32 layers, 8 full attention, 8 KV heads -> 32 / 16 KiB per token.
    35B-A3B, 40 layers, 10 full attention, 4 KV heads -> 20 / 10 KiB per token.

    The GiB columns are checked against the table's own per-token figure
    rather than against its separately-rounded GiB cells: the table is
    internally consistent as rate x context (10 KiB/tok x 64K = 0.625 ->
    "0.6"), so rounding twice would make an exact estimator look wrong.
    """
    shapes = [
        # (block_count, full_attention, kv_heads, embedding, f16 KiB, q8_0 KiB)
        (32, 8, 8, 2560, 32.0, 16.0),
        (40, 10, 4, 2048, 20.0, 10.0),
    ]
    for n_layers, full, kv_heads, emb, f16_kib, q8_kib in shapes:
        est = mm.attention_kv_estimate(gguf_meta={
            "general.architecture": "qwen3",
            "qwen3.block_count": n_layers,
            "qwen3.embedding_length": emb,
            "qwen3.attention.head_count": 32,
            "qwen3.attention.head_count_kv": kv_heads,
            "qwen3.attention.key_length": 128,
            "qwen3.attention.value_length": 128,
            "qwen3.attention.layer_types": [
                "full_attention" if i < full else "linear_attention"
                for i in range(n_layers)
            ],
        })
        assert est.full_layers == full
        assert _within(est.bytes_per_token("f16") / KIB, f16_kib)
        assert _within(est.bytes_per_token("q8_0") / KIB, q8_kib)
        for ctx in (65536, 131072, 262144):
            from_table = q8_kib * KIB * ctx / GIB
            assert _within(est.bytes_at(ctx, "q8_0") / GIB, from_table)


def test_exact_kiB_rates_behind_the_rounded_reference_table():
    """Pin the exact rates, so the table's own rounding is visible.

    LOCAL-STACKS.md section 5 prints 10 KiB/tok for the 35B-A3B row; the shape
    (10 full-attention layers x 4 KV heads x 256 elements, q8_0's 34/32 bytes
    per element) works out to 10.625, which the table rounds to 10 and which
    then yields 0.66 GiB at 64K where the table prints 0.6.
    """
    est = mm.KVEstimate(n_layers=40, full_layers=10, sliding_layers=0,
                        linear_layers=30, sliding_window=0, kv_heads=4,
                        key_len=128, val_len=128, architecture="qwen3moe")

    assert est.bytes_per_token("f16") == 20480.0          # 20.0 KiB exactly
    assert est.bytes_per_token("q8_0") == 10880.0         # 10.625 KiB
    assert round(est.bytes_at(65536, "q8_0") / GIB, 1) == 0.7


# ---------------------------------------------------------------------------
# Architecture 1 — hybrid: only full-attention layers grow
# ---------------------------------------------------------------------------
def test_hybrid_linear_layers_cost_nothing():
    """Linear-attention (GatedDeltaNet/Mamba) layers hold a fixed recurrent
    state, so they contribute no per-token KV at any context size."""
    est = mm.attention_kv_estimate(gguf_meta=_hybrid_meta())
    dense_equivalent = mm.KVEstimate(
        n_layers=64, full_layers=64, sliding_layers=0, linear_layers=0,
        sliding_window=0, kv_heads=8, key_len=128, val_len=128,
    )

    # Same 64-layer model with all attention: 4x the rate of the real hybrid.
    assert est.bytes_per_token("f16") == dense_equivalent.bytes_per_token("f16") / 4
    # Growing the window 8x grows the hybrid's KV 8x (every layer it holds is
    # unbounded) — the discount is structural, not a context-size effect.
    assert _within(est.bytes_at(65536) / est.bytes_at(8192), 8.0, 0.01)


def test_hybrid_overcount_factor_is_8x_against_the_old_formula():
    """The pre-T33 formula was wrong by 8x here — the defect this task fixes.

    It charged all 64 layers instead of the 16 that carry a growing cache (4x)
    and multiplied K and V twice (2x more).
    """
    est = mm.attention_kv_estimate(gguf_meta=_hybrid_meta())

    assert _within(est.overcount_factor(65536, "f16"), 8.0, 0.001)
    assert _within(est.naive_bytes_at(1, "f16") / KIB, 512.0, 0.001)
    assert _within(est.bytes_at(1, "f16") / KIB, 64.0, 0.001)


def test_fit_gpu_layers_offloads_more_on_a_hybrid_than_a_dense_assumption():
    """Same weights, same card: the hybrid packs strictly more layers on-GPU.

    This is the user-visible payoff of the fix — the old estimate reserved 8x
    the VRAM it needed and needlessly offloaded layers to system memory.
    """
    model = _sparse(Path("m.gguf"), 20.0)
    hw = {"vram_free_gb": 20.0}
    hybrid = mm.fit_gpu_layers(model, ctx_size=32768, hardware=hw,
                               gguf_meta=_hybrid_meta())

    # Same model, but labelled as if every layer attended.
    dense_meta = _hybrid_meta()
    dense_meta["qwen3next.attention.layer_types"] = ["full_attention"] * 64
    dense = mm.fit_gpu_layers(model, ctx_size=32768, hardware=hw,
                              gguf_meta=dense_meta)

    assert 0 < hybrid <= 64
    assert hybrid > dense
    # The dense reading is the corrected all-layers formula: one K and one V
    # per head, each charged once. The old code's `2 * n_layers * kv_heads *
    # (key_len + val_len) * ... * 2` counted K and V twice, reserving 16 GiB of
    # a 20 GB card for a cache that needs 8 GiB — it stranded 26 of 64 layers
    # in system memory for nothing. The hybrid reading is the layer-type fix
    # on top: 2 GiB of KV, and 51 layers resident.
    kv = 64 * 8 * (128 + 128) * 32768 * 2
    per_layer = 20.0 * GIB / 64
    budget = 20.0 * GIB * 0.90
    assert dense == int((budget - kv) // per_layer)
    assert hybrid == int((budget - 16 * 8 * (128 + 128) * 32768 * 2) // per_layer)


def test_fit_gpu_layers_kv_cache_type_is_honoured():
    """q8_0 KV frees headroom that f16 does not."""
    model = _sparse(Path("m.gguf"), 20.0)
    hw = {"vram_free_gb": 20.0}
    meta = _hybrid_meta()
    # Force a mid-range answer so the delta cannot be clipped at 0 or 64.
    hw_tight = {"vram_free_gb": 19.0}

    f16 = mm.fit_gpu_layers(model, ctx_size=65536, hardware=hw_tight,
                            gguf_meta=meta, kv_cache_type="f16")
    q8 = mm.fit_gpu_layers(model, ctx_size=65536, hardware=hw_tight,
                           gguf_meta=meta, kv_cache_type="q8_0")

    assert q8 > f16


# ---------------------------------------------------------------------------
# Architecture 2 — dense: no discount, no regression
# ---------------------------------------------------------------------------
def test_dense_arch_charges_every_layer_once():
    """A plain transformer has no layer_types and no sliding window.

    All ``n_layers`` carry a growing cache, so there is no structural discount
    to apply — the only thing the old formula got wrong here is the doubled
    K and V, which is why the overcount factor is exactly 2 and not 8.
    """
    est = mm.attention_kv_estimate(gguf_meta={
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
    })

    assert est is not None
    assert est.full_layers == 32
    assert est.sliding_layers == 0 and est.linear_layers == 0
    # 32 layers x 8 KV heads x 256 elements x 2 bytes = 128 KiB/token.
    assert est.bytes_per_token("f16") == 131072
    # Nothing structural to correct: the naive formula's only error here is its
    # double-counted K and V, so it over-charges by exactly 2.
    assert _within(est.overcount_factor(32768, "f16"), 2.0, 0.001)


def test_dense_kv_and_v_are_not_double_counted():
    """The K+V double count was half the old formula's error.

    A layer holds ``kv_heads * key_len`` scalars of K and the same of V — the
    cache is the sum of the two, never one of them times two.
    """
    est = mm.attention_kv_estimate(gguf_meta={
        "llama.block_count": 8,
        "llama.embedding_length": 2048,
        "llama.attention.head_count": 16,
        "llama.attention.head_count_kv": 4,
        "llama.attention.key_length": 64,
        "llama.attention.value_length": 64,
    })

    assert est.bytes_at(1, "f16") == 8 * 4 * (64 + 64) * 2


# ---------------------------------------------------------------------------
# Architecture 3 — sliding window: bounded, not growing
# ---------------------------------------------------------------------------
def _gemma_meta() -> dict:
    """60 layers, 1024-token window every 6th layer (Gemma-3 shaped)."""
    return {
        "general.architecture": "gemma3",
        "gemma3.block_count": 60,
        "gemma3.embedding_length": 5376,
        "gemma3.attention.head_count": 32,
        "gemma3.attention.head_count_kv": 16,
        "gemma3.attention.key_length": 128,
        "gemma3.attention.value_length": 128,
        "gemma3.attention.sliding_window": 1024,
        "gemma3.attention.sliding_window_pattern": 6,
    }


def test_sliding_window_layers_saturate_at_the_window():
    """Sliding layers are bounded by the window, so KV growth flattens.

    Only the 50 global layers grow with the context; the 10 sliding layers
    stop costing once the window fills.
    """
    est = mm.attention_kv_estimate(gguf_meta=_gemma_meta())

    assert est is not None
    assert est.full_layers == 50
    assert est.sliding_layers == 10
    assert est.sliding_window == 1024
    assert est.n_layers == est.full_layers + est.sliding_layers

    per_layer = est.elements_per_layer * 2.0
    # ctx == window: nothing is bounded yet, every layer costs the full width.
    assert est.bytes_at(1024, "f16") == pytest.approx(60 * 1024 * per_layer)
    # Past the window the total is global layers at ctx + sliding layers at the
    # window, and the marginal cost per extra token is the global layers only.
    ctx = 65536
    expected = (50 * ctx + 10 * 1024) * per_layer
    assert est.bytes_at(ctx, "f16") == pytest.approx(expected)
    # 4x the context costs a little under 4x the KV — the sliding layers' share
    # is fixed, so the overcount against the naive model grows with the window.
    assert est.overcount_factor(ctx, "f16") < 4.0
    assert est.overcount_factor(ctx, "f16") > 2.0


def test_sliding_window_zero_is_not_a_window():
    """``sliding_window = 0`` means disabled, not a zero-width window.

    llama.cpp writes 0 for archs that do not use SWA; treating it as a window
    would collapse every layer's cache to nothing.
    """
    meta = _gemma_meta()
    meta["gemma3.attention.sliding_window"] = 0
    est = mm.attention_kv_estimate(gguf_meta=meta)

    assert est.sliding_layers == 0
    assert est.sliding_window == 0
    assert est.full_layers == 60


def test_layer_types_sliding_label_is_bounded_by_the_window():
    """When layer_types names sliding layers explicitly, that label wins."""
    est = mm.attention_kv_estimate(gguf_meta={
        "general.architecture": "qwen3",
        "qwen3.block_count": 8,
        "qwen3.embedding_length": 2048,
        "qwen3.attention.head_count": 16,
        "qwen3.attention.head_count_kv": 4,
        "qwen3.attention.key_length": 128,
        "qwen3.attention.value_length": 128,
        "qwen3.attention.sliding_window": 512,
        "qwen3.attention.layer_types": [
            "sliding_attention", "linear_attention", "sliding_attention",
            "linear_attention", "full_attention", "linear_attention",
            "full_attention", "linear_attention",
        ],
    })

    assert (est.full_layers, est.sliding_layers, est.linear_layers) == (2, 2, 4)
    per_layer = est.elements_per_layer * 2.0
    assert est.bytes_at(4096, "f16") == pytest.approx(
        (2 * 4096 + 2 * 512) * per_layer)


def test_short_layer_types_array_charges_the_unclassified_layers():
    """A truncated header under-counts cache rather than silently dropping it.

    An unclassified layer is charged as full attention — the safe direction,
    since over-reserving VRAM degrades to slow offload, while under-reserving
    OOMs.
    """
    est = mm.attention_kv_estimate(gguf_meta={
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
        "llama.attention.layer_types": ["linear_attention", "linear_attention"],
    })

    assert est.linear_layers == 2
    assert est.full_layers == 30
    assert est.full_layers + est.linear_layers == 32


def test_unparseable_layer_types_placeholder_falls_back_to_dense():
    """An older parse left ``"<array:64>"`` where the label list belongs.

    That marker is not a layer-type list; reading it as one would classify
    every layer as unrecognised-but-present and lose the block count.
    """
    est = mm.attention_kv_estimate(gguf_meta={
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
        "llama.attention.layer_types": "<array:32>",
    })

    assert est is not None
    assert est.full_layers == 32 and est.layer_types == ()


# ---------------------------------------------------------------------------
# KV quantisation arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv_type,expected", [
    ("f16", 2.0), ("f32", 4.0), ("bf16", 2.0),
    ("q8_0", 34.0 / 32.0),
    ("q5_1", 24.0 / 32.0), ("q5_0", 22.0 / 32.0),
    ("q4_1", 20.0 / 32.0), ("q4_0", 18.0 / 32.0),
    ("q6_k", 210.0 / 256.0), ("q5_k", 176.0 / 256.0),
    ("q4_k", 144.0 / 256.0), ("q3_k", 110.0 / 256.0),
    ("q2_k", 84.0 / 256.0),
])
def test_kv_type_bytes_per_element(kv_type, expected):
    """Block bytes over block elements, from each type's real block layout.

    The Q4/Q5/Q6/Q8 legacy quant types pack 32 elements per block with an fp16
    scale (+ mins for q4_1/q5_1); the K-quants use a 256-element super-block.
    """
    assert mm._kv_bytes_per_element(kv_type) == pytest.approx(expected)


def test_unknown_kv_type_is_priced_as_f16():
    """An unrecognised KV type must not invent free headroom.

    f16 is the llama.cpp default and the widest entry in the table, so falling
    back to it over-reserves rather than under-reserves.
    """
    assert mm._kv_bytes_per_element("q7_turbo") == mm._kv_bytes_per_element("f16")
    assert mm._kv_bytes_per_element(None) == 2.0
    assert mm._kv_bytes_per_element("") == 2.0


def test_kv_type_name_is_case_and_space_insensitive():
    """-ctk/-ctv accept mixed case; the lookup should too."""
    est = mm.attention_kv_estimate(gguf_meta=_hybrid_meta())
    assert est.bytes_per_token("Q8_0") == est.bytes_per_token("q8_0")
    assert est.bytes_per_token(" q8_0 ") == est.bytes_per_token("q8_0")


# ---------------------------------------------------------------------------
# Metadata handling / robustness
# ---------------------------------------------------------------------------
def test_estimate_is_none_without_the_metadata_to_price_kv():
    """Callers fall back to "offload everything" rather than guess."""
    assert mm.attention_kv_estimate(gguf_meta={}) is None
    assert mm.attention_kv_estimate(gguf_meta={"llama.block_count": 32}) is None
    # head_count_kv is the field that makes a KV estimate possible at all.
    assert mm.attention_kv_estimate(gguf_meta={
        "llama.block_count": 32, "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
    }) is None


def test_key_length_falls_back_to_embedding_over_heads():
    """key/value length are optional in practice; derive them when absent."""
    est = mm.attention_kv_estimate(gguf_meta={
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
    })

    assert est is not None
    assert est.key_len == 128 and est.val_len == 128


def test_estimate_never_raises_on_a_bogus_path():
    """Best-effort metadata: a missing file is None, not an exception."""
    assert mm.attention_kv_estimate(Path("/nonexistent/model.gguf")) is None


def test_to_dict_reports_the_residency_numbers_t34_needs():
    """The stack validator's per-card plan consumes this dict (ADR-L5)."""
    est = mm.attention_kv_estimate(gguf_meta=_hybrid_meta())
    d = est.to_dict(ctx_size=65536, kv_cache_type="q8_0")

    assert d["architecture"] == "qwen3next"
    assert d["n_layers"] == 64
    assert d["full_attention_layers"] == 16
    assert d["linear_layers"] == 48
    assert d["kib_per_token"] == pytest.approx(34.0)
    assert d["kv_gib"] == pytest.approx(2.12, abs=0.02)
    assert d["overcount_factor"] == pytest.approx(8.0, abs=0.01)
    # JSON-serialisable: the API layer returns it verbatim.
    import json
    assert json.loads(json.dumps(d)) == d


def test_to_dict_without_a_context_omits_the_total_fields():
    est = mm.attention_kv_estimate(gguf_meta=_hybrid_meta())
    d = est.to_dict()
    assert "kv_gib" not in d and "ctx_size" not in d
    assert d["bytes_per_token"] == 65536


# ---------------------------------------------------------------------------
# GGUF header parse — the estimator's only input
# ---------------------------------------------------------------------------
def _write_gguf(path: Path, kv: dict, *, version: int = 3) -> Path:
    """Minimal GGUF header writer: enough for the metadata reader.

    Supports the value types the reader handles that matter here: u32 scalars,
    strings, and string arrays (``attention.layer_types``). GGUF v3 stores
    every length as u64, v2 as u32.
    """
    lp = "<Q" if version >= 3 else "<I"

    def length(n: int) -> bytes:
        return struct.pack(lp, n)

    def s(text: str) -> bytes:
        raw = text.encode()
        return length(len(raw)) + raw

    body = b""
    for key, value in kv.items():
        body += s(key)
        if isinstance(value, bool):
            body += struct.pack("<I", 7) + struct.pack("<B", int(value))
        elif isinstance(value, int):
            body += struct.pack("<I", 4) + struct.pack("<I", value)
        elif isinstance(value, str):
            body += struct.pack("<I", 8) + s(value)
        elif isinstance(value, list):
            body += struct.pack("<I", 9) + struct.pack("<I", 8) + length(len(value))
            for item in value:
                body += s(item)
        else:
            raise TypeError(type(value))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", version))
        fh.write(length(0))            # tensor count
        fh.write(length(len(kv)))      # kv count
        fh.write(body)
    return path


def test_header_parse_recovers_layer_types_from_a_real_gguf(tmp_path):
    """End-to-end: bytes on disk -> layer_types list -> the 64/34 rates.

    The reader must keep the string array rather than collapsing it to a
    ``"<array:N>"`` marker, or hybrid models fall back to the dense estimate.
    """
    model = _write_gguf(tmp_path / "hybrid.gguf", {
        "general.architecture": "qwen3next",
        "general.file_type": 15,
        "qwen3next.context_length": 262144,
        "qwen3next.block_count": 64,
        "qwen3next.embedding_length": 5120,
        "qwen3next.attention.head_count": 40,
        "qwen3next.attention.head_count_kv": 8,
        "qwen3next.attention.key_length": 128,
        "qwen3next.attention.value_length": 128,
        "qwen3next.attention.layer_types": [
            "linear_attention" if i % 4 else "full_attention" for i in range(64)
        ],
        # A later key: the reader must not stop before the plan is complete.
        "qwen3next.rope.dimension_count": 128,
    })

    meta = mm._read_gguf_metadata(model)
    labels = meta["qwen3next.attention.layer_types"]
    assert isinstance(labels, list) and len(labels) == 64
    assert labels.count("full_attention") == 16
    assert meta["context_length"] == 262144
    assert meta["quantization"] == "Q4_K_M"  # general.file_type 15

    est = mm.attention_kv_estimate(model)
    assert est.full_layers == 16 and est.linear_layers == 48
    assert est.bytes_per_token("f16") / KIB == pytest.approx(64.0)
    assert est.bytes_per_token("q8_0") / KIB == pytest.approx(34.0)


def test_header_parse_recovers_sliding_window_from_a_real_gguf(tmp_path):
    model = _write_gguf(tmp_path / "gemma.gguf", {
        "general.architecture": "gemma3",
        "general.file_type": 15,
        "gemma3.block_count": 60,
        "gemma3.embedding_length": 5376,
        "gemma3.attention.head_count": 32,
        "gemma3.attention.head_count_kv": 16,
        "gemma3.attention.key_length": 128,
        "gemma3.attention.value_length": 128,
        "gemma3.attention.sliding_window": 1024,
        "gemma3.attention.sliding_window_pattern": 6,
    })

    meta = mm._read_gguf_metadata(model)
    assert meta["gemma3.attention.sliding_window"] == 1024
    assert meta["gemma3.attention.sliding_window_pattern"] == 6

    est = mm.attention_kv_estimate(model)
    assert (est.full_layers, est.sliding_layers) == (50, 10)
    # The pattern stride: 60 layers, every 6th slides -> 0,6,...,54 = 10.
    assert est.sliding_layers == len(range(0, 60, 6))


def test_header_parse_reads_a_v2_file(tmp_path):
    """GGUF v2 stores lengths as u32 rather than u64; both versions parse.

    Old GGUFs are still in circulation, and a version-guarded reader that
    rejected v2 would return no metadata for them at all.
    """
    model = _write_gguf(tmp_path / "v2.gguf", {
        "general.architecture": "llama",
        "general.file_type": 1,
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
    }, version=2)

    meta = mm._read_gguf_metadata(model)
    assert meta["architecture"] == "llama"
    assert meta["llama.block_count"] == 32
    assert meta["llama.attention.head_count_kv"] == 8
    assert meta["quantization"] == "F16"

    est = mm.attention_kv_estimate(model)
    assert est.full_layers == 32
    assert est.bytes_per_token("f16") == 131072


def test_header_parse_returns_empty_for_non_gguf(tmp_path):
    """Not every file in the library directory is a GGUF."""
    junk = tmp_path / "notes.gguf"
    junk.write_bytes(b"this is not a model")
    assert mm._read_gguf_metadata(junk) == {}
    assert mm.attention_kv_estimate(junk) is None


def test_header_parse_survives_a_truncated_array(tmp_path):
    """A truncated string array must not spin or return garbage labels.

    The length prefix promises more elements than the file holds; the reader
    bails and keeps whatever it parsed before the damage.
    """
    model = tmp_path / "trunc.gguf"
    model.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                      + struct.pack("<Q", 1) + struct.pack("<Q", 5) + b"foo.x")

    assert mm._read_gguf_metadata(model) == {}


def test_header_parse_rejects_an_implausible_key_count(tmp_path):
    """A corrupt count must not drive a multi-million iteration loop."""
    model = tmp_path / "bad.gguf"
    model.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                      + struct.pack("<Q", 10 ** 9))
    assert mm._read_gguf_metadata(model) == {}
