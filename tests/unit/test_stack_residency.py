"""Unit tests for the per-card residency plan (``harness/stack/residency.py``).

Fully synthetic and offline: the "models" are sparse GGUF files written into
``tmp_path`` with a real header, the model library is a fake object exposing
the same ``resolve_model`` / ``models_dir`` / ``list_local`` surface as
``LlamaServerManager``, and the hardware summary is a plain dict.  No GPU, no
network, no GGUF download, and no dependency on another LSC task's files — the
stacks here are built from the frozen ``Stack``/``Tier`` dataclasses.

The reference shapes are the measured ones in ``LOCAL-STACKS.md`` sections 3,
5, 7 and 8: a 2x19.98 GiB box, the desktop card losing ~1.6 GiB, the 27B
hybrid at 34 KiB/token Q8_0, and a 2-tier stack that fits in ~36.8 GiB.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from harness.stack.residency import (
    DESKTOP_COST_FALLBACK_GIB,
    ResidencyPlan,
    plan_residency,
)
from harness.stack.schema import Stack, Tier

GIB = 1024.0 ** 3
KIB = 1024.0


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------
def _write_gguf(path: Path, kv: dict, *, version: int = 3) -> Path:
    """Minimal GGUF header: u32 scalars, strings and string arrays.

    Same shape the reader in ``harness.models`` parses for the real library, so
    the T33 estimator sees a genuine ``layer_types`` array rather than a stub.
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
    with path.open("wb") as handle:
        handle.write(b"GGUF" + struct.pack("<I", version))
        handle.write(length(0))            # tensor count
        handle.write(length(len(kv)))      # kv count
        handle.write(body)
    return path


def _sized(path: Path, gib: float) -> Path:
    """Grow a GGUF to ``gib`` on disk without spending the disk."""
    with path.open("r+b") as handle:
        handle.truncate(int(gib * GIB))
    return path


def _face_meta() -> dict:
    """Qwen3-Next-shaped 27B: 64 layers, every 4th full attention (§5 row 1)."""
    return {
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


def _worker_meta() -> dict:
    """9B/4B shape: 32 layers, 8 full attention (§5 row 2)."""
    return {
        "general.architecture": "qwen3",
        "qwen3.block_count": 32,
        "qwen3.embedding_length": 2560,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128,
        "qwen3.attention.value_length": 128,
        "qwen3.attention.layer_types": [
            "full_attention" if i < 8 else "linear_attention" for i in range(32)
        ],
    }


class FakeLibrary:
    """The ``LlamaServerManager`` surface the plan is allowed to touch."""

    def __init__(self, root: Path, entries: dict[str, Path] | None = None,
                 metadata: dict[str, dict] | None = None) -> None:
        self.models_dir = root
        self._entries = dict(entries or {})
        self._metadata = dict(metadata or {})

    def resolve_model(self, name: str):
        return self._entries.get(name)

    def list_local(self) -> list[dict]:
        return [
            {"file": str(path.relative_to(self.models_dir)),
             "name": path.stem,
             "gguf_metadata": self._metadata.get(path.name)}
            for path in self._entries.values()
        ]


@pytest.fixture()
def library(tmp_path) -> FakeLibrary:
    """A 2-tier library: a 27B Q6_K face and a 4B Q6_K worker (LOCAL-STACKS §7)."""
    root = tmp_path / "gguf"
    entries = {
        "Qwen3.8-27B-UD-Q6_K.gguf": _sized(
            _write_gguf(root / "Qwen3.8-27B-UD-Q6_K.gguf", _face_meta()), 20.47),
        "Qwen3.8-4B-Q6_K.gguf": _sized(
            _write_gguf(root / "Qwen3.8-4B-Q6_K.gguf", _worker_meta()), 3.32),
    }
    return FakeLibrary(root, entries)


def _two_tier(*, face_ctx: int = 131072, worker_ctx: int = 131072,
              worker_pin: str = "HIP_VISIBLE_DEVICES=1",
              spec: dict | None = None) -> Stack:
    """``stacks/peer-2tier.json`` as data: face on both cards, worker on one.

    The window defaults to 128K because that is what fits per card on the
    reference box; :func:`test_the_reference_box_refuses_the_full_256k_face`
    is the §8 default at 256K and why it is not this one.
    """
    return Stack(
        name="peer-2tier",
        tiers=[
            Tier(role="face", repo="unsloth/Qwen3.8-27B-GGUF",
                 file="Qwen3.8-27B-UD-Q6_K.gguf", ctx=face_ctx, ngl=99,
                 backend="vulkan", cache_k="q8_0", cache_v="q8_0", spec=spec,
                 pin="HIP_VISIBLE_DEVICES=0,1", ts="1,1"),
            Tier(role="worker", repo="empero-ai/Qwen3.8-4B-Distill",
                 file="Qwen3.8-4B-Q6_K.gguf", ctx=worker_ctx, ngl=99,
                 backend="vulkan", cache_k="q8_0", cache_v="q8_0",
                 pin=worker_pin),
        ],
    )


def _box(*, display: int | None = 0, total: float = 19.98,
         used: float = 0.0, desktop_used: float = 1.6) -> dict:
    """A ``_hardware_summary()``-shaped 2x RX 7900 XT box (LOCAL-STACKS §3)."""
    def card(index: int, *, is_display: bool) -> dict:
        return {
            "index": index, "backend": "rocm", "name": "AMD GPU",
            "memory_gb": total,
            "used_gb": desktop_used if is_display else used,
            "free_gb": round(total - (desktop_used if is_display else used), 2),
            "display": is_display, "visible": True,
        }

    return {
        "vram_gb": round(2 * total - desktop_used, 2),
        "vram_free_gb": round(2 * total - desktop_used, 2),
        "combined_vram_gb": round(2 * total, 2),
        "available_ram_gb": 30.0,
        "devices": [card(0, is_display=(display == 0)),
                    card(1, is_display=(display == 1))],
    }


def _plan(stack, library, hardware) -> ResidencyPlan:
    return plan_residency(stack, models_manager=library, hardware=lambda: hardware)


# ---------------------------------------------------------------------------
# acceptance: a fitting stack is accepted, an over-budget one is refused
# ---------------------------------------------------------------------------
def test_fitting_two_tier_stack_is_accepted(library):
    """27B face at 128K + 4B worker, split across two 19.98 GiB cards (§3 box)."""
    plan = _plan(_two_tier(), library, _box())

    assert plan.ok is True, plan.warnings
    assert [c.card for c in plan.per_card] == [0, 1]
    for card in plan.per_card:
        assert card.total <= card.budget
    assert not any("over budget" in w for w in plan.warnings)


def test_the_reference_box_refuses_the_full_256k_face(library):
    """§8 prices its 2-tier default against a *box-wide* ~36 GiB budget.

    Per card the same stack does not fit on 2x19.98: the desktop card carries
    1.6 GiB of framebuffers, and card 1 carries half the face *and* the whole
    worker. The plan is per card, so it refuses — the desktop cost is what
    makes the difference.
    """
    wide = _plan(_two_tier(face_ctx=262144), library, _box())
    narrow = _plan(_two_tier(face_ctx=131072), library, _box())

    assert wide.ok is False
    assert narrow.ok is True
    assert wide.per_card[1].total > wide.per_card[1].budget
    # Give the desktop cost back and the same stack still does not fit: card 1
    # is the one over budget, and it does not drive the display.
    headless = _plan(_two_tier(face_ctx=262144), library, _box(display=None))
    assert headless.per_card[1].total > headless.per_card[1].budget


def test_over_budget_stack_is_refused(library):
    """Both tiers on one card: refuses instead of OOMing (ADR-L5)."""
    stack = _two_tier(worker_pin="HIP_VISIBLE_DEVICES=0", face_ctx=262144)
    stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    stack.tiers[0].ts = None

    plan = _plan(stack, library, _box())

    assert plan.ok is False
    card0 = plan.per_card[0]
    assert card0.total > card0.budget
    assert card0.weights > 20.0          # both tiers' weights
    assert any("card 0 is over budget" in w for w in plan.warnings)
    # The refusal is reported, not raised, and the numbers are still returned.
    assert plan.per_card[1].total == 0.0


def test_a_context_that_does_not_fit_is_refused(library):
    """Weights alone fit on both cards; the KV cache is what tips them over."""
    stack = Stack(name="probe", tiers=[Tier(
        role="face", repo="unsloth/Qwen3.8-27B-GGUF",
        file="Qwen3.8-27B-UD-Q6_K.gguf", ctx=65536, cache_k="q8_0",
        cache_v="q8_0", pin="HIP_VISIBLE_DEVICES=0,1", ts="1,1")])

    fits = _plan(stack, library, _box())
    stack.tiers[0].ctx = 8 * 1024 * 1024   # 8M tokens of Q8_0 KV
    bloated = _plan(stack, library, _box())

    assert fits.ok is True, fits.warnings
    assert bloated.ok is False
    assert bloated.per_card[0].kv > 100 * fits.per_card[0].kv
    assert any("card 0 is over budget" in w for w in bloated.warnings)


# ---------------------------------------------------------------------------
# acceptance: the display-card desktop cost is reported
# ---------------------------------------------------------------------------
def test_display_card_desktop_cost_is_reported_and_charged(library):
    """The card driving the display has already lost its framebuffers (§3)."""
    desktop = _plan(_two_tier(), library, _box(display=0))
    headless = _plan(_two_tier(), library, _box(display=None))

    card0_desktop = desktop.per_card[0]
    card0_headless = headless.per_card[0]
    # Same card, same tiers: only the desktop cost differs, and it is reported.
    assert card0_desktop.budget == pytest.approx(card0_headless.budget - 1.6)
    assert any("card 0 drives a display: 1.6 GiB desktop cost" in w
               for w in desktop.warnings)
    # The headless card pays nothing for a display it does not drive.
    assert not any("desktop cost" in w for w in headless.warnings)


def test_display_cost_uses_live_used_gb_and_names_the_card(library):
    """The figure charged is what the card reports used, not a fixed constant."""
    plan = _plan(_two_tier(), library, _box(display=1, desktop_used=2.4))

    charged = [c for c in plan.per_card if c.card == 1][0]
    other = [c for c in plan.per_card if c.card == 0][0]
    assert other.budget > charged.budget
    assert any("card 1 drives a display: 2.4 GiB desktop cost" in w
               for w in plan.warnings)


def test_display_card_without_used_gb_falls_back_to_the_measured_cost(library):
    """A display card that reports no usage still costs its framebuffers."""
    hw = _box(display=0)
    hw["devices"][0].pop("used_gb")
    hw["devices"][0].pop("free_gb")

    plan = _plan(_two_tier(), library, hw)

    headless = _plan(_two_tier(), library, _box(display=None))
    assert plan.per_card[0].budget == pytest.approx(
        headless.per_card[0].budget - DESKTOP_COST_FALLBACK_GIB)
    assert any("reports no used_gb" in w for w in plan.warnings)


def test_budget_reserves_the_per_process_compute_buffer(library):
    """§3: budget = vram_total - desktop cost - per-process compute buffers."""
    plan = _plan(_two_tier(), library, _box(display=None))

    # Card 1 hosts the face (split) and the worker: 1.5 + 0.5 GiB of buffers.
    card1 = [c for c in plan.per_card if c.card == 1][0]
    assert card1.budget == pytest.approx(19.98 - 1.5 - 0.5)
    # Card 0 hosts the face half only: one 27B-class buffer.
    card0 = [c for c in plan.per_card if c.card == 0][0]
    assert card0.budget == pytest.approx(19.98 - 1.5)


# ---------------------------------------------------------------------------
# acceptance: the T33 estimator is what prices KV (no hardcoded layer count)
# ---------------------------------------------------------------------------
def test_kv_comes_from_the_t33_estimator_at_the_measured_rate(library):
    """64 KiB/token f16, 34 KiB/token Q8_0 for the 64-layer/16-full hybrid (§5).

    If the plan carried its own layer count this assertion would move with the
    model; it is pinned to the estimator's numbers for the metadata on disk.
    """
    from harness.models import attention_kv_estimate

    path = library.resolve_model("Qwen3.8-27B-UD-Q6_K.gguf")
    estimate = attention_kv_estimate(path)
    assert estimate.full_layers == 16 and estimate.linear_layers == 48
    assert estimate.bytes_per_token("q8_0") / KIB == pytest.approx(34.0)

    stack = _two_tier(face_ctx=262144)
    stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    stack.tiers[0].ts = None
    plan = _plan(stack, library, _box())

    expected = estimate.bytes_at(262144, "q8_0") / GIB
    assert plan.per_card[0].kv == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(8.5, abs=0.1)      # §5's 256K Q8_0 column


def test_kv_is_charged_at_the_tier_s_own_cache_type(library):
    """-ctk/-ctv are priced as the tier declares them, not as a fixed f16."""
    f16_stack = _two_tier()
    f16_stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    f16_stack.tiers[0].ts = None
    f16_stack.tiers[0].ctx = 65536
    f16_stack.tiers[0].cache_k = f16_stack.tiers[0].cache_v = "f16"

    q8_stack = _two_tier()
    q8_stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    q8_stack.tiers[0].ts = None
    q8_stack.tiers[0].ctx = 65536

    f16 = _plan(f16_stack, library, _box())
    q8 = _plan(q8_stack, library, _box())

    # 64 KiB/token against 34: the same window costs ~1.9x at f16.
    assert f16.per_card[0].kv / q8.per_card[0].kv == pytest.approx(64 / 34, rel=0.01)
    assert q8.per_card[0].kv == pytest.approx(2.12, abs=0.02)   # §5's 64K column


def test_linear_attention_layers_cost_no_kv(library, tmp_path):
    """The same file relabelled with 64 attention layers costs 4x the KV.

    Both libraries hold an identical 20.47 GiB model; only the header's
    ``layer_types`` differ. If the plan sized KV with a constant layer count
    the two answers would be equal, so this is the test that pins ADR-L5.
    """
    def build(root: Path, labels: list[str]) -> FakeLibrary:
        meta = _face_meta()
        meta["qwen3next.attention.layer_types"] = labels
        entry = _sized(_write_gguf(root / "face-27B.gguf", meta), 20.47)
        return FakeLibrary(root, {"face-27B.gguf": entry})

    hybrid = build(tmp_path / "hybrid",
                   ["linear_attention" if i % 4 else "full_attention"
                    for i in range(64)])
    dense = build(tmp_path / "dense", ["full_attention"] * 64)

    def stack() -> Stack:
        return Stack(name="probe", tiers=[Tier(
            role="face", repo="unsloth/Qwen3.8-27B-GGUF", file="face-27B.gguf",
            ctx=65536, cache_k="q8_0", cache_v="q8_0",
            pin="HIP_VISIBLE_DEVICES=0")])

    hybrid_kv = _plan(stack(), hybrid, _box(display=None)).per_card[0].kv
    dense_kv = _plan(stack(), dense, _box(display=None)).per_card[0].kv

    assert hybrid_kv == pytest.approx(2.12, abs=0.02)   # §5's 64K Q8_0 column
    assert dense_kv == pytest.approx(4 * hybrid_kv, rel=0.01)
    assert dense_kv == pytest.approx(8.5, abs=0.05)     # the 8x naive over-charge


def test_thin_gguf_metadata_warns_instead_of_guessing(library, tmp_path):
    """No block_count means no KV price; the plan says so rather than inventing one."""
    root = tmp_path / "thin"
    entry = _sized(_write_gguf(root / "thin.gguf",
                               {"general.architecture": "llama"}), 4.0)
    thin = FakeLibrary(root, {"thin.gguf": entry})
    stack = Stack(name="probe", tiers=[Tier(
        role="worker", repo="x/y", file="thin.gguf", ctx=131072,
        pin="HIP_VISIBLE_DEVICES=0")])

    plan = _plan(stack, thin, _box(display=None))

    assert plan.per_card[0].kv == 0.0
    assert any("no readable KV metadata" in w for w in plan.warnings)


def test_library_metadata_prices_a_model_the_header_cannot(library, tmp_path):
    """The manager's own ``list_local`` metadata is the fallback for a thin file."""
    root = tmp_path / "stub"
    entry = _sized(_write_gguf(root / "stub.gguf",
                               {"general.architecture": "llama"}), 4.0)
    stub = FakeLibrary(root, {"stub.gguf": entry}, {"stub.gguf": _worker_meta()})
    stack = Stack(name="probe", tiers=[Tier(
        role="worker", repo="x/y", file="stub.gguf", ctx=65536,
        cache_k="q8_0", cache_v="q8_0", pin="HIP_VISIBLE_DEVICES=0")])

    plan = _plan(stack, stub, _box(display=None))

    assert plan.per_card[0].kv > 0.0
    assert not any("no readable KV metadata" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# placement: pin, tensor split, missing files
# ---------------------------------------------------------------------------
def test_tensor_split_shares_weights_and_kv(library):
    """-ts 1,1 halves both the weights and the cache across the pinned cards."""
    from harness.models import attention_kv_estimate

    plan = _plan(_two_tier(), library, _box(display=None))
    card0 = [c for c in plan.per_card if c.card == 0][0]
    card1 = [c for c in plan.per_card if c.card == 1][0]

    face = attention_kv_estimate(
        library.resolve_model("Qwen3.8-27B-UD-Q6_K.gguf"))
    worker = attention_kv_estimate(
        library.resolve_model("Qwen3.8-4B-Q6_K.gguf"))

    assert card0.weights == pytest.approx(20.47 / 2, abs=1e-6)
    assert card1.weights == pytest.approx(20.47 / 2 + 3.32, abs=1e-6)
    # The face's cache is halved; the worker's lands entirely on card 1.
    assert card0.kv == pytest.approx(face.bytes_at(131072, "q8_0") * 0.5 / GIB,
                                     abs=1e-6)
    assert card1.kv == pytest.approx(
        (face.bytes_at(131072, "q8_0") * 0.5
         + worker.bytes_at(131072, "q8_0")) / GIB, abs=1e-6)


def test_uneven_tensor_split_follows_the_ts_ratio(library):
    """3:1 puts three quarters of the face on the first pinned card."""
    stack = _two_tier()
    stack.tiers[1].pin = "HIP_VISIBLE_DEVICES=1"
    stack.tiers[0].ts = "3,1"

    plan = _plan(stack, library, _box(display=None))
    card0 = [c for c in plan.per_card if c.card == 0][0]
    card1 = [c for c in plan.per_card if c.card == 1][0]

    assert card0.weights == pytest.approx(20.47 * 0.75, abs=1e-6)
    assert card1.weights == pytest.approx(20.47 * 0.25 + 3.32, abs=1e-6)

    # KV follows the weights, and the worker's own cache lands entirely on 1.
    from harness.models import attention_kv_estimate

    face = attention_kv_estimate(
        library.resolve_model("Qwen3.8-27B-UD-Q6_K.gguf"))
    worker = attention_kv_estimate(
        library.resolve_model("Qwen3.8-4B-Q6_K.gguf"))
    assert card0.kv == pytest.approx(face.bytes_at(131072, "q8_0") * 0.75 / GIB,
                                     abs=1e-6)
    assert card1.kv == pytest.approx(
        (face.bytes_at(131072, "q8_0") * 0.25
         + worker.bytes_at(131072, "q8_0")) / GIB, abs=1e-6)


def test_span_without_a_tensor_split_is_charged_to_one_card_with_a_warning(library):
    """A tier on two cards with no -ts is unplannable; say so, do not guess."""
    stack = _two_tier()
    stack.tiers[0].ts = None

    plan = _plan(stack, library, _box(display=None))
    card0 = [c for c in plan.per_card if c.card == 0][0]
    card1 = [c for c in plan.per_card if c.card == 1][0]

    assert card0.weights == pytest.approx(20.47, abs=1e-6)
    assert card1.weights == pytest.approx(3.32, abs=1e-6)
    assert any("without a -ts that names each of them" in w for w in plan.warnings)


def test_a_tensor_split_naming_the_wrong_cards_is_refused_to_one(library):
    """``-ts 1,1`` against three pinned cards cannot be apportioned."""
    stack = _two_tier()
    stack.tiers[0].ts = "1,1,1"

    plan = _plan(stack, library, _box(display=None))
    card0 = [c for c in plan.per_card if c.card == 0][0]

    assert card0.weights == pytest.approx(20.47, abs=1e-6)
    assert any("without a -ts that names each of them" in w for w in plan.warnings)


def test_a_pin_of_zero_is_a_card_number_not_a_falsey_pin(library):
    """``HIP_VISIBLE_DEVICES=0`` names card 0; an empty check must not skip it."""
    stack = _two_tier(worker_pin="HIP_VISIBLE_DEVICES=0")
    stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    stack.tiers[0].ts = None

    plan = _plan(stack, library, _box(display=None))
    card0 = [c for c in plan.per_card if c.card == 0][0]
    card1 = [c for c in plan.per_card if c.card == 1][0]

    assert card0.weights == pytest.approx(20.47 + 3.32, abs=1e-6)
    assert card1.weights == 0.0


def test_a_pin_without_a_variable_name_still_names_cards(library):
    """``"0"`` is a bare card list as well as ``HIP_VISIBLE_DEVICES=0``."""
    stack = _two_tier(worker_pin="0")
    stack.tiers[0].pin = "0"
    stack.tiers[0].ts = None

    plan = _plan(stack, library, _box(display=None))

    assert [c for c in plan.per_card if c.card == 0][0].weights == pytest.approx(
        20.47 + 3.32, abs=1e-6)
    assert [c for c in plan.per_card if c.card == 1][0].total == 0.0


def test_a_pin_naming_a_hidden_card_falls_back_to_the_visible_ones(library):
    """HIP_VISIBLE_DEVICES names cards by index; a hidden one is not placeable."""
    hw = _box(display=None)
    hw["devices"][1]["visible"] = False
    stack = _two_tier()
    stack.tiers[0].ts = None
    stack.tiers[1].pin = "HIP_VISIBLE_DEVICES=1"

    plan = _plan(stack, library, hw)

    assert [c.card for c in plan.per_card] == [0]
    assert any("not a visible device" in w for w in plan.warnings)


def test_invisible_cards_are_left_out_of_the_plan(library):
    """The plan only ever places a tier on a card it can see."""
    hw = _box(display=None)
    hw["devices"][1]["visible"] = False
    stack = _two_tier()
    stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    stack.tiers[0].ts = None
    stack.tiers[1].pin = "HIP_VISIBLE_DEVICES=0"

    plan = _plan(stack, library, hw)

    assert [c.card for c in plan.per_card] == [0]
    assert plan.per_card[0].weights > 20.0


def test_a_model_missing_from_the_library_is_refused_not_ignored(library):
    """An undownloaded GGUF has no size; the stack is not 'fine', it is unknown."""
    stack = Stack(name="probe", tiers=[Tier(
        role="face", repo="unsloth/Qwen3.8-27B-GGUF",
        file="Qwen3.8-27B-UD-Q6_K.gguf", ctx=32768)])
    stack.tiers.append(Tier(role="worker", repo="empero-ai/Qwen3.8-4B-Distill",
                            file="not-downloaded.gguf", ctx=32768))

    plan = _plan(stack, library, _box())

    assert plan.ok is False
    assert any("not-downloaded.gguf is not in the local model library" in w
               for w in plan.warnings)
    # The tier that did resolve is still priced, so the UI can show progress.
    assert plan.per_card[0].weights > 20.0


def test_vision_projector_and_draft_head_are_charged(library, tmp_path):
    """§7 sizes the face at 20.47 + 1.28 (MTP) + 0.86 (mmproj) GiB."""
    root = library.models_dir
    proj = _sized(_write_gguf(root / "mmproj-F16.gguf", _worker_meta()), 0.86)
    library._entries["mmproj-F16.gguf"] = proj

    stack = _two_tier(spec={"type": "draft-mtp", "n_max": 3})
    stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    stack.tiers[0].ts = None
    stack.tiers[0].mmproj = "mmproj-F16.gguf"

    plan = _plan(stack, library, _box())

    assert plan.per_card[0].weights == pytest.approx(20.47 + 1.2 + 0.86, abs=1e-6)
    assert any("draft head" in w for w in plan.warnings)


def test_non_positive_ctx_falls_back_to_the_llama_default(library):
    """A tier with ctx 0 is priced at 8192, and the plan says so."""
    stack = _two_tier()
    stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    stack.tiers[0].ts = None
    stack.tiers[0].ctx = 0

    plan = _plan(stack, library, _box(display=None))

    from harness.models import attention_kv_estimate

    path = library.resolve_model("Qwen3.8-27B-UD-Q6_K.gguf")
    assert plan.per_card[0].kv == pytest.approx(
        attention_kv_estimate(path).bytes_at(8192, "q8_0") / GIB, abs=1e-6)
    assert any("ctx is 0" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# no host / no GPU
# ---------------------------------------------------------------------------
def test_a_box_with_no_visible_gpu_is_refused(library):
    plan = _plan(_two_tier(), library, {"devices": [], "vram_gb": None})

    assert plan.ok is False
    assert plan.per_card == []
    assert any("no visible GPU devices" in w for w in plan.warnings)


def test_a_failing_hardware_probe_is_refused_not_raised(library):
    def boom():
        raise RuntimeError("rocm-smi missing")

    plan = plan_residency(_two_tier(), models_manager=library, hardware=boom)

    assert plan.ok is False
    assert any("hardware probe failed" in w for w in plan.warnings)


def test_the_real_model_library_is_the_default(library, monkeypatch):
    """``models_manager=None`` means the real ``LlamaServerManager``, not nothing."""
    import harness.models as models_module

    monkeypatch.setattr(models_module, "LlamaServerManager",
                        lambda *a, **k: library)

    plan = plan_residency(_two_tier(), hardware=lambda: _box())

    assert plan.ok is True, plan.warnings
    assert plan.per_card[1].weights > 10.0


def test_an_unavailable_model_library_is_a_warning_not_a_crash(monkeypatch):
    """A host with no model directory still gets a verdict instead of a traceback."""
    import harness.models as models_module

    def boom(*args, **kwargs):
        raise OSError("no models dir")

    monkeypatch.setattr(models_module, "LlamaServerManager", boom)

    plan = plan_residency(_two_tier(), hardware=lambda: _box())

    assert plan.ok is False
    assert any("model library unavailable" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# the §B3 payload
# ---------------------------------------------------------------------------
def test_to_dict_is_the_b3_validate_payload(library):
    """``{ok, per_card:[{card, weights, kv, total, budget}], warnings[]}``."""
    plan = _plan(_two_tier(), library, _box())
    payload = plan.to_dict()

    assert set(payload) == {"ok", "per_card", "warnings"}
    assert payload["ok"] is True
    assert payload["per_card"]
    for row in payload["per_card"]:
        assert set(row) == {"card", "weights", "kv", "total", "budget"}
        assert all(isinstance(v, (int, float)) for v in row.values())
        assert row["total"] == pytest.approx(row["weights"] + row["kv"], abs=1e-3)
    assert json.loads(json.dumps(payload)) == payload


def test_to_dict_survives_a_round_trip_of_an_unsafe_plan(library):
    """Over-budget is data, not an exception: the API returns it as 200 + ok:false."""
    stack = _two_tier(worker_pin="HIP_VISIBLE_DEVICES=0", face_ctx=262144)
    stack.tiers[0].pin = "HIP_VISIBLE_DEVICES=0"
    stack.tiers[0].ts = None

    payload = _plan(stack, library, _box()).to_dict()

    assert payload["ok"] is False
    assert any("over budget" in w for w in payload["warnings"])
    assert json.loads(json.dumps(payload)) == payload
