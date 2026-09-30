"""Tests for moe_tier.  Stdlib only, cross-platform (CI runs Windows too).

Run:  python -m pytest tools/moe_tier/tests -q
"""
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("moe_tier", ROOT / "moe_tier.py")
moe_tier = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("moe_tier", moe_tier)   # dataclasses need the module registered
_spec.loader.exec_module(moe_tier)


# ------------------------------------------------------------------ gguf builder

def _s(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_bool_array(key: str, values: list[bool]) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 7)  # ARRAY of BOOL
    out += struct.pack("<Q", len(values))
    return out + bytes(1 if v else 0 for v in values)


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def write_gguf(path: Path, meta: list[bytes], tensors: list[tuple[str, int]],
               alignment: int = 32) -> None:
    """tensors: (name, data_bytes).  Sizes are multiples of `alignment` so the
    offset table recovers them exactly."""
    header = bytearray()
    header += struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(meta))
    for kv in meta:
        header += kv
    offsets = []
    offset = 0
    for name, nbytes in tensors:
        assert nbytes % alignment == 0
        dims = [nbytes]
        header += _s(name)
        header += struct.pack("<I", 1) + struct.pack("<Q", dims[0])
        header += struct.pack("<I", 0)           # ggml type (unused here)
        header += struct.pack("<Q", offset)
        offsets.append(offset)
        offset += nbytes
    pad = (-len(header)) % alignment
    header += b"\0" * pad
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\0" * offset)


def olmoe_meta(block_count: int = 2, experts: int = 4) -> list[bytes]:
    return [
        _kv_str("general.architecture", "olmoe"),
        _kv_str("general.name", "test olmoe"),
        _kv_u32("olmoe.block_count", block_count),
        _kv_u32("olmoe.context_length", 4096),
        _kv_u32("olmoe.embedding_length", 2048),
        _kv_u32("olmoe.attention.head_count", 16),
        _kv_u32("olmoe.attention.head_count_kv", 16),
        _kv_u32("olmoe.expert_count", experts),
        _kv_u32("olmoe.expert_used_count", 2),
        _kv_u32("general.alignment", 32),
    ]


@pytest.fixture
def tiny_gguf(tmp_path):
    path = tmp_path / "tiny.gguf"
    tensors = [
        ("token_embd.weight", 128),
        ("blk.0.ffn_gate_inp.weight", 64),
        ("blk.0.ffn_gate_exps.weight", 256),
        ("blk.0.ffn_up_exps.weight", 256),
        ("blk.0.ffn_down_exps.weight", 256),
        ("blk.1.ffn_gate_exps.weight", 256),
        ("blk.1.ffn_up_exps.weight", 256),
        ("blk.1.ffn_down_exps.weight", 256),
        ("output.weight", 128),
    ]
    write_gguf(path, olmoe_meta(), tensors)
    return path


def test_read_gguf_metadata_and_sizes(tiny_gguf):
    info = moe_tier.read_gguf(tiny_gguf)
    assert info.arch == "olmoe"
    assert info.block_count == 2
    assert info.context_length == 4096
    assert info.head_count_kv == 16
    assert info.expert_count == 4
    sizes = info.sizes()
    assert sizes["token_embd.weight"] == 128
    assert sizes["blk.0.ffn_gate_exps.weight"] == 256
    assert info.expert_bytes() == 256 * 6
    assert info.expert_bytes_per_layer() == {0: 768, 1: 768}
    assert info.total_bytes() == 128 + 64 + 256 * 6 + 128


def test_read_gguf_rejects_non_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf file at all")
    with pytest.raises(SystemExit):
        moe_tier.read_gguf(bad)


def test_read_gguf_captures_recurrent_layers(tmp_path):
    path = tmp_path / "hy.gguf"
    meta = olmoe_meta() + [_kv_bool_array("olmoe.attention.recurrent_layers",
                                          [True, False, True, False])]
    write_gguf(path, meta, [("token_embd.weight", 128)])
    info = moe_tier.read_gguf(path)
    assert info.recurrent_layers == [True, False, True, False]


# -------------------------------------------------------------------- devices

ROCM = """ggml_cuda_init: found 2 ROCm devices (Total VRAM: 40928 MiB):
  Device 0: AMD Radeon RX 7900 XT, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 20464 MiB
Available devices:
  ROCm0: AMD Radeon RX 7900 XT (20464 MiB, 20404 MiB free)
  ROCm1: AMD Radeon RX 7900 XT (20464 MiB, 20428 MiB free)"""

CUDA = """Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 23402 MiB free)"""

METAL = """ggml_metal_init: picking default device: Apple M2 Max
Available devices:
  Metal: Apple M2 Max (10922 MiB, 10922 MiB free)"""


def test_parse_devices_rocm_cuda_metal():
    d = moe_tier.parse_devices(ROCM)
    assert [(x.backend, x.index, x.total_mib) for x in d] == \
        [("ROCm", 0, 20464), ("ROCm", 1, 20464)]
    assert d[0].free_mib == 20404

    d = moe_tier.parse_devices(CUDA)
    assert d[0].backend == "CUDA" and d[0].total_mib == 24564

    d = moe_tier.parse_devices(METAL)
    assert d[0].backend == "Metal" and d[0].name == "Apple M2 Max"


def test_physical_cores_sane():
    n, src = moe_tier.physical_cores()
    assert n is not None and n >= 1 and src


# ----------------------------------------------------------------------- plan

def _dev(n=2, total=20464):
    return [moe_tier.Device("ROCm", i, f"GPU {i}", total, total - 60)
            for i in range(n)]


def _info(tmp_path, *, expert_mib_per_layer=64, layers=16, non_expert_mib=512,
          ctx=4096, head_kv=16, head=16, emb=2048, key_length=None,
          recurrent=None) -> moe_tier.GGUFInfo:
    tensors = []
    per_tensor_mib = max(1, expert_mib_per_layer // 3)
    for layer in range(layers):
        for proj in ("gate", "up", "down"):
            tensors.append(moe_tier.TensorInfo(
                f"blk.{layer}.ffn_{proj}_exps.weight", [1], 0, 0,
                per_tensor_mib * 1024 * 1024))
    tensors.append(moe_tier.TensorInfo(
        "token_embd.weight", [1], 0, 0, non_expert_mib * 1024 * 1024))
    return moe_tier.GGUFInfo(
        path=tmp_path / "m.gguf", arch="olmoe", name="m",
        block_count=layers, context_length=ctx, embedding_length=emb,
        head_count=head, head_count_kv=head_kv, key_length=key_length,
        expert_count=64, expert_used_count=8,
        tensors=tensors, data_start=0, file_size=0,
        recurrent_layers=list(recurrent) if recurrent else None)


def test_plan_fits_single_gpu(tmp_path):
    info = _info(tmp_path, expert_mib_per_layer=64, layers=16,
                 non_expert_mib=512)
    p = moe_tier.make_plan(info, _dev(2), context=4096, threads=8)
    assert p.n_cpu_moe == 0
    assert p.split_mode == "none"
    assert p.est_vram_mib < 20464


def test_plan_spills_to_cpu(tmp_path):
    # 16 layers x 800 MiB experts = 12.8 GiB; 4 GiB card can hold a few layers
    info = _info(tmp_path, expert_mib_per_layer=800, layers=16,
                 non_expert_mib=512)
    p = moe_tier.make_plan(info, _dev(1, total=4096), context=4096, threads=8)
    assert 0 < p.n_cpu_moe < 16
    assert p.expert_layers_on_gpu == 16 - p.n_cpu_moe
    assert p.split_mode == "none"


def test_plan_non_expert_alone_too_big(tmp_path):
    info = _info(tmp_path, expert_mib_per_layer=10, layers=16,
                 non_expert_mib=8000)
    p = moe_tier.make_plan(info, _dev(1, total=4096), context=4096, threads=8)
    assert p.n_cpu_moe == 16
    assert any("do not fit" in n for n in p.notes)


def test_plan_vram_budget_override_simulates_other_machines(tmp_path):
    info = _info(tmp_path)
    full = moe_tier.make_plan(info, _dev(2), context=4096, threads=8)
    small = moe_tier.make_plan(info, _dev(2), context=4096, threads=8,
                               vram_budget_mib=2048)
    assert full.n_cpu_moe == 0
    assert small.n_cpu_moe > full.n_cpu_moe


def test_plan_commands(tmp_path):
    info = _info(tmp_path, expert_mib_per_layer=800, layers=16)
    p = moe_tier.make_plan(info, _dev(1, total=4096), context=2048, threads=6)
    cmd = p.server_cmd("/eng", "127.0.0.1", 8090)
    assert "/eng/llama-server" in cmd and "-ncmoe" in cmd and "-t 6" in cmd
    assert "--split-mode none" in cmd
    bench = p.bench_cmd("/eng")
    assert "llama-bench" in bench and "-p 512" in bench


def test_plan_gpu_selection(tmp_path):
    info = _info(tmp_path, expert_mib_per_layer=800, layers=16)
    p = moe_tier.make_plan(info, _dev(2), context=4096, threads=8, gpus=[1])
    assert len(p.gpus) == 1 and p.gpus[0].index == 1


# -------------------------------------------------- KV: hybrid geometry -----

def test_kv_growing_only_on_non_recurrent_layers(tmp_path):
    rec = [True, False] * 8                       # 16 layers, 8 KV layers
    info = _info(tmp_path, layers=16, head_kv=2, head=16, emb=2048,
                 key_length=256, recurrent=rec)
    p = moe_tier.make_plan(info, _dev(1), context=4096, threads=8)
    assert p.kv_layers == 8 and p.kv_recurrent_layers == 8
    want = 2 * 8 * 2 * 256 * 2.0 * 4096 / 1024 / 1024
    assert abs(p.kv_mib - want) < 1e-6
    # the all-layers assumption would count double
    info2 = _info(tmp_path, layers=16, head_kv=2, head=16, emb=2048,
                  key_length=256)
    p2 = moe_tier.make_plan(info2, _dev(1), context=4096, threads=8)
    assert abs(p2.kv_mib - 2 * want) < 1e-6


def test_kv_recurrent_state_from_ssm_metadata(tmp_path):
    rec = [True, False] * 8
    info = _info(tmp_path, layers=16, key_length=256, recurrent=rec)
    info.metadata = {"olmoe.ssm.state_size": 128, "olmoe.ssm.group_count": 16,
                     "olmoe.ssm.time_step_rank": 32, "olmoe.ssm.conv_kernel": 4}
    p = moe_tier.make_plan(info, _dev(1), context=4096, threads=8)
    key_dim, value_dim = 128 * 16, 128 * 32
    elems = (key_dim + value_dim) * 3 + 128 * 128 * 32
    want = 8 * elems * 4 / 1024 / 1024
    assert abs(p.kv_state_mib - want) < 0.051
    # fixed size: context growth changes KV, not the recurrent state
    p2 = moe_tier.make_plan(info, _dev(1), context=8192, threads=8)
    assert p2.kv_state_mib == p.kv_state_mib
    assert p2.kv_mib > p.kv_mib


def test_kv_recurrent_state_missing_metadata_is_flagged(tmp_path):
    info = _info(tmp_path, layers=16, recurrent=[True, False] * 8)
    p = moe_tier.make_plan(info, _dev(1), context=4096, threads=8)
    assert p.kv_state_mib == 0.0
    assert any("ssm.* metadata" in n for n in p.notes)


def test_kv_quality_and_byte_model_measured(tmp_path):
    # the byte rates and the measured PPL deltas are pinned (release, 50x512)
    assert moe_tier.KV_BYTES_PER_ELEM == {"f16": 2.0, "bf16": 2.0, "f32": 4.0,
                                          "q8_0": 1.0625, "q4_0": 0.5625}
    assert moe_tier.KV_QUALITY_DELTA_PCT["q8_0"] == -0.036
    assert moe_tier.KV_QUALITY_DELTA_PCT["q4_0"] == 0.240
    info = _info(tmp_path)
    p16 = moe_tier.make_plan(info, _dev(1), context=4096, threads=8)
    p4 = moe_tier.make_plan(info, _dev(1), context=4096, threads=8,
                            kv_type="q4_0")
    assert p4.kv_mib < p16.kv_mib
    assert p4.kv_quality_delta_pct == 0.240
    assert any("KV quality" in n and "q4_0" in n for n in p4.notes)
    # the byte model scales exactly with bytes-per-element (1 decimal rounding)
    assert abs(p4.kv_mib * 2.0 / 0.5625 - p16.kv_mib) < 0.2


# ------------------------------------------------------------------------ cli

def test_cli_plan_json(tiny_gguf, tmp_path):
    devices = tmp_path / "devices.json"
    devices.write_text(json.dumps([
        {"backend": "ROCm", "index": 0, "name": "GPU", "total_mib": 20464,
         "free_mib": 20400}]))
    out = tmp_path / "plan.json"
    res = subprocess.run(
        [sys.executable, str(ROOT / "moe_tier.py"), "plan",
         "--model", str(tiny_gguf), "--devices-json", str(devices),
         "--context", "512", "--threads", "4", "--json", str(out)],
        capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    plan = json.loads(out.read_text())
    assert plan["n_cpu_moe"] == 0
    assert plan["split_mode"] == "none"
    assert plan["arch"] == "olmoe"


def test_cli_plan_simulated_small_card(tiny_gguf, tmp_path):
    devices = tmp_path / "devices.json"
    devices.write_text(json.dumps([
        {"backend": "CUDA", "index": 0, "name": "GPU", "total_mib": 20464,
         "free_mib": 20400}]))
    res = subprocess.run(
        [sys.executable, str(ROOT / "moe_tier.py"), "plan",
         "--model", str(tiny_gguf), "--devices-json", str(devices),
         "--vram-budget-mib", "128", "--threads", "4"],
        capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "ncmoe" in res.stdout or "n_cpu_moe" in res.stdout
