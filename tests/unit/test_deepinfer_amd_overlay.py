"""Unit tests for the AMD overlay of the vendored DeepInfer (``vendor/deepinfer_amd``).

No GPU and no DeepInfer: the overlay's hardware parsing, its platform
object, and the torch-native replacements for the CUDA-only packages are
pure enough to check on the CPU.

What is under test:

* the KFD graphics-architecture decoding (the numbers are the ones this
  host's ``/sys/class/kfd`` reports);
* the FP8 capability rule (RDNA3 has none, CDNA3 and RDNA4 do);
* the platform object: it is neither the CUDA platform nor the Ascend one;
* the bf16 paths of the replacements, and the refusal of the FP8/FP4 ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT / "vendor"
for _extra in (str(VENDOR), str(VENDOR / "deepinfer")):
    if Path(_extra).is_dir() and _extra not in sys.path:
        sys.path.insert(0, _extra)

# The overlay lives under vendor/, which this repo deliberately does not commit
# (the claims gate rejects vendor/**).  So on a clean clone - CI - the import
# must fail *quietly*: a module-level pytest.skip/importorskip raises
# BaseException, which the import sweep does not catch, turning the whole sweep
# red.  A plain try/except + a module marker keeps the module importable and
# skips only these tests.
_IMPORT_ERROR: str | None = None
try:
    import torch  # noqa: F401
    import deepinfer_amd  # noqa: E402
    from deepinfer_amd import hardware as amd_hardware  # noqa: E402
    from deepinfer_amd.platform import PlatformAMD  # noqa: E402
except ImportError as exc:  # torch absent, or vendor/deepinfer_amd not checked out
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"DeepInfer AMD overlay unavailable ({_IMPORT_ERROR})")

# The overlay registers the torch-native replacements for the CUDA-only
# packages (deep_gemm, tile_kernels, ...).  On a non-AMD host the install
# is a no-op, so the replacements are registered directly here.
if _IMPORT_ERROR is None:
    deepinfer_amd.install()
    if "deep_gemm" not in sys.modules:
        deepinfer_amd._install_op_shims()


# ---------------------------------------------------------------------------
# the KFD graphics architecture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "version,expected",
    [
        (110000, "gfx1100"),   # RDNA3 (this host)
        (120000, "gfx1200"),   # RDNA4
        (103202, "gfx1032"),   # RDNA2
        (94200, "gfx942"),     # CDNA3
        (0, ""),
    ],
)
def test_gfx_arch_decodes_the_kfd_version(version: int, expected: str) -> None:
    assert amd_hardware.gfx_arch(version) == expected


# ---------------------------------------------------------------------------
# the FP8 capability rule
# ---------------------------------------------------------------------------
def _gpu(gfx: str) -> dict:
    return {"node": 1, "gfx": gfx, "gfx_target_version": 0, "simd_count": 168}


def test_supports_fp8_is_false_on_rdna3(monkeypatch) -> None:
    monkeypatch.setattr(amd_hardware, "gpu_nodes", lambda **_kw: [_gpu("gfx1100")])
    assert amd_hardware.supports_fp8() is False


def test_supports_fp8_is_true_on_cdna3_and_rdna4(monkeypatch) -> None:
    for gfx in ("gfx942", "gfx950", "gfx1200"):
        monkeypatch.setattr(amd_hardware, "gpu_nodes", lambda **_kw: [_gpu(gfx)])
        assert amd_hardware.supports_fp8() is True


# ---------------------------------------------------------------------------
# the platform
# ---------------------------------------------------------------------------
def test_the_amd_platform_is_neither_cuda_nor_ascend() -> None:
    platform = PlatformAMD()
    assert platform.is_gpu is False
    assert platform.is_npu is False
    assert platform.is_amd is True
    assert platform.tilelang_ops_backend == "amd"
    # ROCm's PyTorch exposes RCCL under torch's "nccl" backend name.
    assert platform.distributed_backend == "nccl"


def test_the_amd_platform_keeps_the_gpu_scale_factor_dtype() -> None:
    # gfx1100 has no FP8, but the scale-factor *dtype* stays the 32-bit GPU
    # layout so the code that views a scale tensor keeps working.
    assert PlatformAMD().fp8_sf_type is torch.int32


# ---------------------------------------------------------------------------
# the replacements for the CUDA-only packages
# ---------------------------------------------------------------------------
def test_the_bf16_gemm_matches_the_reference() -> None:
    import deep_gemm

    a = torch.randn(7, 16, dtype=torch.bfloat16)
    b = torch.randn(32, 16, dtype=torch.bfloat16)   # (out, in)
    out = torch.empty(7, 32, dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(a, b, out, None)
    assert torch.allclose(out, (a @ b.t()).to(torch.bfloat16), atol=2e-2)


def test_the_fp8_paths_refuse_with_a_reason() -> None:
    import deep_gemm

    with pytest.raises(NotImplementedError, match="FP8"):
        deep_gemm.fp8_fp4_gemm_nt(None, None, None, None)


def test_the_swizzle_needs_no_fp8() -> None:
    import tile_kernels

    with pytest.raises(NotImplementedError, match="FP8"):
        tile_kernels.quant.per_token_cast(torch.zeros(2, 2))


def test_the_roty_applies_the_cos_sin_cache_in_place() -> None:
    import tile_kernels.transform as transform

    # One position, one head, rotary_dim 2: the cache holds the cosine in
    # its first half and the sine in its second, so [[0, 1]] is a 90-degree
    # rotation (cos = 0, sin = 1).
    cache = torch.tensor([[0.0, 1.0]])
    query = torch.tensor([[[1.0, 0.0]]])
    positions = torch.tensor([0])
    transform.apply_rotary(query, cache, None, positions=positions, interleaved=True)
    assert torch.allclose(query[0, 0], torch.tensor([0.0, 1.0]), atol=1e-6)

    # The NeoX layout rotates the two halves instead of adjacent pairs.
    cache = torch.tensor([[1.0, 0.0]])          # cos = 1, sin = 0: identity
    query = torch.tensor([[[1.0, 0.0]]])
    transform.apply_rotary(query, cache, None, positions=positions, interleaved=False)
    assert torch.allclose(query[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_the_moe_gate_selects_and_normalises() -> None:
    import tile_kernels.moe as moe

    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    idx, weights = moe.moe_topk_gate(logits, num_topk=2, routed_scaling_factor=1.0)
    assert idx.tolist() == [[3, 2]]
    assert torch.allclose(weights.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)


def test_the_select_topk_honours_the_in_place_contract() -> None:
    from deep_select.interface import topk

    scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]])
    out = torch.empty(1, 2, dtype=torch.int32)
    values, indices = topk(scores, 2, sorted=False, indices_type=torch.int32,
                            output_idx=out, return_value=True)
    assert indices.dtype is torch.int32
    assert set(indices.flatten().tolist()) == {1, 2}
    assert out.tolist() == indices.tolist()
    assert values.shape == (1, 2)


def test_the_ep_buffer_reports_the_local_alignment() -> None:
    import deep_ep

    assert deep_ep.get_num_allocation_alignment() > 0
    assert deep_ep.get_num_rdma_alignment() > 0
