"""T2 — rotation definition, orthogonality, and absorbed-model equivalence.

Acceptance (HIVE-PLAN.md §5): `R·Rᵀ = I` (≤1e-5); absorbed output ==
reference output on a random two-layer stack (≤1e-4); handles 5120/10240.
Synthetic tensors only — no GPU, no downloads.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from experiments.ternary import rotation as rot

SPEC_PATH = Path(__file__).resolve().parents[2] / "experiments" / "ternary" / "spec.md"
SPEC_SHA256 = "9fe182ad37729ed730442d10e5e6184e14287acd4985ce1cc9cac9157de9463b"

SEED = 1337


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    return json.loads(blocks[0])


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == rot.SPEC_SHA256


@pytest.mark.parametrize(
    "d,expected",
    [
        (5120, 1024),
        (10240, 1024),
        (2048, 1024),
        (4096, 1024),
        (768, 256),
        (128, 128),
        (96, 32),
        (6, 2),
        (3, 1),
        (1, 1),
    ],
)
def test_block_size_rule(d: int, expected: int) -> None:
    assert rot.block_size(d) == expected


def test_block_size_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        rot.block_size(0)
    with pytest.raises(ValueError):
        rot.block_size(-1024)


@pytest.mark.parametrize("g", [2, 4, 8, 16, 128])
def test_hadamard_is_orthonormal_and_symmetric(g: int) -> None:
    h = rot.hadamard(g)
    assert h.shape == (g, g)
    assert np.allclose(h, h.T)
    assert np.allclose(h @ h.T, np.eye(g), atol=1e-12)
    assert np.isclose(np.abs(h).max(), 1 / np.sqrt(g))


def test_hadamard_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError):
        rot.hadamard(3)
    with pytest.raises(ValueError):
        rot.hadamard(0)


@pytest.mark.parametrize("g", [1, 2, 128, 1024])
def test_sign_vectors_are_pm1_and_deterministic(g: int) -> None:
    s1 = rot.sign_vector(SEED, "hidden", 0, g)
    s2 = rot.sign_vector(SEED, "hidden", 0, g)
    assert set(np.unique(s1).tolist()) <= {-1.0, 1.0}
    assert np.array_equal(s1, s2)
    assert s1.shape == (g,)


def test_signs_vary_across_blocks_domains_and_seeds() -> None:
    a = rot.sign_vector(SEED, "hidden", 0, 128)
    b = rot.sign_vector(SEED, "hidden", 1, 128)
    c = rot.sign_vector(SEED, "head", 0, 128)
    d = rot.sign_vector(SEED + 1, "hidden", 0, 128)
    for other in (b, c, d):
        assert not np.array_equal(a, other)


@pytest.mark.parametrize("d", [128, 1024])
def test_materialized_rotation_is_orthogonal(d: int) -> None:
    r = rot.materialize_rotation(d, SEED)
    assert np.allclose(r @ r.T, np.eye(d), atol=1e-5)
    assert np.allclose(r.T @ r, np.eye(d), atol=1e-5)


def test_apply_matches_materialized_single_block() -> None:
    d = 128
    r = rot.materialize_rotation(d, SEED)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, d))
    assert np.allclose(rot.apply_rotation(x, rot.rotations_for(d, SEED)), x @ r.T, atol=1e-12)


def test_apply_transpose_is_inverse_multiblock() -> None:
    d = 5120
    rots = rot.rotations_for(d, SEED)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, d))
    back = rot.apply_rotation(rot.apply_rotation(x, rots), rots, transpose=True)
    assert np.allclose(back, x, atol=1e-5)
    assert np.isclose(np.linalg.norm(rot.apply_rotation(x, rots)), np.linalg.norm(x), rtol=1e-12)


def test_apply_rejects_non_multiple_dim() -> None:
    x = np.zeros((2, 100))
    with pytest.raises(ValueError):
        rot.apply_rotation(x, rot.rotations_for(128, SEED))


def test_absorb_input_output_identities() -> None:
    d = 1024
    rots = rot.rotations_for(d, SEED)
    r = rot.materialize_rotation(d, SEED)
    rng = np.random.default_rng(2)
    w = rng.standard_normal((32, d))
    assert np.allclose(rot.absorb_input(w, rots), w @ r.T, atol=1e-12)
    w2 = rng.standard_normal((d, 32))
    assert np.allclose(rot.absorb_output(w2, rots), r @ w2, atol=1e-12)
    # R and Rᵀ differ when S is non-trivial: catching the classic absorption bug.
    w3 = rng.standard_normal((d, d))
    assert not np.allclose(rot.absorb_input(w3, rots), rot.absorb_output(w3, rots))


@pytest.mark.parametrize("d", [5120, 10240])
def test_absorbed_two_layer_stack_equivalence(d: int) -> None:
    """Output-rotated residual + input-absorbed linears == reference model."""
    rots = rot.rotations_for(d, SEED)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, d)) * 0.5
    w1 = rng.standard_normal((256, d)) * 0.02
    b1 = rng.standard_normal(256) * 0.01
    w2 = rng.standard_normal((d, 256)) * 0.02
    b2 = rng.standard_normal(d) * 0.01
    w_lm = rng.standard_normal((32, d)) * 0.02

    def silu(v: np.ndarray) -> np.ndarray:
        return v / (1.0 + np.exp(-v))

    def reference() -> np.ndarray:
        z = silu(x @ w1.T + b1)
        h = z @ w2.T + b2
        return rot.rms_norm(h) @ w_lm.T

    x_rot = rot.apply_rotation(x, rots)
    z_rot = silu(x_rot @ rot.absorb_input(w1, rots).T + b1)
    h_rot = z_rot @ rot.absorb_output(w2, rots).T + rot.apply_rotation(b2, rots)
    logits = rot.rms_norm(h_rot) @ rot.absorb_input(w_lm, rots).T

    assert h_rot.shape == (3, d)
    assert not np.allclose(h_rot, z_rot @ w2.T + b2), "rotation must actually rotate"
    assert np.allclose(logits, reference(), atol=1e-4)


def test_rms_norm_commutes_with_rotation() -> None:
    d = 5120
    rots = rot.rotations_for(d, SEED)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, d)) * 3.0
    assert np.allclose(rot.rms_norm(rot.apply_rotation(x, rots)), rot.apply_rotation(rot.rms_norm(x), rots), atol=1e-12)


def test_norm_scale_folding_equivalence() -> None:
    d = 5120
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, d))
    gamma = rng.uniform(0.5, 1.5, size=d)
    w = rng.standard_normal((32, d)) * 0.02
    fused = rot.rms_norm(x) @ rot.fold_norm_scale(w, gamma).T
    plain = (rot.rms_norm(x) * gamma) @ w.T
    assert np.allclose(fused, plain, atol=1e-12)


def test_absorb_shapes_are_validated() -> None:
    rots = rot.rotations_for(1024, SEED)
    with pytest.raises(ValueError):
        rot.absorb_input(np.zeros((4, 100)), rots)
    with pytest.raises(ValueError):
        rot.absorb_output(np.zeros((100, 4)), rots)


def test_empty_dimension_is_identity() -> None:
    x = np.zeros((2, 0))
    assert rot.apply_rotation(x, [np.ones(1)]).shape == (2, 0)
