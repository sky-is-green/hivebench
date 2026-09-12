"""Unit tests for cortex.drone_pool (S3.4)."""

from cortex.drone_pool import DronePool


def _factory():
    state = {"n": 0}

    def make():
        state["n"] += 1
        return {"id": state["n"]}

    return make


def test_starts_with_one_instance():
    pool = DronePool(drone_factory=_factory())
    assert len(pool.instances) == 1


def test_scale_up_with_load_and_vram():
    pool = DronePool(max_instances=3, drone_factory=_factory())
    count = pool.scale_if_needed(queue_depth=30, available_vram_mb=2000)
    assert count == 3


def test_scale_capped_by_vram():
    pool = DronePool(max_instances=3, drone_factory=_factory())
    count = pool.scale_if_needed(queue_depth=30, available_vram_mb=400)
    assert count == 1  # no headroom to add instances


def test_scale_capped_by_max_instances():
    pool = DronePool(max_instances=2, drone_factory=_factory())
    count = pool.scale_if_needed(queue_depth=100, available_vram_mb=5000)
    assert count == 2


def test_scale_down_when_idle():
    pool = DronePool(max_instances=3, drone_factory=_factory())
    pool.scale_if_needed(queue_depth=30, available_vram_mb=2000)
    assert len(pool.instances) == 3
    count = pool.scale_if_needed(queue_depth=1, available_vram_mb=2000)
    assert count == 1