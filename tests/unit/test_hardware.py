"""Unit tests for harness.hardware (sysfs VRAM, visibility, disk)."""

from __future__ import annotations

import sys

from harness.hardware import (
    disk_summary,
    gpu_processes,
    linux_amd_devices,
    parse_visible_indices,
)


def test_parse_visible_indices():
    assert parse_visible_indices(None) is None
    assert parse_visible_indices("") is None
    assert parse_visible_indices("1") == [1]
    assert parse_visible_indices("0,1") == [0, 1]
    assert parse_visible_indices(" 2 , 3 ") == [2, 3]
    # UUIDs / MIG ids are not index lists -> caller falls back to all
    assert parse_visible_indices("GPU-abc") is None


def test_disk_summary_reports_free(tmp_path):
    out = disk_summary({"here": str(tmp_path), "nope": "/does/not/exist"})
    assert "here" in out and out["here"]["free_gb"] > 0
    assert "nope" not in out


def test_gpu_processes_is_a_list():
    assert isinstance(gpu_processes(), list)


def test_linux_amd_devices_have_total_and_free():
    if sys.platform == "win32":
        return
    devices = linux_amd_devices()
    for device in devices:
        assert device["memory_gb"] and device["memory_gb"] > 0
        assert device["free_gb"] is not None
        assert device["free_gb"] <= device["memory_gb"]
        assert "index" in device
