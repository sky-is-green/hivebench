"""Host hardware introspection for fit estimates and GPU selection.

Linux/AMD is sysfs-primary (this host has no ``rocm-smi``/``amdsmi``): per-card
total and **free** VRAM, GTT (the CPU-offload budget on AMD), PCI BDF, driver
and whether a display connector is attached. Per-process VRAM comes from
``/proc/<pid>/fdinfo`` (``drm-total-vram``), which is how the desktop tools
attribute usage.

Deliberately no thermal/power/clock polling: that is real-time telemetry for a
dedicated monitor, not something the harness should spend cycles on.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Optional


def parse_visible_indices(value: Optional[str]) -> Optional[list[int]]:
    """Parse ``HIP_VISIBLE_DEVICES``/``CUDA_VISIBLE_DEVICES`` into indices.

    None when unset (all devices visible) or when the value is not a plain
    index list (UUIDs/MIG), so callers fall back to "all visible"."""
    if value is None or str(value).strip() == "":
        return None
    tokens = [t.strip() for t in str(value).split(",") if t.strip() != ""]
    if not tokens:
        return None
    indices: list[int] = []
    for token in tokens:
        if not token.lstrip("-").isdigit():
            return None
        indices.append(int(token))
    return indices


def _gb(value) -> Optional[float]:
    try:
        return round(int(value) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return None


def linux_amd_devices() -> list[dict]:
    """Enumerate amdgpu cards from sysfs.

    Sorted by PCI BDF ascending — the order ROCm/HIP assigns device indices,
    which is why ``HIP_VISIBLE_DEVICES=1`` selects the card whose BDF sorts
    second (0000:07:00.0 here, not sysfs card0)."""
    devices: list[dict] = []
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        card = dev.split("/")[-2]

        def read(name: str) -> Optional[str]:
            try:
                with open(f"{dev}/{name}") as handle:
                    return handle.read().strip()
            except OSError:
                return None

        total = read("mem_info_vram_total")
        if not total:
            continue
        bdf = driver = None
        try:
            with open(f"{dev}/uevent") as handle:
                for line in handle:
                    if line.startswith("PCI_SLOT_NAME="):
                        bdf = line.split("=", 1)[1].strip()
                    elif line.startswith("DRIVER="):
                        driver = line.split("=", 1)[1].strip()
        except OSError:
            pass
        display = False
        for status_path in glob.glob(f"/sys/class/drm/{card}-*/status"):
            try:
                with open(status_path) as handle:
                    if handle.read().strip() == "connected":
                        display = True
                        break
            except OSError:
                continue
        busy = read("gpu_busy_percent")
        vendor, device_id = read("vendor"), read("device")
        mem = _gb(total)
        used = _gb(read("mem_info_vram_used"))
        devices.append({
            "card": card,
            "bdf": bdf,
            "driver": driver,
            "vendor": vendor,
            "device": device_id,
            "backend": "rocm",
            "name": f"AMD GPU {vendor or ''}:{device_id or ''}".strip(),
            "memory_gb": mem,
            "used_gb": used,
            "free_gb": round(max(0.0, mem - used), 2)
            if mem is not None and used is not None else None,
            "gtt_total_gb": _gb(read("mem_info_gtt_total")),
            "gtt_used_gb": _gb(read("mem_info_gtt_used")),
            "util_pct": int(busy) if busy and busy.isdigit() else None,
            "display": display,
        })
    devices.sort(key=lambda d: d.get("bdf") or "")
    for index, device in enumerate(devices):
        device["index"] = index
    return devices


def _fdinfo_vram_bytes(pid: int) -> int:
    """Sum ``drm-total-vram`` across a process's fds (bytes; fdinfo is KiB)."""
    total_kib = 0
    try:
        for path in glob.glob(f"/proc/{pid}/fdinfo/*"):
            try:
                with open(path) as handle:
                    for line in handle:
                        if line.startswith("drm-total-vram:"):
                            total_kib += int(line.split(":", 1)[1].strip().split()[0])
            except (OSError, ValueError):
                continue
    except Exception:  # noqa: BLE001 - process may vanish
        return 0
    return total_kib * 1024


def gpu_processes() -> list[dict]:
    """Processes holding GPU memory, ranked by VRAM (on-demand; not polled)."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pid = int(proc.info["pid"])
            used = _fdinfo_vram_bytes(pid)
            if used <= 0:
                continue
            rows.append({"pid": pid, "name": proc.info.get("name") or "",
                         "vram_mb": round(used / (1024 ** 2), 1)})
        except Exception:  # noqa: BLE001 - process vanished
            continue
    rows.sort(key=lambda r: r["vram_mb"], reverse=True)
    return rows


def disk_summary(paths: dict[str, str]) -> dict:
    """Free/total GB per labelled path, skipping unreadable ones."""
    out: dict[str, dict] = {}
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return out
    for label, path in paths.items():
        try:
            usage = psutil.disk_usage(path)
            out[label] = {"free_gb": round(usage.free / (1024 ** 3), 2),
                          "total_gb": round(usage.total / (1024 ** 3), 2)}
        except Exception:  # noqa: BLE001
            continue
    return out
