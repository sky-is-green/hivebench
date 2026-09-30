#!/usr/bin/env python3
"""moe_tier — pick a MoE placement plan for any llama.cpp build and any machine.

Why this exists
---------------
Big MoE models are served by splitting them across VRAM, RAM/CPU and (in the
systems this is modelled on) SSD.  llama.cpp already gives us the levers:

  * which layers run on the GPU            (-ngl, --split-mode, --main-gpu)
  * which layers' *experts* stay on the CPU (-ncmoe / --n-cpu-moe)
  * how many CPU threads the tail gets      (-t)

What it does not give us is the arithmetic: *how many* expert layers must go to
the CPU so the model + KV + buffers fit the VRAM you actually have.  This tool
does that arithmetic from two portable inputs:

  1. the engine's own device list  (`llama-bench --list-devices`) — the same
     output shape for ROCm, CUDA, Metal, Vulkan and CPU-only builds, and
  2. the model's GGUF header (tensor table + metadata) — no weights are read.

Nothing here is tied to a fork, an OS or a vendor.  Point it at any recent
llama.cpp binary on any machine and it produces a placement + ready commands.

Usage
-----
    python moe_tier.py probe  --engine-dir /path/to/bin
    python moe_tier.py plan   --model model.gguf --engine-dir /path/to/bin
    python moe_tier.py plan   --model model.gguf --devices-json devices.json \
                              --vram-budget-mib 4096 --json plan.json
    python moe_tier.py serve  --model model.gguf --engine-dir /path/to/bin [--run]

Only the Python standard library is required.  Python 3.9+.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------- devices

DEVICE_RE = re.compile(
    r"^\s*(?P<backend>[A-Za-z]+)(?P<index>\d*):\s*(?P<name>.+?)\s*"
    r"\((?P<total>\d+)\s*MiB(?:,\s*(?P<free>\d+)\s*MiB free)?\)\s*$")


@dataclass
class Device:
    backend: str          # ROCm / CUDA / Metal / Vulkan / CPU / ...
    index: int
    name: str
    total_mib: int
    free_mib: int | None = None


def parse_devices(text: str) -> list[Device]:
    """Parse `--list-devices` output.  Ignores init chatter; tolerant of any
    backend's name column.  CPU-only lines ('CPU: ...') are accepted without
    a VRAM figure."""
    devices: list[Device] = []
    for line in text.splitlines():
        if "MiB" not in line:
            continue
        m = DEVICE_RE.match(line)
        if not m:
            continue
        backend = m.group("backend")
        idx = int(m.group("index") or 0)
        if backend.lower() == "cpu":
            devices.append(Device(backend, idx, m.group("name"), 0, 0))
        else:
            devices.append(Device(backend, idx, m.group("name"),
                                  int(m.group("total")),
                                  int(m.group("free")) if m.group("free") else None))
    return devices


def list_devices(engine_dir: Path) -> list[Device]:
    bin_name = "llama-bench" + (".exe" if os.name == "nt" else "")
    exe = engine_dir / bin_name
    if not exe.exists():
        raise SystemExit(f"{exe} not found — pass --engine-dir with the "
                         f"llama.cpp binaries")
    out = subprocess.run([str(exe), "--list-devices"], capture_output=True,
                         text=True, timeout=120)
    devices = parse_devices(out.stdout + out.stderr)
    if not devices:
        raise SystemExit("could not parse --list-devices output")
    return devices


# ---------------------------------------------------------------------- gguf

# GGUF metadata value types (spec v3)
_GGUF_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
                 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int
    size: int | None = None      # bytes, filled from the offset table


@dataclass
class GGUFInfo:
    path: Path
    arch: str
    name: str
    block_count: int
    context_length: int | None
    embedding_length: int | None
    head_count: int | None
    head_count_kv: int | None
    key_length: int | None
    expert_count: int | None
    expert_used_count: int | None
    metadata: dict = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    data_start: int = 0
    file_size: int = 0
    recurrent_layers: list[bool] | None = None

    # -- derived -----------------------------------------------------------
    def sizes(self) -> dict[str, int]:
        return {t.name: (t.size or 0) for t in self.tensors}

    def expert_bytes_per_layer(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for t in self.tensors:
            m = re.match(r"^blk\.(\d+)\.", t.name)
            if not m or "_exps." not in t.name:
                continue
            layer = int(m.group(1))
            out[layer] = out.get(layer, 0) + (t.size or 0)
        return out

    def expert_bytes(self) -> int:
        return sum(t.size or 0 for t in self.tensors if "_exps." in t.name)

    def total_bytes(self) -> int:
        return sum(t.size or 0 for t in self.tensors)


def _read_scalar(f, vtype: int):
    fmt, size = _GGUF_SCALARS[vtype]
    return struct.unpack(fmt, f.read(size))[0]


def read_gguf(path: str | Path) -> GGUFInfo:
    """Stream the GGUF header: metadata + tensor infos only, no weights."""
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        metadata: dict = {}
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8", "replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            metadata[key] = _read_value(f, vtype, key)
        tensors: list[TensorInfo] = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode("utf-8", "replace")
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append(TensorInfo(name, dims, ggml_type, offset))
        header_end = f.tell()

    alignment = int(metadata.get("general.alignment", 32) or 32)
    data_start = (header_end + alignment - 1) // alignment * alignment

    # Exact tensor byte sizes from the offset table (alignment padding lands in
    # the gap, <= alignment bytes per tensor — irrelevant for placement).
    ordered = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(ordered):
        nxt = ordered[i + 1].offset if i + 1 < len(ordered) else \
            file_size - data_start
        t.size = max(0, nxt - t.offset)

    arch = str(metadata.get("general.architecture", ""))
    pre = arch + "."
    get = lambda key: metadata.get(pre + key)  # noqa: E731

    rec = metadata.get(pre + "attention.recurrent_layers")

    return GGUFInfo(
        path=path, arch=arch,
        name=str(metadata.get("general.name", path.stem)),
        block_count=int(get("block_count") or 0),
        context_length=_maybe_int(get("context_length")),
        embedding_length=_maybe_int(get("embedding_length")),
        head_count=_maybe_int(get("attention.head_count")),
        head_count_kv=_maybe_int(get("attention.head_count_kv")),
        key_length=_maybe_int(get("attention.key_length")),
        expert_count=_maybe_int(get("expert_count")),
        expert_used_count=_maybe_int(get("expert_used_count")),
        metadata={k: v for k, v in metadata.items()
                  if not isinstance(v, (list, bytes)) and not k.startswith("tokenizer")},
        tensors=tensors, data_start=data_start, file_size=file_size,
        recurrent_layers=rec if isinstance(rec, list) else None,
    )


def _maybe_int(v):
    if v is None or isinstance(v, (list, dict)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _read_value(f, vtype: int, key: str = ""):
    if vtype in _GGUF_SCALARS:
        return _read_scalar(f, vtype)
    if vtype == _GGUF_STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    if vtype == _GGUF_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if elem_type == _GGUF_STRING:      # tokenizer vocabulary etc.
            for _ in range(n):
                slen = struct.unpack("<Q", f.read(8))[0]
                f.seek(slen, 1)
            return None
        if key.endswith("attention.recurrent_layers"):
            # hybrid geometry: per-layer recurrent flags drive the KV model
            return [bool(_read_scalar(f, elem_type)) for _ in range(n)]
        fmt, size = _GGUF_SCALARS.get(elem_type, ("<B", 1))
        f.seek(n * size, 1)                # skip; we don't need array values
        return None
    raise SystemExit(f"unknown GGUF metadata type {vtype}")


# -------------------------------------------------------------------- threads

def physical_cores() -> tuple[int | None, str]:
    """Best-effort physical core count.  Returns (count, source)."""
    if sys.platform.startswith("linux") and Path("/proc/cpuinfo").exists():
        pairs = set()
        phys = core = None
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif not line.strip() and phys is not None and core is not None:
                pairs.add((phys, core)); phys = core = None
        if pairs:
            return len(pairs), "linux /proc/cpuinfo"
    if sys.platform == "darwin":
        try:
            n = int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                   capture_output=True, text=True).stdout)
            return n, "macOS sysctl"
        except Exception:
            pass
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True, text=True, timeout=20).stdout
            nums = [int(x) for x in re.findall(r"\d+", out)]
            if nums:
                return sum(nums), "windows wmic"
        except Exception:
            pass
    logical = os.cpu_count() or 1
    if logical >= 8:
        # assume SMT and suggest physical cores; overridable with --threads
        return max(1, logical // 2), f"assumed SMT (logical={logical})"
    return logical, "logical cpu count"


# ---------------------------------------------------------------------- plan

KV_BYTES_PER_ELEM = {"f16": 2.0, "bf16": 2.0, "f32": 4.0,
                     "q8_0": 1.0625, "q4_0": 0.5625}

# Measured on the released 35B GGUF (50 chunks x 512 ctx, -fa on; handoff
# 2026-09-29): f16 7.4851, q8_0 7.4824 (-0.036%), q4_0 7.5031 (+0.240%).
# Percent PPL delta vs f16; the planner carries it so the type choice is priced.
KV_QUALITY_DELTA_PCT = {"f16": 0.0, "bf16": 0.0, "f32": 0.0,
                        "q8_0": -0.036, "q4_0": 0.240}


@dataclass
class Plan:
    model: str
    arch: str
    name: str
    devices: list[Device]
    gpus: list[Device]
    split_mode: str
    main_gpu: int
    n_gpu_layers: int
    n_cpu_moe: int
    threads: int
    threads_source: str
    context: int
    kv_mib: float
    kv_type: str = "f16"
    kv_layers: int = 0
    kv_recurrent_layers: int = 0
    kv_state_mib: float = 0.0
    kv_quality_delta_pct: float = 0.0
    model_mib: float = 0.0
    non_expert_mib: float = 0.0
    expert_mib: float = 0.0
    expert_layers_on_gpu: int = 0
    expert_layers_total: int = 0
    est_vram_mib: float = 0.0
    vram_budget_mib: float = 0.0
    notes: list[str] = field(default_factory=list)

    def engine_args(self) -> list[str]:
        args = ["-ngl", str(self.n_gpu_layers), "-ncmoe", str(self.n_cpu_moe),
                "-t", str(self.threads)]
        if self.split_mode == "none":
            args += ["--split-mode", "none", "--main-gpu", str(self.main_gpu)]
        else:
            args += ["--split-mode", "layer"]
        return args

    def server_cmd(self, engine_dir: str, host: str = "127.0.0.1",
                   port: int = 8080) -> str:
        exe = os.path.join(engine_dir, "llama-server" +
                           (".exe" if os.name == "nt" else ""))
        return " ".join([exe, "-m", self.model, "-c", str(self.context)]
                        + self.engine_args()
                        + ["--host", host, "--port", str(port)])

    def bench_cmd(self, engine_dir: str, p: int = 512, n: int = 128,
                  r: int = 3) -> str:
        exe = os.path.join(engine_dir, "llama-bench" +
                           (".exe" if os.name == "nt" else ""))
        return " ".join([exe, "-m", self.model, "-p", str(p), "-n", str(n),
                         "-r", str(r)] + self.engine_args())

    def as_dict(self) -> dict:
        d = asdict(self)
        d["devices"] = [asdict(x) for x in self.devices]
        d["gpus"] = [asdict(x) for x in self.gpus]
        return d


def make_plan(info: GGUFInfo, devices: list[Device], *,
              context: int | None = None, threads: int | None = None,
              gpus: list[int] | None = None,
              vram_budget_mib: float | None = None,
              reserve_mib: float = 1024, kv_type: str = "f16") -> Plan:
    """Decide split mode + how many expert layers must stay on the CPU."""
    gpu_devs = [d for d in devices if d.backend.lower() != "cpu" and d.total_mib > 0]
    if gpus is not None:
        gpu_devs = [d for d in gpu_devs if d.index in gpus]
    if not gpu_devs:
        raise SystemExit("no GPU devices found — CPU-only serving needs no plan "
                         "(use -ngl 0)")

    if context is None:
        context = min(info.context_length or 4096, 8192)
    if threads is None:
        threads, threads_src = physical_cores()
    else:
        threads_src = "cli override"

    # --- footprint --------------------------------------------------------
    model_bytes = info.total_bytes()
    expert_bytes = info.expert_bytes()
    non_expert = model_bytes - expert_bytes
    per_layer = info.expert_bytes_per_layer()
    n_layers = info.block_count or (max(per_layer) + 1 if per_layer else 0)
    layer_bytes = (sum(per_layer.values()) / len(per_layer)) if per_layer else 0.0

    n_kv = info.head_count_kv or info.head_count or 1
    head_dim = info.key_length or ((info.embedding_length or 0) //
                                   max(info.head_count or 1, 1)) or 0

    # Hybrid geometry: only *non-recurrent* layers carry context-growing KV;
    # the GDN layers carry a fixed-size recurrent state (kept in f32 by the
    # engine): conv state (key_dim + value_dim)*(d_conv-1) + ssm state
    # d_state * d_state * n_v_heads, with key_dim = d_state*n_k_heads and
    # value_dim = d_state*n_v_heads (ssm.* metadata).
    n_rec = 0
    kv_layers = n_layers
    if info.recurrent_layers:
        rec = (list(info.recurrent_layers) + [False] * n_layers)[:n_layers]
        n_rec = sum(1 for x in rec if x)
        kv_layers = n_layers - n_rec
    kv_bytes = 2 * kv_layers * n_kv * head_dim * KV_BYTES_PER_ELEM.get(kv_type, 2.0)
    kv_mib = kv_bytes * context / 1024 / 1024

    kv_state_mib = 0.0
    state_warning = ""
    if n_rec:
        pre = info.arch + "."
        d_state = _maybe_int(info.metadata.get(pre + "ssm.state_size"))
        n_group = _maybe_int(info.metadata.get(pre + "ssm.group_count"))
        dt_rank = _maybe_int(info.metadata.get(pre + "ssm.time_step_rank"))
        d_conv = _maybe_int(info.metadata.get(pre + "ssm.conv_kernel"))
        if None not in (d_state, n_group, dt_rank, d_conv):
            key_dim = d_state * n_group
            value_dim = d_state * dt_rank
            elems = ((key_dim + value_dim) * (d_conv - 1)
                     + d_state * d_state * dt_rank)
            kv_state_mib = n_rec * elems * 4 / 1024 / 1024
        else:
            state_warning = ("recurrent layers present but ssm.* metadata is "
                             "incomplete; fixed recurrent state not counted")

    budget = (vram_budget_mib if vram_budget_mib is not None
              else float(sum(d.total_mib for d in gpu_devs)))
    reserve = reserve_mib * len(gpu_devs)
    usable = budget - reserve

    notes: list[str] = []
    if state_warning:
        notes.append(state_warning)
    expert_gpu_budget = usable - non_expert / 1048576 - kv_mib - kv_state_mib
    if expert_gpu_budget < 0:
        notes.append("non-expert weights + KV do not fit VRAM: reduce --context, "
                     "use --no-kv-offload, or lower -ngl")
        expert_layers_gpu = 0
    else:
        expert_layers_gpu = min(n_layers, int(expert_gpu_budget /
                                              max(layer_bytes / 1048576, 1e-9)))
    n_cpu_moe = max(0, n_layers - expert_layers_gpu)
    if n_cpu_moe == 0 and layer_bytes:
        notes.append("all expert layers fit in VRAM")
    notes.append(f"KV {kv_mib:.0f} MiB at {context} tokens ({kv_type}; "
                 f"{kv_layers} KV + {n_rec} GDN state {kv_state_mib:.0f} MiB); "
                 f"KV quality {KV_QUALITY_DELTA_PCT.get(kv_type, 0.0):+.3f}% PPL "
                 f"vs f16 (measured); reserve {reserve_mib:.0f} MiB/GPU")

    # --- single GPU or split ---------------------------------------------
    expert_on_gpu_mib = expert_layers_gpu * layer_bytes / 1048576
    footprint = non_expert / 1048576 + kv_mib + kv_state_mib + reserve_mib
    best = max(gpu_devs, key=lambda d: d.total_mib)
    if footprint <= best.total_mib:                 # fits one card on its own
        split_mode, main_gpu = "none", best.index
        notes.append(f"weights+KV fit one card ({best.name}); "
                     f"no layer split needed")
    else:
        split_mode, main_gpu = "layer", gpu_devs[0].index

    est_vram = footprint + expert_on_gpu_mib
    if est_vram > budget:
        notes.append("WARNING: estimated VRAM exceeds budget; expect CPU spill")

    return Plan(
        model=str(info.path), arch=info.arch, name=info.name,
        devices=devices, gpus=gpu_devs, split_mode=split_mode,
        main_gpu=main_gpu, n_gpu_layers=99, n_cpu_moe=n_cpu_moe,
        threads=threads, threads_source=threads_src, context=context,
        kv_mib=round(kv_mib, 1), kv_type=kv_type, kv_layers=kv_layers,
        kv_recurrent_layers=n_rec, kv_state_mib=round(kv_state_mib, 1),
        kv_quality_delta_pct=KV_QUALITY_DELTA_PCT.get(kv_type, 0.0),
        model_mib=round(model_bytes / 1048576, 1),
        non_expert_mib=round(non_expert / 1048576, 1),
        expert_mib=round(expert_bytes / 1048576, 1),
        expert_layers_on_gpu=expert_layers_gpu,
        expert_layers_total=n_layers, est_vram_mib=round(est_vram, 1),
        vram_budget_mib=budget, notes=notes)


# ----------------------------------------------------------------------- cli

def _fmt_plan(p: Plan) -> str:
    lines = [
        f"model    {p.name} [{p.arch}]  {p.model_mib:.0f} MiB "
        f"({p.expert_mib:.0f} expert / {p.non_expert_mib:.0f} non-expert)",
        f"devices  " + ", ".join(
            f"{d.backend}{d.index} {d.name} {d.total_mib} MiB" for d in p.gpus),
        f"context  {p.context}  →  KV {p.kv_mib:.0f} MiB {p.kv_type} "
        f"({p.kv_layers} KV + {p.kv_recurrent_layers} GDN state "
        f"{p.kv_state_mib:.0f} MiB; {p.kv_quality_delta_pct:+.3f}% PPL) "
        f"({p.vram_budget_mib:.0f} MiB budget)",
        f"threads  {p.threads} ({p.threads_source})",
        f"split    {p.split_mode}" + (f" (main-gpu {p.main_gpu})"
                                     if p.split_mode == "none" else ""),
        f"experts  {p.expert_layers_on_gpu}/{p.expert_layers_total} layers on "
        f"GPU  →  -ncmoe {p.n_cpu_moe}",
        f"est VRAM {p.est_vram_mib:.0f} MiB",
    ]
    lines += [f"  note: {n}" for n in p.notes]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe", help="list devices + thread defaults")
    probe.add_argument("--engine-dir", required=True)
    probe.add_argument("--json", metavar="FILE")

    def add_plan_args(p):
        p.add_argument("--model", required=True)
        p.add_argument("--engine-dir", help="dir with llama-bench/llama-server")
        p.add_argument("--devices-json", help="use a devices list instead of probing")
        p.add_argument("--context", type=int)
        p.add_argument("--threads", type=int)
        p.add_argument("--gpus", help="comma-separated device indexes")
        p.add_argument("--vram-budget-mib", type=float,
                       help="override total VRAM across the selected GPUs "
                            "(simulate another machine; combine with --gpus to "
                            "model a single card)")
        p.add_argument("--reserve-mib", type=float, default=1024)
        p.add_argument("--kv-type", default="f16",
                       choices=sorted(KV_BYTES_PER_ELEM))
        p.add_argument("--json", metavar="FILE")

    plan = sub.add_parser("plan", help="compute a placement plan")
    add_plan_args(plan)

    serve = sub.add_parser("serve", help="plan + launch llama-server")
    add_plan_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--run", action="store_true",
                       help="exec (default: print the command)")

    a = ap.parse_args(argv)

    if a.cmd == "probe":
        devices = list_devices(Path(a.engine_dir))
        threads, src = physical_cores()
        out = {"devices": [asdict(d) for d in devices],
               "physical_cores": threads, "source": src}
        for d in devices:
            print(f"  {d.backend}{d.index}: {d.name} — {d.total_mib} MiB"
                  + (f" ({d.free_mib} MiB free)" if d.free_mib is not None else ""))
        print(f"  threads: {threads} ({src})")
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
        return 0

    # plan / serve
    if a.devices_json:
        raw = json.loads(Path(a.devices_json).read_text()) if \
            Path(a.devices_json).exists() else json.loads(a.devices_json)
        if isinstance(raw, dict):
            raw = raw["devices"]
        devices = [Device(**d) for d in raw]
    elif a.engine_dir:
        devices = list_devices(Path(a.engine_dir))
    else:
        raise SystemExit("pass --engine-dir or --devices-json")

    info = read_gguf(a.model)
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None
    p = make_plan(info, devices, context=a.context, threads=a.threads,
                  gpus=gpus, vram_budget_mib=a.vram_budget_mib,
                  reserve_mib=a.reserve_mib, kv_type=a.kv_type)

    print(_fmt_plan(p))
    engine_dir = a.engine_dir or "<engine-dir>"
    print("\n  serve: " + p.server_cmd(engine_dir, getattr(a, "host", "127.0.0.1"),
                                        getattr(a, "port", 8080)))
    print("  bench: " + p.bench_cmd(engine_dir))
    if a.json:
        Path(a.json).write_text(json.dumps(p.as_dict(), indent=1))
        print(f"  json  -> {a.json}")

    if a.cmd == "serve" and a.run:
        cmd = p.server_cmd(engine_dir, a.host, a.port).split()
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
