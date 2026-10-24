#!/usr/bin/env python3
"""scripts/check_model_storage.py - where the harness's models actually live.

A model on a slow device costs far more than its size suggests.  The PLE
table of the Qwen3.8-Flash-Next model is read *randomly* (one 90-byte row per
token per layer), so every row is a seek: on the FUSE mount of a rotational
disk that is 7.5 ms, on the NVMe it is 0.19 ms - 40x.  Measured on
2026-10-21, moving that model's 66 GB of shards onto the NVMe took the
decode rate from 3.39 tok/s to 20.52 tok/s, and the PLE gather from 412 ms to
2.9 ms per round.

This walks every stack, resolves its model the way the harness does, and
reports the size and the device behind it - flagging anything that sits on a
FUSE mount or a rotational disk.  A stack whose model is not in the library
at all is reported too: it would be fetched from Hugging Face at load time
into the library, which is on the fast device, so that is fine, but it is
worth seeing.

    python scripts/check_model_storage.py            # the report
    python scripts/check_model_storage.py --slow     # only the slow ones
    python scripts/check_model_storage.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mounts() -> list[tuple[pathlib.Path, str, str]]:
    """(mount point, fs type, source) from /proc/self/mountinfo, longest first."""
    out: list[tuple[pathlib.Path, str, str]] = []
    try:
        text = pathlib.Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split(" - ")
        if len(parts) < 2:
            continue
        left, right = parts[0].split(), parts[1].split()
        if len(left) < 5 or len(right) < 2:
            continue
        point = left[4].replace("\\040", " ")
        out.append((pathlib.Path(point), right[0], right[1]))
    out.sort(key=lambda m: len(m[0].parts), reverse=True)
    return out


def _rotational(source: str) -> bool | None:
    """True/False for a block device's queue/rotational, None if unknown."""
    if not source.startswith("/dev/"):
        return None
    name = source.rsplit("/", 1)[-1]
    # /dev/nvme0n1p5 -> /sys/class/block/nvme0n1p5 -> ../../nvme0n1
    for base in ("/sys/class/block",):
        link = pathlib.Path(base) / name
        try:
            real = link.resolve()
        except OSError:
            continue
        for parent in real.parents:
            queue = parent / "queue" / "rotational"
            try:
                if queue.exists():
                    return queue.read_text().strip() == "1"
            except OSError:
                continue
    return None


def _device_kind(path: pathlib.Path) -> str:
    """'fuse', 'rotational' or 'solid' for the filesystem behind `path`."""
    try:
        target = path.resolve()
    except OSError:
        target = path
    for point, fs_type, source in _mounts():
        try:
            target.relative_to(point)
        except ValueError:
            continue
        if fs_type.startswith("fuse"):
            return "fuse"
        rot = _rotational(source)
        if rot is True:
            return "rotational"
        if rot is False:
            return "solid"
        return fs_type
    return "unknown"



def _stack_files() -> list[pathlib.Path]:
    return sorted((REPO / "stacks").glob("*.json"))


def _resolve(file_field: str, library: pathlib.Path) -> pathlib.Path | None:
    """The harness's own rule (harness/models.py: resolve_model)."""
    candidate = pathlib.Path(file_field)
    if candidate.is_file():
        return candidate.resolve()
    direct = library / file_field
    if direct.is_file():
        return direct.resolve()
    matches = [p for p in library.rglob("*.gguf") if file_field in p.stem or file_field == p.name]
    return matches[0].resolve() if matches else None


def collect(library: pathlib.Path | None = None) -> list[dict]:
    library = library or (REPO / "models" / "gguf")
    rows: list[dict] = []
    for stack_file in _stack_files():
        try:
            data = json.loads(stack_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rows.append({"stack": stack_file.name, "error": str(exc)})
            continue
        for tier in data.get("tiers", []):
            field = tier.get("file") or ""
            resolved = _resolve(field, library) if field else None
            row = {
                "stack": stack_file.name,
                "role": tier.get("role", "?"),
                "repo": tier.get("repo"),
                "file": field,
                "path": None,
                "gib": None,
                "device": None,
                "kind": None,
                "slow": False,
                "note": "",
            }
            if resolved is None:
                row["note"] = "not in the library (fetched from Hugging Face at load time)"
            else:
                st = resolved.stat()
                kind = _device_kind(resolved)
                row.update(
                    path=str(resolved),
                    gib=round(st.st_size / 2**30, 2),
                    device=f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}",
                    kind=kind,
                    slow=kind in ("fuse", "rotational"),
                )
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=pathlib.Path, default=None,
                    help="the model library (default: <repo>/models/gguf)")
    ap.add_argument("--slow", action="store_true", help="only report the slow ones")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    rows = collect(args.library)
    slow = [r for r in rows if r.get("slow")]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 1 if slow else 0

    shown = slow if args.slow else rows
    for row in shown:
        if "error" in row:
            print(f"  {row['stack']:34s} ERROR {row['error']}")
            continue
        if row["path"] is None:
            print(f"  {row['stack']:34s} {row['role']:10s} {'(fetch)':>10s}  {row['note']}")
            continue
        flag = "SLOW" if row["slow"] else "ok"
        print(f"  {row['stack']:34s} {row['role']:10s} {row['gib']:8.1f} GiB  "
              f"{row['device']:>5s} {row['kind']:<10s} {flag:4s}  {row['path']}")
    if not args.slow:
        print(f"\n  {len(rows)} tier(s), {len(slow)} on a slow device")
    return 1 if slow else 0


if __name__ == "__main__":
    sys.exit(main())
