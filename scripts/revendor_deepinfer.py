#!/usr/bin/env python3
"""Re-vendor the DeepInfer inference framework from upstream deepseek-ai/DeepInfer.

Run from a normal terminal (NOT the AI sandbox):

    venv/bin/python scripts/revendor_deepinfer.py [--pin <sha>] [--source <path|url>]

What it does:
  1. Resolves the upstream commit (--pin, default: upstream HEAD) from --source
     (a local checkout or a git URL; a URL is cloned shallowly into a temp dir).
  2. Copies the upstream tree into vendor/deepinfer/, skipping build artifacts,
     __pycache__ and the CUDA/Ascend submodule payloads (the AMD lane replaces
     them with torch-native shims; see vendor/deepinfer_amd/).
  3. Re-applies our AMD port patches from vendor/deepinfer_amd/patches/ on top
     of the pristine upstream files.
  4. Stamps vendor/deepinfer/PROVENANCE.md (origin, commit, date, per-file
     sha256, and the patch set that was applied).
  5. Prints a per-file status vs the currently vendored copy.

It never commits anything itself -- review `git diff vendor/` and commit with
CLAIMS_GATE=skip (the claims gate rejects vendor/** even as deletions).
"""
import argparse
import datetime
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

DEFAULT_SOURCE = "https://github.com/deepseek-ai/DeepInfer.git"
DEFAULT_PIN = "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "deepinfer")
OVERLAY = os.path.join(ROOT, "vendor", "deepinfer_amd")
PATCH_DIR = os.path.join(OVERLAY, "patches")

#: Directory names never copied out of the upstream tree.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist", ".venv"}

#: Upstream submodule payloads (vendor/DeepGEMM, vendor/FlashMLA, ...). They are
#: CUDA/Ascend trees the AMD lane does not build; the empty submodule stubs are
#: kept so the upstream layout still resolves.
SKIP_SUBMODULE_DIRS = {"DeepGEMM", "DeepGEMM-Ascend", "DeepEP", "DeepEP-Ascend", "DeepSelect", "FlashMLA"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def resolve_source(source, pin):
    """Return (tree_path, commit_sha). Clones into a temp dir when source is a URL."""
    if os.path.isdir(os.path.join(source, ".git")):
        head = _run(["git", "-C", source, "rev-parse", "HEAD"]).stdout.strip()
        commit = pin or head
        ok = _run(["git", "-C", source, "cat-file", "-e", f"{commit}^{{commit}}"])
        if ok.returncode != 0:
            sys.exit(f"commit {commit} not found in {source}")
        return source, commit
    tmp = tempfile.mkdtemp(prefix="deepinfer-vendor-")
    clone = os.path.join(tmp, "DeepInfer")
    print(f"cloning {source} ...")
    ok = _run(["git", "clone", "--filter=blob:none", source, clone])
    if ok.returncode != 0:
        sys.exit(f"clone failed: {ok.stderr.strip()}")
    commit = pin or _run(["git", "-C", clone, "rev-parse", "HEAD"]).stdout.strip()
    if pin:
        ok = _run(["git", "-C", clone, "checkout", pin])
        if ok.returncode != 0:
            sys.exit(f"checkout {pin} failed: {ok.stderr.strip()}")
    return clone, commit


def _copy_tree(src, dst):
    """Copy the upstream tree into dst, skipping artifacts and submodule payloads."""
    for entry in sorted(os.listdir(src)):
        src_path = os.path.join(src, entry)
        dst_path = os.path.join(dst, entry)
        if os.path.isdir(src_path):
            if entry in SKIP_DIRS:
                continue
            if entry == "vendor":
                os.makedirs(dst_path, exist_ok=True)
                for sub in sorted(os.listdir(src_path)):
                    if sub in SKIP_SUBMODULE_DIRS:
                        # Keep the stub so the upstream layout still resolves.
                        os.makedirs(os.path.join(dst_path, sub), exist_ok=True)
                        continue
                    s = os.path.join(src_path, sub)
                    d = os.path.join(dst_path, sub)
                    if os.path.isdir(s):
                        _copy_tree(s, d)
                    else:
                        shutil.copy2(s, d)
                continue
            os.makedirs(dst_path, exist_ok=True)
            _copy_tree(src_path, dst_path)
        else:
            if entry.endswith((".pyc", ".so", ".pyo")):
                continue
            shutil.copy2(src_path, dst_path)


def _relative_files(root):
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith((".pyc", ".so", ".pyo")):
                continue
            out.append(os.path.relpath(os.path.join(base, name), root))
    return sorted(out)


def apply_patches():
    """Apply our AMD port patches to the vendored tree. Returns the patch names."""
    if not os.path.isdir(PATCH_DIR):
        return []
    applied = []
    for name in sorted(os.listdir(PATCH_DIR)):
        if not name.endswith(".patch"):
            continue
        ok = _run(["git", "apply", "--whitespace=nowarn", "-p1",
                   os.path.join(PATCH_DIR, name)], cwd=ROOT)
        if ok.returncode != 0:
            sys.exit(f"patch {name} failed to apply:\n{ok.stdout}\n{ok.stderr}")
        applied.append(name)
    return applied


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pin", default=DEFAULT_PIN, help="upstream commit to vendor")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="local checkout or git URL")
    args = ap.parse_args()

    source, commit = resolve_source(args.source, args.pin)

    if os.path.isdir(VENDOR):
        shutil.rmtree(VENDOR)
    os.makedirs(VENDOR, exist_ok=True)
    _copy_tree(source, VENDOR)
    patches = apply_patches()

    rows = []
    for rel in _relative_files(VENDOR):
        path = os.path.join(VENDOR, rel)
        rows.append((rel.replace(os.sep, "/"), sha256(path)))

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(os.path.join(VENDOR, "PROVENANCE.md"), "w") as f:
        f.write("# Vendored framework provenance\n\n")
        f.write(f"- Origin: deepseek-ai/DeepInfer (`{args.source}`)\n")
        f.write(f"- Commit: {commit}\n- Re-vendored: {stamp} by scripts/revendor_deepinfer.py\n")
        f.write(f"- License: Apache-2.0 (upstream)\n")
        if patches:
            f.write("- AMD port patches applied from `vendor/deepinfer_amd/patches/`:\n")
            for name in patches:
                f.write(f"  - `{name}`\n")
        else:
            f.write("- AMD port patches applied: none\n")
        f.write("\n| file | sha256 (first 16) |\n|---|---|\n")
        for name, digest in rows:
            f.write(f"| {name} | {digest[:16]}... |\n")

    print(f"vendored {len(rows)} files from {source} @ {commit[:12]}")
    if patches:
        print(f"applied {len(patches)} AMD port patch(es): {', '.join(patches)}")
    print("review `git diff vendor/`, then commit with CLAIMS_GATE=skip")


if __name__ == "__main__":
    main()
