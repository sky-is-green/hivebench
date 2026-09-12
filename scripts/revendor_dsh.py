#!/usr/bin/env python3
"""Re-vendor the dsh Python SDK from the local deepseek-harness fork.

Run from a normal terminal (NOT the AI sandbox):

    venv/bin/python scripts/revendor_dsh.py [--pin <sha>] [fork_path]

What it does:
  1. Resolves the fork commit (--pin, default: fork HEAD) and verifies the
     python/sdk/src/deepseek_harness package exists at that commit.
  2. Copies every top-level *.py from that package into vendor/deepseek_harness/.
  3. Stamps vendor/PROVENANCE.md (origin, commit, date, per-file sha256).
  4. Prints a per-file status vs the currently vendored copy.

It never commits anything itself — review `git diff vendor/` and commit with
CLAIMS_GATE=skip (the claims gate rejects vendor/** even as deletions).
"""
import argparse, datetime, hashlib, os, subprocess, sys

DEFAULT_FORK = "/home/penis/Desktop/work/deepseek-harness"
PKG_REL = "python/sdk/src/deepseek_harness"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "deepseek_harness")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pin", help="fork commit to vendor from (default: fork HEAD)")
    ap.add_argument("fork", nargs="?", default=DEFAULT_FORK)
    args = ap.parse_args()

    fork = os.path.abspath(args.fork)
    if not os.path.exists(os.path.join(fork, ".git")):
        sys.exit(f"not a git checkout: {fork}")
    head = subprocess.run(["git", "-C", fork, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    pin = args.pin or head
    r = subprocess.run(["git", "-C", fork, "cat-file", "-e", f"{pin}:{PKG_REL}"],
                       capture_output=True)
    if r.returncode != 0:
        sys.exit(f"package {PKG_REL} not found at {pin[:12]} in {fork}")

    listing = subprocess.run(
        ["git", "-C", fork, "ls-tree", "--name-only", pin, PKG_REL + "/"],
        capture_output=True, text=True).stdout.splitlines()
    py_files = [os.path.basename(f) for f in listing if f.endswith(".py")]
    if not py_files:
        sys.exit("no .py files found in package at pinned commit")

    os.makedirs(VENDOR, exist_ok=True)
    rows = []
    for name in sorted(py_files):
        blob = subprocess.run(
            ["git", "-C", fork, "show", f"{pin}:{PKG_REL}/{name}"],
            capture_output=True).stdout
        dst = os.path.join(VENDOR, name)
        cur = open(dst, "rb").read() if os.path.exists(dst) else None
        status = "identical" if cur == blob else ("changed" if cur is not None else "new")
        with open(dst, "wb") as f:
            f.write(blob)
        rows.append((name, sha256(dst), status))

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(os.path.join(VENDOR, "PROVENANCE.md"), "w") as f:
        f.write("# Vendored SDK provenance\n\n")
        f.write(f"- Origin: sky-is-green/deepseek-harness (fork), `{PKG_REL}/`\n")
        f.write(f"- Commit: {pin}\n- Re-vendored: {stamp} by scripts/revendor_dsh.py\n\n")
        f.write("| file | sha256 (first 16) |\n|---|---|\n")
        for name, digest, _ in rows:
            f.write(f"| {name} | {digest[:16]}... |\n")

    print(f"vendored {len(rows)} files from {fork} @ {pin[:12]}")
    for name, _, status in rows:
        print(f"  {status:9s} {name}")
    print("review `git diff vendor/`, then commit with CLAIMS_GATE=skip")


if __name__ == "__main__":
    main()
