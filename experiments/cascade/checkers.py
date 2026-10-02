"""Task checkers and JSON extraction — shared by the cascade experiments.

Kept free of ``harness`` imports on purpose: the eval scripts run under the
ROCm venv (for the local decision models), which does not carry the sidecar's
web dependencies.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from typing import Optional


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", folded).strip()


def _last_number(text: str) -> Optional[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _code_block(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return blocks[-1].strip()
    return None


def check(task: dict, answer: str) -> bool:
    """Apply the task's checker to an answer string."""
    if not answer or not answer.strip():
        return False
    spec = task["checker"]
    kind = spec["type"]
    if kind == "contains":
        normalized = _normalize(answer)
        return any(_normalize(e) in normalized for e in spec["expect"])
    if kind == "number":
        got = _last_number(answer)
        if got is None:
            return False
        return abs(got - float(spec["expect"])) <= float(spec.get("tol", 1e-6))
    if kind == "code":
        code = _code_block(answer)
        if code is None:
            return False
        program = code + "\n\n" + "\n".join(spec["tests"]) + "\nprint('PASS')\n"
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, dir="/tmp/opencode"
        ) as fh:
            fh.write(program)
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, path], capture_output=True, text=True, timeout=15
            )
            return proc.returncode == 0 and "PASS" in proc.stdout
        except subprocess.TimeoutExpired:
            return False
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    raise ValueError(f"unknown checker {kind!r}")


def extract_json(text: str) -> Optional[dict]:
    """First JSON object in a text (fences and prose tolerated)."""
    if not text:
        return None
    for match in re.finditer(r"\{.*?\}", text, re.S):
        try:
            return json.loads(match.group(0))
        except ValueError:
            continue
    try:
        return json.loads(text)
    except ValueError:
        return None
