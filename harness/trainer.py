"""Preset trainer — evidence-driven composition iteration (X15–X18).

The product bet: "your AI gets better at your workflow the more you use it."
dsh has no weight-level training; "training" here means evidence-driven
composition iteration:

1. **Evidence pass (X15)** — mine session logs through the ``session-query``
   seam for successful tool-use traces, failure modes, and unused tools per
   preset.
2. **Candidate pass (X16)** — LLM-drafted composition diffs (prompt-section
   rewrites, tool add/remove) written as CANDIDATE presets in the user root,
   never overwriting live ones.
3. **Eval loop (X17)** — run candidate vs baseline headless on bench protocol
   tasks, score PES + task pass rate, emit a comparison report.
4. **Promotion flow (X18)** — one command promotes a validated candidate to
   a real preset id (rename + roster rewrite), with rollback kept as the
   untouched candidate directory.

All local, all auditable files. Uses only sanctioned authoring surfaces:
files in the user's preset root.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ToolTrace:
    """One tool invocation mined from a session log."""

    tool: str
    success: bool
    duration_ms: float
    args_summary: str = ""
    error: str = ""


@dataclass
class SessionEvidence:
    """Mined evidence from one dsh session log."""

    session_id: str
    preset: str
    total_turns: int
    tool_traces: list[ToolTrace] = field(default_factory=list)
    completed: bool = False
    errors: list[str] = field(default_factory=list)
    unused_tools: set[str] = field(default_factory=set)

    @property
    def tool_success_rate(self) -> float:
        if not self.tool_traces:
            return 0.0
        return sum(1 for t in self.tool_traces if t.success) / len(self.tool_traces)

    @property
    def tools_used(self) -> set[str]:
        return {t.tool for t in self.tool_traces}

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "preset": self.preset,
            "total_turns": self.total_turns,
            "tool_calls": len(self.tool_traces),
            "tool_success_rate": round(self.tool_success_rate, 3),
            "tools_used": sorted(self.tools_used),
            "completed": self.completed,
            "errors": self.errors[:5],
        }


@dataclass
class CandidatePreset:
    """A drafted composition variant awaiting eval."""

    name: str
    baseline: str
    changes: dict  # {tool_add: [], tool_remove: [], prompt_rewrite: str}
    evidence_summary: dict
    path: str = ""
    eval_result: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "baseline": self.baseline,
            "changes": self.changes, "evidence": self.evidence_summary,
            "path": self.path, "eval": self.eval_result,
        }


# ---------------------------------------------------------------------------
# X15: Evidence pass
# ---------------------------------------------------------------------------
def mine_evidence(session_root: Path, all_tools: list[str]) -> list[SessionEvidence]:
    """Scan dsh session JSONL logs for tool-use patterns and outcomes."""
    import zstandard

    evidences = []
    if not session_root.is_dir():
        return evidences
    for log_path in sorted(session_root.rglob("session.jsonl.zstd")):
        session_id = log_path.parent.name
        dctx = zstandard.ZstdDecompressor()
        with open(log_path, "rb") as fh:
            reader = dctx.stream_reader(fh, read_across_frames=True)
            text = reader.read().decode("utf-8", errors="replace")
        ev = SessionEvidence(session_id=session_id, preset="", total_turns=0)
        preset = ""
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            data = record.get("data") or {}
            if kind == "tool/call":
                tool_name = str(data.get("name") or data.get("tool") or "?")
                ev.tool_traces.append(ToolTrace(tool=tool_name, success=True))
            elif kind == "tool/result":
                if ev.tool_traces:
                    ev.tool_traces[-1].success = not data.get("isError", False)
            elif kind == "turn/end":
                ev.total_turns += 1
                reason = (data.get("reason") or {}).get("kind", "")
                if reason == "completed":
                    ev.completed = True
                elif reason == "error":
                    ev.errors.append(str((data.get("reason") or {}).get("error", ""))[:200])
        ev.preset = preset
        ev.unused_tools = set(all_tools) - ev.tools_used
        evidences.append(ev)
    return evidences


def summarize_evidence(evidences: list[SessionEvidence]) -> dict:
    """Aggregate evidence across sessions into a trainer-ready summary."""
    if not evidences:
        return {"sessions": 0}
    all_tools: Counter = Counter()
    success_by_tool: dict[str, list[bool]] = defaultdict(list)
    for ev in evidences:
        for t in ev.tool_traces:
            all_tools[t.tool] += 1
            success_by_tool[t.tool].append(t.success)
    tool_stats = {}
    for tool, count in all_tools.most_common():
        successes = success_by_tool[tool]
        tool_stats[tool] = {
            "calls": count,
            "success_rate": round(sum(successes) / len(successes), 3) if successes else 0,
        }
    return {
        "sessions": len(evidences),
        "completed": sum(1 for e in evidences if e.completed),
        "total_turns": sum(e.total_turns for e in evidences),
        "total_tool_calls": sum(len(e.tool_traces) for e in evidences),
        "tool_stats": tool_stats,
        "avg_success_rate": round(
            sum(e.tool_success_rate for e in evidences) / len(evidences), 3
        ) if evidences else 0,
        "sessions_with_errors": sum(1 for e in evidences if e.errors),
    }


# ---------------------------------------------------------------------------
# X16: Candidate pass
# ---------------------------------------------------------------------------
def draft_candidate(
    evidence_summary: dict,
    baseline_preset_path: Path,
    output_dir: Path,
    name: str,
) -> CandidatePreset:
    """Draft a composition variant from the evidence.

    v1 heuristic (no LLM call needed): remove tools with 0 calls across all
    sessions (dead weight), keep everything else. A later version can use
    the loaded LLM to draft prompt-section rewrites.
    """
    tool_stats = evidence_summary.get("tool_stats", {})
    dead_tools = [t for t, s in tool_stats.items() if s["calls"] == 0]
    low_success = [t for t, s in tool_stats.items()
                   if s["calls"] > 0 and s["success_rate"] < 0.5]

    changes = {
        "tool_remove": dead_tools,
        "tool_review": low_success,
        "prompt_rewrite": None,  # LLM-drafted in a future version
    }

    # Copy the baseline preset file to the candidate path
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / f"{name}.cordis.yml"
    if baseline_preset_path.is_file():
        shutil.copy2(baseline_preset_path, candidate_path)

    # Write the evidence + changes as a sidecar JSON for auditability
    meta_path = output_dir / f"{name}.trainer-meta.json"
    meta_path.write_text(json.dumps({
        "baseline": str(baseline_preset_path),
        "changes": changes,
        "evidence_summary": evidence_summary,
        "drafted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    return CandidatePreset(
        name=name,
        baseline=str(baseline_preset_path),
        changes=changes,
        evidence_summary=evidence_summary,
        path=str(candidate_path),
    )


# ---------------------------------------------------------------------------
# X17: Eval loop
# ---------------------------------------------------------------------------
def evaluate_candidate(
    candidate: CandidatePreset,
    harness_factory: Callable,
    bench_tasks: list[str],
    max_convs: int = 2,
) -> dict:
    """Run candidate vs baseline on the bench protocol, score, compare.

    ``harness_factory`` is a callable that takes a cordis config path and
    returns a ``DeepSeekHarness``-like object with ``run(message, session_id)``.
    In production this is the SDK; in tests it's a stub.
    """
    results = {"candidate": {}, "baseline": {}, "verdict": "pending"}
    for label, config_path in (
        ("candidate", candidate.path),
        ("baseline", candidate.baseline),
    ):
        scores = []
        for task in bench_tasks:
            try:
                harness = harness_factory(config_path)
                result = harness.run(task)
                scores.append({
                    "task": task[:60],
                    "finish": result.finish_reason,
                    "response_len": len(result.final_response or ""),
                    "ok": result.finish_reason == "completed",
                })
            except Exception as exc:
                scores.append({"task": task[:60], "ok": False, "error": str(exc)[:200]})
        pass_rate = (sum(1 for s in scores if s.get("ok")) / len(scores)
                     if scores else 0)
        results[label] = {"tasks": scores, "pass_rate": round(pass_rate, 3)}

    c_pr = results["candidate"]["pass_rate"]
    b_pr = results["baseline"]["pass_rate"]
    results["verdict"] = (
        "candidate-wins" if c_pr > b_pr
        else "baseline-wins" if b_pr > c_pr
        else "tie"
    )
    candidate.eval_result = results
    return results


# ---------------------------------------------------------------------------
# X18: Promotion flow
# ---------------------------------------------------------------------------
def promote(candidate: CandidatePreset, presets_root: Path,
            new_name: str) -> dict:
    """Promote a validated candidate to a real preset (copy + rename).
    Rollback is kept as the untouched candidate directory."""
    src = Path(candidate.path)
    if not src.is_file():
        return {"ok": False, "error": f"candidate file not found: {src}"}
    dest = presets_root / f"{new_name}.cordis.yml"
    if dest.exists():
        return {"ok": False, "error": f"preset already exists: {dest}"}
    presets_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "ok": True, "promoted_to": str(dest),
        "rollback_kept": src,
        "note": "restart the studio to pick up the new preset",
    }
