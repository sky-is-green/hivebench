"""Splinter needle-in-haystack retrieval test.

Phase 1: Ingest SPLINTER-CONTEXT.md into a fresh splinter conversation via
         /v1/splinter/curate (curation only, no LLM generation — fast).
Phase 2: Ask targeted questions via /v1/splinter/curate and verify the
         correct facts appear in assembled_content.
Phase 3: Dump inspection data for the final query to show chunk-level detail.

Run: python -m tests.integration.test_splinter_retrieval
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

SIDECAR = "http://127.0.0.1:8765"
CID = "retrieval-test"
def _context_file() -> Path:
    for _dir in ("splinter-memory", "strata-memory"):
        for _name in ("SPLINTER-CONTEXT.md", "STRATA-CONTEXT.md"):
            _c = Path.home() / "Desktop/work" / _dir / _name
            if _c.is_file():
                return _c
    return Path.home() / "Desktop/work" / "splinter-memory" / "SPLINTER-CONTEXT.md"

CONTEXT_FILE = _context_file()
CHUNK_SIZE = 3000  # chars per turn (~750 tokens)


def post(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def reset_conversation():
    try:
        post(f"{SIDECAR}/v1/splinter/reset", {"conversation_id": CID})
        print("  reset ok")
    except Exception as e:
        print(f"  reset: {e}")


def split_into_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split at paragraph boundaries near the target size."""
    chunks = []
    current = ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def ingest_context() -> int:
    """Feed SPLINTER-CONTEXT.md through /v1/splinter/curate in chunks."""
    text = CONTEXT_FILE.read_text(encoding="utf-8")
    chunks = split_into_chunks(text)
    print(f"  {len(chunks)} chunks to ingest (avg {sum(len(c) for c in chunks)//len(chunks)} chars)")
    
    t0 = time.time()
    failures = 0
    for i, chunk in enumerate(chunks):
        query = f"[block {i+1}/{len(chunks)}] {chunk}"
        try:
            post(f"{SIDECAR}/v1/splinter/curate", {
                "query": query,
                "conversation_id": CID,
            }, timeout=30)
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  chunk {i+1} failed: {e}")
            continue
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(chunks) - i - 1) / rate
            print(f"  {i+1}/{len(chunks)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    elapsed = time.time() - t0
    ok = len(chunks) - failures
    print(f"  done: {ok}/{len(chunks)} ingested in {elapsed:.1f}s")
    return ok


def check_store():
    state = get(f"{SIDECAR}/v1/splinter/state")
    convs = state.get("conversations", {})
    if CID in convs:
        info = convs[CID]
        print(f"  store: {info}")
    else:
        print(f"  conversations in state: {state.get('count')}")
        # Try inspect
        try:
            ins = get(f"{SIDECAR}/v1/splinter/inspect/{CID}")
            chunks = ins.get("store_chunks", ins.get("chunks", []))
            print(f"  store chunks: {len(chunks)}")
        except Exception as e:
            print(f"  inspect: {e}")


# Needle questions: (question, expected_substring)
NEEDLES = [
    ("What commit hash did RC1 land as in the splinter-memory repo?", "c9b2a58"),
    ("What is the P1-FLOOR relevance floor default threshold value?", "0.35"),
    ("What GPU model and VRAM size is this machine running?", "7900 XT"),
    ("What KV cache quantization setting is configured for llama-server?", "q4_0"),
    ("What port number does the splinter sidecar listen on?", "8765"),
    ("Which GGUF model file is currently loaded by llama-server?", "IQ4_XS"),
    ("What is the context window size in tokens for the current model config?", "150000"),
    ("What does the RC2 fix do about the similarity comparison in assembly?", "normalize"),
    ("How many chunks were kept versus dropped during the harness_state dedup?", "830"),
    ("What is the architectural principle regarding hivebench and splinter-memory repos?", "dependency direction"),
]


def run_retrieval_tests():
    print(f"\n=== Retrieval Tests ({len(NEEDLES)} needles) ===")
    results = []
    for question, expected in NEEDLES:
        try:
            resp = post(f"{SIDECAR}/v1/splinter/curate", {
                "query": question,
                "conversation_id": CID,
            }, timeout=30)
            assembled = resp.get("assembled_content", "")
            found = expected.lower() in assembled.lower()
            status = "PASS" if found else "MISS"
            results.append((question, expected, status, assembled[:200]))
            print(f"  [{status}] {question[:55]:55s} -> '{expected}'")
        except Exception as e:
            results.append((question, expected, f"ERROR", str(e)))
            print(f"  [ERR] {question[:55]:55s} -> {e}")

    passed = sum(1 for _, _, s, _ in results if s == "PASS")
    print(f"\n  Result: {passed}/{len(results)} needles retrieved")
    return results


def dump_inspection():
    """Show chunk-level detail for the last query."""
    try:
        ins = get(f"{SIDECAR}/v1/splinter/inspect/{CID}")
        print("\n=== Last Query Inspection ===")
        print(f"  mode: {ins.get('mode')}")
        print(f"  budget: {ins.get('budget')}")
        print(f"  token_count: {ins.get('token_count')}")
        sel = ins.get("selected_chunks", ins.get("chunks", []))
        if sel:
            print(f"  selected chunks ({len(sel)}):")
            for c in sel[:5]:
                cid_short = c.get("id", c.get("chunk_id", "?"))[:8]
                score = c.get("score", "?")
                content = (c.get("content", "") or "")[:80].replace("\n", " ")
                print(f"    [{cid_short}] score={score} :: {content}")
        else:
            # Try other keys
            for k in ins:
                if isinstance(ins[k], list) and len(ins[k]) > 0:
                    print(f"  {k}: {len(ins[k])} items")
            print(f"  keys: {list(ins.keys())[:15]}")
    except Exception as e:
        print(f"\n  inspection unavailable: {e}")


def main():
    print("=== Splinter Needle-in-Haystack Retrieval Test ===\n")

    print("[1/4] Resetting conversation...")
    reset_conversation()

    print("\n[2/4] Ingesting SPLINTER-CONTEXT.md via /v1/splinter/curate...")
    n = ingest_context()
    if n == 0:
        print("FATAL: no chunks ingested")
        return 1

    print("\n[3/4] Verifying store...")
    check_store()

    print("\n[4/4] Running retrieval tests...")
    results = run_retrieval_tests()

    dump_inspection()

    passed = sum(1 for _, _, s, _ in results if s == "PASS")
    total = len(results)
    print(f"\n{'='*50}")
    print(f"FINAL: {passed}/{total} needles found")
    print(f"{'='*50}")
    return 0 if passed >= total * 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
