import json, time, urllib.request

BASE = "http://127.0.0.1:8765"
MODEL = "unsloth/Qwen3.8-27B-GGUF"
CONV = "long-1"

FACTS = {
    3:  "Important detail to remember: my dog's name is Biscuit and he was born in March 2019.",
    8:  "Another fact: I am allergic to penicillin. Doctor confirmed it last year.",
    15: "Remember this too: the password for my home router is tulip-orchid-42.",
    25: "One more thing to keep in mind: my sister's birthday is November 3rd.",
    35: "Final fact: I drive a green Volvo XC60 with license plate KLM-789.",
}

FILLERS = [
    "What's a good way to organize a small apartment kitchen?",
    "Explain the difference between TCP and UDP in one short paragraph.",
    "Suggest three low-maintenance houseplants for a dark room.",
    "What are the main causes of slow Wi-Fi at home?",
    "Give me a simple recipe that uses only eggs, bread, and cheese.",
    "What's the capital of Australia? Just the name.",
    "How do I sharpen kitchen knives without a special tool?",
    "Name two benefits of walking daily.",
]

def turn(content):
    body = json.dumps({"model": MODEL, "messages":[{"role":"user","content":content}]}).encode()
    r = urllib.request.Request(f"{BASE}/v1/openai/chat/completions", data=body,
        headers={"Content-Type":"application/json","X-Strata-Conversation":CONV})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=600) as resp:
        d = json.load(resp)
    return d["choices"][0]["message"]["content"], time.time()-t0

results = []
for n in range(1, 42):
    if n in FACTS:
        content = FACTS[n]
    else:
        content = FILLERS[(n-1) % len(FILLERS)]
    try:
        a, dt = turn(content)
        tag = "FACT" if n in FACTS else "fill"
        print(f"T{n:>2} [{tag}] {dt:.1f}s :: {a[:70].strip()}", flush=True)
    except Exception as e:
        print(f"T{n:>2} ERROR: {e}", flush=True)

print("=== RECALL BATTERY ===")
checks = [
    ("What is my dog's name and when was he born?", "Biscuit"),
    ("What am I allergic to?", "penicillin"),
    ("What is the password for my home router?", "tulip-orchid-42"),
    ("When is my sister's birthday?", "November 3"),
    ("What car do I drive and what's the license plate?", "KLM-789"),
]
score = 0
for q, needle in checks:
    a, dt = turn(q)
    ok = needle.lower() in a.lower()
    score += ok
    print(f"{'PASS' if ok else 'FAIL'} ({dt:.1f}s) Q: {q} | A: {a[:90].strip()}")
print(f"SCORE: {score}/5")
