import sys
from hiveclient import turn

CONV = "long-3"
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

start, end = int(sys.argv[1]), int(sys.argv[2])
for n in range(start, end + 1):
    content = FACTS.get(n) or FILLERS[(n - 1) % len(FILLERS)]
    tag = "FACT" if n in FACTS else "fill"
    try:
        a, dt = turn(CONV, content)
        print(f"T{n:>2} [{tag}] {dt:.1f}s :: {a[:60].strip()}", flush=True)
    except Exception as e:
        print(f"T{n:>2} ERROR: {e}", flush=True)
