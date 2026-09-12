"""Detach-launch the harness so it survives sandbox tool-call reaping."""
import os, sys
pid = os.fork()
if pid == 0:
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    log = open("harness-detached.log", "ab")
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    # tokenizers (Rust) global rayon pool cannot survive fork+re-init;
    # disable its internal parallelism (CPU MiniLM doesn't need it).
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "harness"])
# parent: wait briefly to confirm the child is up
import time, urllib.request
time.sleep(8)
try:
    r = urllib.request.urlopen("http://127.0.0.1:8765/", timeout=5)
    print(f"detached harness up (launched pid {pid})")
except Exception as e:
    print("launch check failed:", e)
