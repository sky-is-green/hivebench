#!/usr/bin/env python3
"""Install CPU-only torch into the venv (shadows Studio's ROCm build).

Run from any normal terminal (NOT the AI sandbox, which caps files at 100 MiB):

    cd ~/Desktop/work/hivebench && venv/bin/python install_cpu_torch.py

What it does:
  1. Downloads torch-2.6.0+cpu wheel (~187 MiB) into this directory (resumable).
  2. Extracts it straight into venv/lib/python3.13/site-packages/ -- no pip,
     so nothing in ~/.unsloth/studio is touched or uninstalled.
  3. Verifies `import torch` now resolves to the venv CPU build.
"""
import os, sys, subprocess, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WHEEL = "torch-2.6.0+cpu-cp313-cp313-linux_x86_64.whl"
URL = "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp313-cp313-linux_x86_64.whl"
SITE = os.path.join(HERE, "venv", "lib", "python3.13", "site-packages")

os.chdir(HERE)

# 1) download (resumable)
if not os.path.exists(WHEEL):
    done = 0
    for attempt in range(5):
        headers = {"User-Agent": "Mozilla/5.0"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if (done and r.status == 206) else ("wb" if r.status == 200 else "ab")
                if r.status == 200:
                    done = 0
                with open(WHEEL, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        done += len(b)
            print(f"download complete: {os.path.getsize(WHEEL)//1024//1024} MiB")
            break
        except Exception as e:
            print(f"attempt {attempt+1} failed at {done//1024} KiB: {e!r}")
    else:
        sys.exit("download failed after 5 attempts; re-run to resume")
else:
    print(f"wheel already present: {os.path.getsize(WHEEL)//1024//1024} MiB")

# 2) extract into venv site-packages (new names only: torch/, torch-*.dist-info/)
print("extracting into venv site-packages...")
with zipfile.ZipFile(WHEEL) as z:
    z.extractall(SITE)

# 3) verify the shadow works from a fresh interpreter
check = ("import torch; assert 'cpu' in torch.__version__ and '/venv/' in torch.__file__, "
         "(torch.__version__, torch.__file__); print('OK torch', torch.__version__)")
p = subprocess.run([sys.executable, "-c", check])
sys.exit(p.returncode)
