"""Bounded bootstrap on the already-inspected default Colab kernel."""

import os
import subprocess
import sys

if sys.version_info[:2] != (3, 13) or not os.environ.get("COLAB_RELEASE_TAG"):
    raise RuntimeError("inspected default Colab Python3.13 required")
subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.13.1"], check=True, timeout=150)
print("Default-kernel bootstrap complete", flush=True)
