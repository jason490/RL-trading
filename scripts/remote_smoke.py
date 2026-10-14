"""Bootstrap an isolated research interpreter via the official Colab CLI."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback

root=Path('/content/rl-stocks')
root.mkdir(exist_ok=True)
try:
    with tarfile.open('/content/research-upload.tar.gz') as archive:
        archive.extractall(root,filter='data')
    env=dict(os.environ, UV_CACHE_DIR='/content/uv-cache',
             UV_PYTHON_INSTALL_DIR='/content/python', MPLCONFIGDIR='/content/matplotlib',
             RL_CHECKPOINT_ARCHIVE='/content/research-checkpoint.tar.gz')
    uv=shutil.which('uv')
    if not uv:raise RuntimeError('remote uv bootstrap tool missing')
    sys.path.insert(0,str(root/'src'))
    from rl_stocks.runtime.colab.jobs import bootstrap_commands
    for args in bootstrap_commands(uv,root):
        subprocess.run(args,cwd=root,env=env,check=True)
except BaseException:
    (root/'artifacts').mkdir(exist_ok=True)
    (root/'artifacts'/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    with tarfile.open('/content/research-artifacts.tar.gz','w:gz') as archive:
        if (root/'artifacts').exists():archive.add(root/'artifacts',arcname='artifacts')
