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
    if sys.version_info[:2] != (3,13):
        raise RuntimeError('Colab Python changed; align and verify the project before training')
    print('Colab/research Python:',sys.version.split()[0],flush=True)
    uv=shutil.which('uv')
    if not uv:raise RuntimeError('remote uv bootstrap tool missing')
    sys.path.insert(0,str(root/'src'))
    from rl_stocks.runtime.colab.jobs import bootstrap_commands,bootstrap_environment
    env=bootstrap_environment(os.environ,root)
    (root/'artifacts').mkdir(exist_ok=True)
    for args in bootstrap_commands(uv,root,sys.executable):
        result=subprocess.run(args,cwd=root,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True)
        with (root/'artifacts/bootstrap-log.txt').open('a') as log:
            log.write(result.stdout)
        print(result.stdout,flush=True)
        result.check_returncode()
except BaseException:
    (root/'artifacts').mkdir(exist_ok=True)
    with (root/'artifacts'/'failure.txt').open('a') as failure:
        failure.write(traceback.format_exc())
    raise
finally:
    with tarfile.open('/content/research-artifacts.partial.tar.gz','w:gz') as archive:
        if (root/'artifacts').exists():archive.add(root/'artifacts',arcname='artifacts')
    os.replace('/content/research-artifacts.partial.tar.gz','/content/research-artifacts.tar.gz')
    Path('/content/research-artifacts.ready').touch()
