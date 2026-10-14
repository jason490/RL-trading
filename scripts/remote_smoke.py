"""Executed by official `colab exec`; repository payload uploaded separately."""
import os
from pathlib import Path
import sys
import tarfile
import traceback

root=Path('/content/rl-stocks')
root.mkdir(exist_ok=True)
with tarfile.open('/content/research-upload.tar.gz') as archive:
    archive.extractall(root,filter='data')
os.chdir(root)
os.environ['RL_CHECKPOINT_ARCHIVE']='/content/research-checkpoint.tar.gz'
sys.path.insert(0,str(root/'src'))
try:
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Python 3.12 required')
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('A100 hardware not verified')
    from rl_stocks.core.contracts import digest
    import json
    payload=json.loads((root/'dataset.json').read_text())
    dataset=root/'data'/digest(payload)
    dataset.mkdir(parents=True,exist_ok=True)
    (root/'dataset.json').rename(dataset/'dataset.json')
    from rl_stocks.training.runner import train
    # CPU profile first on the same data; A100 smoke then runs each algorithm.
    train(dataset,'ppo',seed=11,device='cpu',smoke=True,steps=512)
    for algorithm,steps in [('ppo',512),('sac',1024),('td3',1024)]:
        train(dataset,algorithm,seed=11,device='cuda',smoke=True,steps=steps)
except BaseException:
    (root/'artifacts').mkdir(exist_ok=True)
    (root/'artifacts'/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    with tarfile.open('/content/research-artifacts.tar.gz','w:gz') as archive:
        if (root/'artifacts').exists():archive.add(root/'artifacts',arcname='artifacts')
