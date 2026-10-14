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
    from rl_stocks.runtime.colab.jobs import job_plan
    plan=json.loads((root/'job-plan.json').read_text())
    research=plan['jobs'][-1]['algorithm'] if not plan['jobs'][-1]['smoke'] else None
    if plan!=job_plan(research):raise ValueError('unsupported remote job plan')
    for job in plan['jobs']:
        overrides={'n_steps':1000,'batch_size':50} if job['algorithm']=='ppo' and not job['smoke'] else None
        print('Starting job:',job,flush=True)
        run,bundle=train(dataset,seed=plan['seed'],overrides=overrides,**job)
        print('Completed job:',str(run),'bundle:',str(bundle),flush=True)
except BaseException:
    (root/'artifacts').mkdir(exist_ok=True)
    (root/'artifacts'/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    with tarfile.open('/content/research-artifacts.tar.gz','w:gz') as archive:
        if (root/'artifacts').exists():archive.add(root/'artifacts',arcname='artifacts')
