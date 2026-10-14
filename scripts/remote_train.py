"""Research worker; never relies on the Colab notebook kernel's Python."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('Python 3.12 required')
import torch
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
    raise RuntimeError('A100 hardware not verified')
from rl_stocks.core.contracts import digest
from rl_stocks.runtime.colab.jobs import job_plan
from rl_stocks.training.runner import train

payload = json.loads((root/'dataset.json').read_text())
dataset = root/'data'/digest(payload)
dataset.mkdir(parents=True, exist_ok=True)
(root/'dataset.json').rename(dataset/'dataset.json')
plan = json.loads((root/'job-plan.json').read_text())
research = plan['jobs'][-1]['algorithm'] if not plan['jobs'][-1]['smoke'] else None
if plan != job_plan(research):
    raise ValueError('unsupported remote job plan')
completed = []
for job in plan['jobs']:
    overrides = {'n_steps': 1000, 'batch_size': 50} if job['algorithm']=='ppo' and not job['smoke'] else None
    print('Starting job:', job, flush=True)
    run, bundle = train(dataset, seed=plan['seed'], overrides=overrides, **job)
    completed.append({'run': str(run), 'bundle_id': bundle.name})
    print('Completed job:', str(run), 'bundle:', str(bundle), flush=True)
(root/'artifacts/completion.json').write_text(json.dumps({
    'plan': plan, 'dataset': dataset.name, 'python': list(sys.version_info[:3]),
    'gpu': torch.cuda.get_device_name(), 'completed': completed}, indent=2))
