"""Research worker; never relies on the Colab notebook kernel's Python."""
import json
from pathlib import Path
import sys
import traceback

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))

def main():
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError('Colab-matched Python 3.13 required')
    import torch
    cuda=torch.cuda.is_available()
    gpu=torch.cuda.get_device_name() if cuda else None
    print('Runtime:',sys.version.split()[0],'Torch:',torch.__version__,
          'CUDA available:',cuda,'GPU:',gpu,flush=True)
    if not cuda or 'A100' not in gpu:
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


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (root/'artifacts').mkdir(exist_ok=True)
        (root/'artifacts/failure.txt').write_text(traceback.format_exc())
        raise
