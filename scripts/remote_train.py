"""Research worker; never relies on the Colab notebook kernel's Python."""
import json
import os
from pathlib import Path
import sys
import traceback

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))

def main():
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError('Colab-matched Python 3.13 required')
    from rl_stocks.runtime.colab.jobs import validate_plan,bootstrap_environment
    os.environ.update(bootstrap_environment(os.environ,root))
    plan = json.loads((root/'job-plan.json').read_text())
    validate_plan(plan)
    import torch
    cuda=torch.cuda.is_available()
    gpu=torch.cuda.get_device_name() if cuda else None
    print('Runtime:',sys.version.split()[0],'Torch:',torch.__version__,
          'CUDA available:',cuda,'GPU:',gpu,flush=True)
    if (plan['accelerator']=='CPU' and cuda or
        plan['accelerator']!='CPU' and (not cuda or plan['accelerator'] not in gpu)):
        raise RuntimeError('requested runtime hardware not verified')
    from rl_stocks.core.contracts import digest
    from rl_stocks.training.runner import train

    payload = json.loads((root/'dataset.json').read_text())
    if plan.get('version')==20:
        from rl_stocks.training.sac_regularization import validate_payload
        validate_payload(payload)
    if plan.get('version')==19:
        from rl_stocks.training.diversity import validate_payload
        validate_payload(payload,plan['diversity_study']['arm'])
    if plan.get('version')==18:
        from rl_stocks.training.td3_state import validate_payload
        validate_payload(payload)
    if plan.get('version')==17:
        from rl_stocks.training.sac_temperature import validate_payload
        validate_payload(payload)
    if plan.get('version')==16:
        from rl_stocks.training.sac_state import validate_payload
        validate_payload(payload)
    if plan.get('version')==15:
        from rl_stocks.training.history.spec import validate_payload
        validate_payload(payload,plan['history_study']['arm'])
    if plan.get('version')==12:
        from rl_stocks.runtime.colab.jobs import validate_context_payload
        if digest(payload)!=plan['dataset_contract']['identity']:raise ValueError('remote trend-context dataset identity mismatch')
        validate_context_payload(payload,plan)
    dataset = root/'data'/digest(payload)
    dataset.mkdir(parents=True, exist_ok=True)
    (root/'dataset.json').rename(dataset/'dataset.json')
    completed = []
    for job in plan['jobs']:
        overrides = {'n_steps': 1000, 'batch_size': 50} if job['algorithm'] in ('ppo','recurrent_ppo') and not job['smoke'] else None
        print('Starting job:', job, flush=True)
        seed = job.get('seed', plan.get('seed'))
        parameters = {k:v for k,v in job.items() if k not in ('seed','overrides')}
        if 'overrides' in job:overrides=job['overrides']
        if plan.get('version')==14:
            from rl_stocks.training.supervised.forest_runner import run as fit_forest
            os.environ['RL_COLAB_OWNED_WORKER']='1'
            run,bundle=fit_forest(dataset,root/'artifacts')
        elif plan.get('version')==13:
            from rl_stocks.training.supervised.runner import run as fit_signal
            os.environ['RL_COLAB_OWNED_WORKER']='1'
            run,bundle=fit_signal(dataset,root/'artifacts')
        else:
            run, bundle = train(dataset, seed=seed, overrides=overrides, **parameters)
        completed.append({'run': str(run), 'bundle_id': bundle.name})
        print('Completed job:', str(run), 'bundle:', str(bundle), flush=True)
        # Preserve all completed jobs between minute-metered transfers.
        import tarfile
        checkpoint = Path(os.environ['RL_CHECKPOINT_ARCHIVE'])
        temporary = checkpoint.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as archive:
            archive.add(root/'artifacts', arcname='artifacts')
        temporary.replace(checkpoint)
    (root/'artifacts/completion.json').write_text(json.dumps({
        'plan': plan, 'dataset': dataset.name, 'python': list(sys.version_info[:3]),
        'gpu': gpu, 'completed': completed}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (root/'artifacts').mkdir(exist_ok=True)
        (root/'artifacts/failure.txt').write_text(traceback.format_exc())
        raise
