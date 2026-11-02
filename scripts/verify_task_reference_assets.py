"""Verify retained control consistency before reference performance evaluation."""
import argparse
from pathlib import Path
import signal

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.provenance import repository_state
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.training.task_reference_assets import PROTOCOL_SHA256, protocol, verify_control
from rl_stocks.training.task_conditioning import protocol as base_protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/task-reference-assets-v1')
    args = parser.parse_args()
    root = Path.cwd()
    spec = protocol(root)
    git = repository_state(root)
    if git['dirty']:
        raise ValueError('clean committed asset-verification source required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    def expired(signum, frame):
        raise TimeoutError('bounded control-asset verification expired')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1200)
    try:
        dataset = read_dataset(root/'data/datasets'/base_protocol()['dataset'])
        cells = {}
        for role in spec['controls']:
            print('Verifying physical control:', role, flush=True)
            cells[role] = verify_control(root, dataset, role=role)
            (output / f'{role}.json').write_bytes(canonical(cells[role]))
        result = {'schema':'task-reference-control-assets-v1', 'protocol_sha256':PROTOCOL_SHA256,
                  'source':git, 'dataset':digest(dataset), 'controls':cells,
                  'training_performed':False, 'native_acceptance_changed':False,
                  'reference_performance_evaluated':False, 'development_allowed':False,
                  'colab_allocated':False, 'performance_promotion':False}
        (output/'result.json').write_bytes(canonical(result))
        print('Both original controls passed physical consistency verification', flush=True)
    finally:
        signal.alarm(0)
