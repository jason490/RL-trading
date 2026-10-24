"""Accept only completed, retrieved and owned-cleaned Colab critic smokes."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.inference.bundle import checksum
from rl_stocks.runtime.colab.jobs import critic_plan
from rl_stocks.training.distributional_critics import (
    PROTOCOL_SHA256, protocol, validate_initial_comparison, validate_payload,
)
from rl_stocks.training.distributional_restoration import verify_owned_run
from rl_stocks.training.distributional_runtime import projection


def accept(state_path, source_proof_path, data_path):
    state = json.loads(Path(state_path).read_text())
    source = json.loads(Path(source_proof_path).read_text())
    if state['status'] != 'complete' or state.get('cleanup_verified') is not True:
        raise ValueError('complete critic smokes with verified owned termination required')
    spec = protocol()
    expected_ids = {f'critic-{arm}-11-smoke' for arm in spec['arms']}
    if set(state['completed']) != expected_ids:
        raise ValueError('exactly three fixed owned critic smoke tasks required')
    data = validate_payload(read_dataset(data_path))
    records = {}
    for arm in spec['arms']:
        item = state['completed'][f'critic-{arm}-11-smoke']
        root = Path(item['root'])
        paths = list((root / 'runs').glob('*/metadata.json'))
        if len(paths) != 1:
            raise ValueError('one owned critic run required')
        metadata = json.loads(paths[0].read_text())
        if (metadata['git']['revision'] != source['source_revision'] or metadata['git']['dirty']
                or metadata['source_hash'] != source['source_hash'] or metadata['study_protocol_sha256'] != PROTOCOL_SHA256
                or metadata['dataset'] != spec['dataset'] or metadata['smoke'] is not True
                or metadata['seed'] != 11 or metadata['actual_steps'] != 1024):
            raise ValueError('actual clean fixed critic smoke source/steps required')
        completion = json.loads((root / 'completion.json').read_text())
        if completion['plan'] != critic_plan(arm, smoke=True):
            raise ValueError('actual critic smoke completion plan mismatch')
        config = SimulationConfig(**metadata['simulation'])
        for filename in ('selected-validation.json', 'validation.json'):
            validate_development_report(json.loads((paths[0].parent / filename).read_text()), data, config)
        bundle = root / 'frozen' / metadata['bundle_id']
        restored = verify_owned_run(paths[0].parent, bundle, arm, smoke=True)
        records[arm] = dict(session=item['session'], run=paths[0].parent.name,
            bundle_id=metadata['bundle_id'], duration_seconds=metadata['duration_seconds'],
            initial=metadata['critic_initialization'], metadata_sha256=checksum(paths[0]),
            python=metadata['python'], **restored)
    initial = validate_initial_comparison({arm: r['initial'] for arm, r in records.items()}, True)
    return dict(schema='distributional-smoke-acceptance-v1', passed=True, protocol_sha256=PROTOCOL_SHA256,
                source_proof=source, state_sha256=checksum(state_path), cleanup_verified=True,
                initialization_checks=initial, records=records, exact_frozen_predictions=192,
                runtime_projection=projection({a: r['duration_seconds'] for a, r in records.items()}),
                training_performed=False, optimizer_steps=0, test_evaluated=False, screen_allocated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'source-proof', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = accept(args.state, args.source_proof, args.dataset)
    write_once(args.output, (json.dumps(result, indent=2) + '\n').encode())
    print('Accepted3 owned-cleaned1024-transition smokes and192 exact frozen predictions; no screen allocated.')


if __name__ == '__main__':
    main()
