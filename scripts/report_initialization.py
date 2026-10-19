"""Summarize the complete, matched four-arm Colab initialization screen."""
import argparse
import json
import math
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.initialization import group_metrics, screen_gate
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.training.initialization import initialization_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('configs/experiments/action_initialization.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    state = json.loads(args.state.read_text())
    if state['status'] != 'complete' or not state['cleanup_verified']:
        raise ValueError('complete screen and verified owned cleanup required')
    universe = json.loads(Path(protocol['universe']).read_text())
    stocks = {i['symbol'] for i in universe['instruments'] if i['kind'] == 'stock'}
    cells, contract, baselines = {}, None, None
    for item in state['completed'].values():
        paths = list((Path(item['root']) / 'runs').glob('*/metadata.json'))
        if len(paths) != 1:
            raise ValueError('one run per initialization task required')
        path = paths[0]
        meta = json.loads(path.read_text())
        arm = meta['initialization']['arm']
        if (arm not in protocol['arms'] or arm in cells or meta['seed'] != protocol['screen_seed']
                or meta['status'] != 'complete' or meta['smoke'] or meta['algorithm'] != 'ppo'
                or meta['requested_steps'] != protocol['steps_per_cell']
                or meta['actual_steps'] != protocol['steps_per_cell']
                or meta['dataset'] != protocol['dataset'] or meta['device'] != 'cpu'
                or meta['initialization'] != initialization_spec(arm)):
            raise ValueError('unexpected or duplicate screen cell')
        current = {key: meta[key] for key in ('dataset', 'source_hash', 'git', 'simulation',
                    'versions', 'resolved_parameters', 'reward_config', 'evaluation_every')}
        if meta['git']['dirty'] != '' or (contract is not None and current != contract):
            raise ValueError('clean, matched source and configuration required')
        contract = current
        cell = {'seed': meta['seed'], 'bundle_id': meta['bundle_id'], 'run': path.parent.name,
                'initialization': meta['initialization'], 'reports': {}}
        for phase, filename in [('selected', 'selected-validation.json'), ('final', 'validation.json')]:
            report_path = path.parent / filename
            results = json.loads(report_path.read_text())
            if any(r['split'] != 'validation' for r in results.values()):
                raise ValueError('validation reports required')
            current_baselines = {key: r['baselines']['buy_hold']['metrics'] for key, r in results.items()}
            if baselines is not None and current_baselines != baselines:
                raise ValueError('baseline panel mismatch')
            baselines = current_baselines
            cell[phase] = group_metrics(results, stocks)
            cell[phase]['declining_stocks'] = declining_stock_summary(results, stocks)
            cell['reports'][phase] = checksum(report_path)
            expected = meta['selected_score'] if phase == 'selected' else meta['validation_checkpoints'][-1]['score']
            if not math.isclose(cell[phase]['score'], expected, abs_tol=1e-10):
                raise ValueError('report and checkpoint score mismatch')
        if meta['validation_checkpoints'][-1]['phase'] != 'post_update':
            raise ValueError('final post-update endpoint required')
        cells[arm] = cell
    comparison = screen_gate(cells, protocol)
    output = {'protocol_sha256': digest(protocol), 'protocol': protocol,
              'matched_contract': contract, 'cells': cells, 'screen': comparison,
              'objective_achieved': False,
              'interpretation': 'Development screen only; no independent profitability evidence'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
