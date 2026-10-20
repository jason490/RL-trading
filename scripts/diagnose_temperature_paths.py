"""Compare saved SAC temperature paths; no fitting, dataset access or selection."""
import argparse
import json
from pathlib import Path
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.participation import diagnose_groups
from rl_stocks.training.sac_temperature import PROTOCOL_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=Path('.runtime/temperature-screen-state.json'))
    parser.add_argument('--summary', type=Path, default=Path('results/sac-temperature-2026-10-20/summary.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    state = json.loads(args.state.read_text())
    summary = json.loads(args.summary.read_text())
    if state['status'] != 'complete' or not state['cleanup_verified']:
        raise ValueError('completed screen with verified cleanup required')
    if (summary['protocol_sha256'] != PROTOCOL_SHA256
            or not summary['control_reproduction']['passed'] or summary['test_evaluated']):
        raise ValueError('verified development-only reproduction required')
    universe = Path('configs/universes/sector_research.json')
    instruments = json.loads(universe.read_text())['instruments']
    stocks = {i['symbol'] for i in instruments if i['kind'] == 'stock'}
    symbols = {i['symbol'] for i in instruments if i['role'] == 'tradable'}
    protected = {str(p): checksum(p) for p in (args.state, args.summary, universe)}
    records = {}
    for arm, cell in summary['cells'].items():
        item = state['completed'][f'temperature-{arm}-11']
        run = Path(item['root']) / 'runs' / cell['run']
        metadata_path = run / 'metadata.json'
        protected[str(metadata_path)] = checksum(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        if (metadata['algorithm'] != 'sac' or metadata['seed'] != 11
                or metadata['actual_steps'] != 50000 or metadata['smoke']
                or metadata['dataset'] != cell['dataset'] or metadata['status'] != 'complete'
                or metadata['study_protocol_sha256'] != PROTOCOL_SHA256):
            raise ValueError('fixed completed 50k SAC screen required')
        records[arm] = {}
        for phase, contract in cell['reports'].items():
            path = run / contract['file']
            protected[str(path)] = checksum(path)
            if protected[str(path)] != contract['sha256']:
                raise ValueError('saved report checksum mismatch')
            results = json.loads(path.read_text())
            if {r['symbol'] for r in results.values()} != symbols:
                raise ValueError('configured stock/ETF panel required')
            records[arm][phase] = diagnose_groups(results, stocks)
    if set(records) != {'default', 'cold_start'}:
        raise ValueError('both declared arms required')
    changes = {}
    for arm, phases in records.items():
        changes[arm] = {}
        for kind in ('stock', 'etf'):
            early = phases['selected']['groups'][kind]['equal_weight_means']
            late = phases['final']['groups'][kind]['equal_weight_means']
            changes[arm][kind] = {
                'net_return_change_pp': 100 * (late['net_return'] - early['net_return']),
                'long_contribution_change_pp': 100 * (late['long_return_contribution'] - early['long_return_contribution']),
                'short_contribution_change_pp': 100 * (late['short_return_contribution'] - early['short_return_contribution']),
                'cost_drag_change_pp': late['incremental_cost_drag_pp'] - early['incremental_cost_drag_pp'],
                'saved_path_noncost_change_pp': late['saved_path_noncost_excess_pp'] - early['saved_path_noncost_excess_pp'],
                'long_exposure_change': late['mean_long_exposure'] - early['mean_long_exposure'],
                'short_exposure_change': late['mean_short_abs_exposure'] - early['mean_short_abs_exposure'],
            }
    if not all(checksum(Path(p)) == h for p, h in protected.items()):
        raise ValueError('source files mutated')
    result = dict(records=records, selected_to_final_changes=changes,
                  source_sha256=protected, diagnostic_script_sha256=checksum(Path(__file__)),
                  source_files_unchanged=True, dataset_opened=False, training_performed=False,
                  colab_allocated=False, test_evaluated=False, objective_achieved=False,
                  interpretation='Saved-path accounting only. Cost add-back is not a zero-cost replay or causal attribution. Long/short contributions already include costs; do not subtract them twice. No checkpoint selection or promotion.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(changes, indent=2))


if __name__ == '__main__':
    main()
