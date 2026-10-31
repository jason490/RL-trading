"""Characterize fixed training-only references under unchanged simulation rules."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from rl_stocks.core.contracts import digest
from rl_stocks.core.provenance import source_identity
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.feasible_references import (
    ROOT, PROTOCOL_SHA256, SOURCE_FILES, FixedReference, protocol)
from rl_stocks.evaluation.offpolicy_fit import checksum, evaluate_fixed_window, inference_only
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.reporting.html import report
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.participation import diagnose_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.artifacts.exists():
        raise ValueError('fresh summary and artifact paths required')
    for name in SOURCE_FILES:
        if subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError('commit verified protocol and implementation before replay')
    spec = protocol()
    data_path = ROOT/'data/datasets'/spec['dataset']/'dataset.json'
    summary_path = ROOT/spec['fitted_summary']
    if checksum(summary_path) != spec['fitted_summary_sha256']:
        raise ValueError('exact accepted fitted-period receipt required')
    ledger_path = ROOT/'.runtime/usage-ledger.json'
    closure_path = ROOT/'.runtime/recurrent-comparison-screen-closure-v1.json'
    ledger = json.loads(ledger_path.read_text())
    if (any(a['status'] != 'stopped' for a in ledger['attempts'])
            or not json.loads(closure_path.read_text())['family_closed']):
        raise ValueError('stopped research source and preserved family closure required')
    if any(p.resolve().is_relative_to(data_path.parent) for p in (args.output, args.artifacts)):
        raise ValueError('derived outputs outside immutable dataset required')
    paths = [data_path, summary_path, ledger_path, closure_path]+[ROOT/name for name in SOURCE_FILES]
    protected = {str(p):checksum(p) for p in paths}
    package_hash = source_identity()
    data = json.loads(data_path.read_text())
    if digest(data) != spec['dataset'] or digest(data['normalizer']) != spec['normalizer_sha256']:
        raise ValueError('original immutable dataset/normalizer required')
    stocks = set(data['manifest']['evaluation_symbols'])-{'SPY', 'QQQ', 'IWM', 'DIA'}
    if len(stocks) != 22:
        raise ValueError('fixed22-stock/four-ETF panel required')
    args.artifacts.mkdir(parents=True)
    records = {}
    with inference_only():
        for name, action in spec['actions'].items():
            model = FixedReference(name)
            results = evaluate_fixed_window(model, data)
            if model.action != tuple(action) or any(
                    tuple(row['policy_action']) != tuple(action)
                    for result in results.values() for row in result['rows']):
                raise ValueError('unconditional fixed requests must remain unchanged')
            if digest({k:v['baselines']['buy_hold'] for k,v in results.items()}) != spec['holding_path_sha256']:
                raise ValueError('identical holding paths to fitted-policy diagnosis required')
            groups = group_metrics(results, stocks)
            diagnosis = diagnose_groups(results, stocks, period='training-window')
            if name == 'cash' and any(r['metrics']['net_return'] != 0 or r['metrics']['costs'] != 0
                                      or r['events'] for r in results.values()):
                raise ValueError('cash reference must remain flat without costs or fills')
            destination = args.artifacts/(name+'.html')
            report(results, destination, 'Fixed training reference; no learning, selection or promotion')
            records[name] = dict(action=action, training=groups, behavior_groups=diagnosis['groups'],
                decliners_reporting_only=declining_stock_summary(results, stocks),
                episodes=len(results), execution_rows=sum(len(r['rows']) for r in results.values()),
                derived_report_sha256=checksum(destination.with_suffix('.json')))
            print(name, 'training mean return', round(groups['score']*100, 3), flush=True)
    if source_identity() != package_hash or not all(checksum(p) == value for p,value in protected.items()):
        raise ValueError('protected sources, ledger or closure changed')
    result = dict(protocol_sha256=PROTOCOL_SHA256,
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        package_sha256=package_hash, source_sha256=protected, records=records,
        simulation=asdict(SimulationConfig()), normalization_sha256=digest(data['normalizer']),
        holding_path_sha256=spec['holding_path_sha256'], source_files_unchanged=True,
        project_units_unchanged=ledger['consumed'], training_performed=False,
        learned_models_loaded=False, colab_allocated=False, provider_requested=False,
        development_replayed=False, retired_test_access=False, policy_selected=False,
        family_reopened=False, objective_achieved=False,
        interpretation='Unconditional descriptive references, not an optimal attainable-return bound. '
                       'Boundary geometry remains active and entries can be clipped/rejected. '
                       'No direction chosen from outcomes; decliners reporting-only. '
                       'No trained policy, simulation rule, checkpoint or promotion changes.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
