"""Replay only accepted selected/final GRU/SAC policies on fixed fitted history."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

from rl_stocks.core.contracts import digest
from rl_stocks.core.provenance import source_identity
from rl_stocks.evaluation.offpolicy_fit import (ROOT, SOURCE_FILES, PROTOCOL_SHA256,
    accepted_cells, checksum, evaluate_fixed_window, inference_only, load_phase, protocol)
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.reporting.html import report
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.participation import diagnose_groups
from rl_stocks.training.sac_update_worker import preserved_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.artifacts.exists():
        raise ValueError('fresh summary and artifact destinations required')
    for name in SOURCE_FILES:
        if subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError('commit verified fixed protocol/implementation before replay')
    spec = protocol()
    receipt, cells = accepted_cells()
    dataset_path = ROOT/'data/datasets'/spec['dataset']/'dataset.json'
    inputs = [dataset_path.parent] + [cell['folder'] for cell in cells.values()]
    if any(p.resolve().is_relative_to(source) for p in (args.output, args.artifacts) for source in inputs):
        raise ValueError('derived outputs must be outside all inputs')
    protected = {str(dataset_path):checksum(dataset_path),
                 str(ROOT/spec['acceptance']):checksum(ROOT/spec['acceptance'])}
    for name in SOURCE_FILES:
        protected[str(ROOT/name)] = checksum(ROOT/name)
    for cell in cells.values():
        for path in cell['folder'].rglob('*'):
            if path.is_file():
                protected[str(path)] = checksum(path)
    dataset = json.loads(dataset_path.read_text())
    if digest(dataset) != spec['dataset'] or digest(dataset['normalizer']) != spec['normalizer_sha256']:
        raise ValueError('original immutable dataset and normalization required')
    stocks = set(dataset['manifest']['evaluation_symbols'])-{'SPY', 'QQQ', 'IWM', 'DIA'}
    if len(stocks) != 22:
        raise ValueError('original22-stock/four-ETF diagnostic panel required')
    normalizer = digest(dataset['normalizer'])
    package_hash = source_identity()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    torch.set_num_threads(1)
    args.artifacts.mkdir(parents=True)
    records = {}
    holding_hash = None
    with inference_only(), preserved_rng():
        for arm in spec['arms']:
            records[arm] = {}
            for phase in spec['phases']:
                model, state_hash, pointer = load_phase(arm, phase, cells[arm])
                before = state_hash()
                results = evaluate_fixed_window(model, dataset)
                if state_hash() != before or digest(dataset['normalizer']) != normalizer:
                    raise ValueError('policy/optimizer/counters/normalization changed during inference')
                groups = group_metrics(results, stocks)
                behavior = diagnose_groups(results, stocks, period='training-window')
                baselines = digest({k:v['baselines']['buy_hold'] for k,v in results.items()})
                if holding_hash is not None and baselines != holding_hash:
                    raise ValueError('identical matched holding paths required')
                holding_hash = baselines
                destination = args.artifacts/(arm+'-'+phase+'.html')
                report(results, destination, 'Fixed fitted-period diagnostic; no selection or promotion')
                comparable = (all(g['excess_return']*100 >= spec['minimum_descriptive_excess_pp']
                                  for g in groups['groups'].values())
                              and groups['groups']['stock']['median_excess_return']*100
                              >= spec['minimum_descriptive_median_stock_excess_pp'])
                records[arm][phase] = dict(checkpoint=pointer, state_before_sha256=before,
                    state_after_sha256=state_hash(), training=groups,
                    development=cells[arm]['proof'][phase], training_comparability_flag=comparable,
                    training_decliners_reporting_only=declining_stock_summary(results, stocks),
                    behavior_groups=behavior['groups'], derived_report_sha256=checksum(destination.with_suffix('.json')),
                    episodes=len(results), execution_rows=sum(len(r['rows']) for r in results.values()))
                print(arm, phase, 'fitted mean return', round(groups['score']*100, 3),
                      'descriptive comparable', comparable, flush=True)
                del model
    if source_identity() != package_hash or not all(checksum(path) == value for path,value in protected.items()):
        raise ValueError('immutable source files changed')
    result = dict(protocol_sha256=PROTOCOL_SHA256, diagnostic_source_revision=revision,
        diagnostic_package_sha256=package_hash, acceptance_receipt_sha256=receipt['receipt_sha256'],
        records=records, source_sha256=protected, normalization_sha256=normalizer,
        holding_path_sha256=holding_hash, source_files_unchanged=True,
        weights_optimizers_counters_normalization_unchanged=True, training_performed=False,
        colab_allocated=False, provider_requested=False, retired_test_access=False,
        checkpoint_selected=False, family_reopened=False, objective_achieved=False,
        inference_backend=dict(python=list(sys.version_info[:3]), torch=torch.__version__,
                               machine=platform.machine(), cpu=torch.backends.cpu.get_cpu_capability(),
                               threads=torch.get_num_threads()),
        interpretation='Eligible fitted history, not independent holdout or proof every bar was sampled. '
                       'Different periods/regimes and execution hardware limit causal claims. '
                       'Descriptive flags never change checkpoints, closed-family gates or promotion.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
