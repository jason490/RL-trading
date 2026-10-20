"""Replay fixed final100k SAC policies on the existing common fitted window."""
import json
from pathlib import Path
import torch
from diagnose_training_fit import weights
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.training_fit import training_window, evaluate_training_window
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.training_regimes import summarize_training_regimes
from rl_stocks.reporting.html import report
from rl_stocks.rl.registry.factory import CLASSES


def main():
    spec_path=Path('configs/experiments/diversity_fit_diagnostic.json')
    spec=json.loads(spec_path.read_text())
    output=Path('results/sac-diversity-fit-2026-10-20/summary.json')
    artifacts=Path('artifacts/sac-diversity-fit-2026-10-20')
    if output.exists() or artifacts.exists():raise ValueError('new output paths required')
    state_path=Path(spec['source_state']);summary_path=Path(spec['source_summary'])
    state=json.loads(state_path.read_text());summary=json.loads(summary_path.read_text())
    if state['status']!='complete' or not state['cleanup_verified'] or summary['test_evaluated']:
        raise ValueError('completed cleaned development source required')
    data_path=Path('data/datasets')/spec['dataset']
    protected={str(p):checksum(p) for p in (spec_path,state_path,summary_path,data_path/'dataset.json')}
    data=read_dataset(data_path)
    if digest(data)!=spec['dataset']:raise ValueError('immutable dataset identity mismatch')
    view=training_window(data);normalizer=digest(data['normalizer'])
    stocks={i['symbol'] for i in data['manifest']['universe_config']['instruments'] if i['kind']=='stock'}
    records={};torch.set_num_threads(1);artifacts.mkdir(parents=True)
    for arm in spec['arms']:
        cell=summary['cells'][arm];item=state['completed'][f'diversity-{arm}-11']
        run=Path(item['root'])/'runs'/cell['run']
        metadata_path=run/'metadata.json';model_path=run/'latest.zip'
        for p in (metadata_path,model_path):protected[str(p)]=checksum(p)
        m=json.loads(metadata_path.read_text())
        if (m['algorithm']!='sac' or m['actual_steps']!=100000 or m['seed']!=11
            or m['smoke'] or m['status']!='complete' or m['dataset']!=cell['dataset']
            or m['observation']['profile']!='execution_full'):
            raise ValueError('fixed final100k full-state SAC required')
        model=CLASSES['sac'].load(model_path,device='cpu');model.policy.set_training_mode(False)
        before=weights(model);updates=model._n_updates
        results=evaluate_training_window(model,view,SimulationConfig(**m['simulation']),'execution_full')
        if weights(model)!=before or model._n_updates!=updates or digest(data['normalizer'])!=normalizer:
            raise ValueError('weights/updates/normalization mutated')
        directory=artifacts/arm;directory.mkdir()
        report(results,directory/'training.html','Fitted-period diagnostic only; no selection')
        groups=group_metrics(results,stocks)
        records[arm]=dict(run=cell['run'],training=groups,regimes=summarize_training_regimes(results),
            model_sha256=protected[str(model_path)],weights_sha256=before,
            updates_unchanged=updates,derived_report_sha256=checksum(directory/'training.json'))
        print(arm,json.dumps(groups),flush=True)
    if not all(checksum(Path(p))==h for p,h in protected.items()):raise ValueError('sources mutated')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(protocol_sha256=digest(spec),records=records,source_sha256=protected,
        script_sha256=checksum(Path(__file__)),normalization_sha256=normalizer,
        source_files_unchanged=True,weights_updates_normalization_unchanged=True,
        training_performed=False,colab_allocated=False,test_evaluated=False,
        checkpoint_selected=False,objective_achieved=False,interpretation=spec['interpretation']),indent=2)+'\n')


if __name__=='__main__':main()
