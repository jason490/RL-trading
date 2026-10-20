"""Run fixed saved final policies on a preregistered fitted-period window."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.training_fit import protocol, PROTOCOL_SHA256, training_window, evaluate_training_window
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.inference.bundle import checksum
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.html import report


def weights(model):
    h=hashlib.sha256()
    for key,value in sorted(model.policy.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    if getattr(model,"log_ent_coef",None) is not None:
        h.update(model.log_ent_coef.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    args=parser.parse_args();spec=protocol()
    if args.output.exists() or args.artifacts.exists():raise ValueError('new result and artifact paths required')
    data_path=Path('data/datasets')/spec['dataset'];data_file=data_path/'dataset.json'
    protected={str(data_file):checksum(data_file)}
    data=read_dataset(data_path)
    if digest(data)!=spec['dataset']:raise ValueError('immutable source identity mismatch')
    view=training_window(data);normalizer_hash=digest(data['normalizer'])
    stocks={i['symbol'] for i in data['manifest']['universe_config']['instruments'] if i['kind']=='stock'}
    args.artifacts.mkdir(parents=True);records={};torch.set_num_threads(1)
    for algorithm,source in spec['sources'].items():
        state_path=Path(source['state']);protected[str(state_path)]=checksum(state_path)
        state=json.loads(state_path.read_text())
        if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('completed source with verified cleanup required')
        item=state['completed'][source['task']];root=Path('artifacts/colab')/item['session']/'retrieved/artifacts'
        paths=list((root/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one source run required')
        path=paths[0];m=json.loads(path.read_text());model_path=path.parent/'latest.zip';development_path=path.parent/'validation.json'
        for f in (path,model_path,development_path):protected[str(f)]=checksum(f)
        if (m['algorithm']!=algorithm or m['dataset']!=spec['dataset'] or m['actual_steps']!=spec['source_steps']
                or m['seed']!=spec['seed'] or m['smoke'] or m['status']!='complete'
                or m['observation']['profile']!=spec['observation_profile']
                or m['initialization']['arm']!=source['initialization'] or m['reward_config']!=spec['reward']
                or m['policy_architecture']['variant']!=spec['policy_variant']):
            raise ValueError('fixed saved-policy contract mismatch')
        config=SimulationConfig(**m['simulation'])
        if asdict(config)!=asdict(SimulationConfig()):raise ValueError('unchanged default costs/simulation required')
        development=json.loads(development_path.read_text());validate_development_report(development,data,config)
        model=CLASSES[algorithm].load(model_path,device='cpu');model.policy.set_training_mode(False)
        before=weights(model);updates=model._n_updates
        results=evaluate_training_window(model,view,config,spec['observation_profile'])
        if before!=weights(model) or model._n_updates!=updates or digest(data['normalizer'])!=normalizer_hash:
            raise ValueError('weights, updates or preprocessing changed')
        artifact=args.artifacts/algorithm;artifact.mkdir()
        report(results,artifact/'training.html','Training-period diagnostic only; no selection')
        groups=group_metrics(results,stocks);dev=group_metrics(development,stocks)
        thresholds=spec['benchmark_flags']
        comparable=(all(g['excess_return']*100>=thresholds['minimum_stock_and_etf_excess_pp'] for g in groups['groups'].values())
                    and groups['groups']['stock']['median_excess_return']*100>=thresholds['minimum_median_stock_excess_pp'])
        records[algorithm]=dict(run=path.parent.name,source_revision=m['git']['revision'],initialization=m['initialization'],
            model_sha256=protected[str(model_path)],weights_sha256=before,updates_unchanged=updates,
            training=groups,development=dev,training_comparability_flag=comparable,
            training_declining_stocks=declining_stock_summary(results,stocks),
            derived_report_sha256=checksum(artifact/'training.json'),
            per_symbol={r['symbol']:dict(policy_return=r['metrics']['net_return'],holding_return=r['baselines']['buy_hold']['metrics']['net_return']) for r in results.values()})
        print(algorithm,'training return',round(100*groups['score'],3),'holding',round(100*sum(r['baselines']['buy_hold']['metrics']['net_return'] for r in results.values())/len(results),3),'comparable',comparable,flush=True)
    if not all(checksum(Path(p))==h for p,h in protected.items()):raise ValueError('source files mutated')
    result=dict(protocol_sha256=PROTOCOL_SHA256,records=records,source_sha256=protected,
        diagnostic_source_sha256={p:checksum(Path(p)) for p in ['scripts/diagnose_training_fit.py','src/rl_stocks/evaluation/training_fit.py','src/rl_stocks/evaluation/runner.py']},
        training_window=dict(anchor=spec['anchor_session'],first_execution=spec['first_execution_session'],last=spec['last_session'],
            episodes=len(view['episodes']),execution_rows=sum(len(e['bars'])-1 for e in view['episodes'].values())),
        normalization_sha256=normalizer_hash,source_files_unchanged=True,weights_updates_normalization_unchanged=True,
        training_performed=False,colab_allocated=False,test_evaluated=False,checkpoint_selected=False,objective_achieved=False,
        interpretation=spec['interpretation'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
