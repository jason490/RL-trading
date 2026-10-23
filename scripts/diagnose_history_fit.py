"""Fixed saved recurrent policies on eligible training history; no fitting."""
import argparse,json,subprocess
from pathlib import Path
import torch
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.training_fit import (PROTOCOL_SHA256 as WINDOW_SHA,protocol as window_protocol,
    training_window,evaluate_common_recurrent_training_window)
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.optimizer import optimizer_record
from rl_stocks.reporting.html import report
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.training_history import (protocol as source_protocol,validate_payload,modes,
    model_record,validate_record)

PROTOCOL_SHA256='e319c26f6cddecfaf3b2b3e847941c77c2bdf0e0ce8d7f8e466353f6b14a3143'
SOURCE_FILES=('configs/experiments/history_fit_diagnostic.json','scripts/diagnose_history_fit.py',
              'src/rl_stocks/evaluation/training_fit.py','src/rl_stocks/evaluation/runner.py',
              'src/rl_stocks/reporting/optimizer.py','src/rl_stocks/training/training_history.py')


def protocol():
    spec=json.loads(Path('configs/experiments/history_fit_diagnostic.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed recurrent-fit protocol changed')
    if spec['window_protocol_sha256']!=WINDOW_SHA or any(spec[k]!=window_protocol()[k] for k in ('anchor_session','first_execution_session','last_session')):
        raise ValueError('unchanged common fitted window required')
    return spec


from rl_stocks.reporting.optimizer import optimizer_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.artifacts.exists():raise ValueError('new result/artifact paths required')
    spec=protocol();root=Path.cwd().resolve()
    for name in SOURCE_FILES:
        committed=subprocess.check_output(['git','show','HEAD:'+name])
        if committed!=(root/name).read_bytes():raise ValueError('commit verified fixed protocol/implementation before replay')
    inventory_path=Path(spec['source_inventory']);inventory=json.loads(inventory_path.read_text());protected={}
    def protect(path,accepted=False):
        path=Path(path);relative=str(path.resolve().relative_to(root));h=checksum(path)
        if accepted and inventory.get(relative)!=h:raise ValueError('accepted retained-source hash mismatch')
        protected[relative]=h
    for name in [*SOURCE_FILES,spec['source_state'],spec['source_summary'],spec['source_inventory']]:protect(name)
    state=json.loads(Path(spec['source_state']).read_text());accepted=json.loads(Path(spec['source_summary']).read_text())
    if (state['status']!='complete' or not state['cleanup_verified'] or accepted['test_evaluated']
            or set(accepted['cells'])!=set(spec['arms']) or set(spec['arms'])!=set(source_protocol()['arms'])):
        raise ValueError('complete accepted cleaned development source required')
    datasets={}
    for arm in spec['arms']:
        data_path=Path('data/datasets')/spec['datasets'][arm];protect(data_path/'dataset.json',True)
        datasets[arm]=validate_payload(read_dataset(data_path),arm)
    from rl_stocks.evaluation.training_fit import common_training_views
    views=common_training_views(datasets)
    normalizer=digest(datasets['full']['normalizer'])
    stocks={i['symbol'] for i in datasets['full']['manifest']['universe_config']['instruments'] if i['kind']=='stock'}
    if digest(source_protocol())!=spec['source_protocol_sha256']:raise ValueError('fixed source protocol required')
    records={};baseline=None;torch.set_num_threads(1);a.artifacts.mkdir(parents=True)
    for arm in spec['arms']:
        data=datasets[arm];view=views[arm]
        cell=accepted['cells'][arm];item=state['completed'][f'recency-{arm}-11']
        run=Path(item['root'])/'runs'/cell['run'];mp=run/'metadata.json';protect(mp,True);m=json.loads(mp.read_text())
        if (m['status']!='complete' or m['smoke'] or m['seed']!=spec['seed'] or m['actual_steps']!=spec['source_final_steps']
                or m['dataset']!=spec['datasets'][arm] or m['bundle_id']!=cell['bundle_id'] or m['observation']['profile']!=spec['observation_profile']
                or m['action_execution']!=modes(arm) or m['source_hash']!=accepted['matched_contract']['source_hash']):
            raise ValueError('fixed accepted recurrent model required')
        config=SimulationConfig(**m['simulation'])
        if config!=SimulationConfig():raise ValueError('unchanged default accounting required')
        phases={}
        for phase,name,key in [('selected','best.zip','recency_selected'),('final','latest.zip','recency_final')]:
            model_path=run/name;protect(model_path,True);model=CLASSES[m['algorithm']].load(model_path,device='cpu')
            model.policy.set_training_mode(False);before=model_record(model);opt=optimizer_fingerprint(model)
            validate_record(before,arm,selected_steps=m['selected_step'] if phase=='selected' else None)
            if before!=m[key]:raise ValueError('restored accepted model mismatch')
            results=evaluate_common_recurrent_training_window(model,view,config,modes(arm)['evaluation_exposure_mode'])
            if (model_record(model)!=before or optimizer_fingerprint(model)!=opt or digest(data['normalizer'])!=normalizer):
                raise ValueError('weights, optimizer or normalization mutated')
            bh=digest({k:r['baselines']['buy_hold'] for k,r in results.items()})
            if baseline is not None and bh!=baseline:raise ValueError('matched fitted holding paths required')
            baseline=bh;directory=a.artifacts/arm/phase;directory.mkdir(parents=True)
            report(results,directory/'training.html','Eligible training-period replay; no fitting or selection')
            groups=group_metrics(results,stocks);flags=spec['benchmark_flags']
            comparable=(all(100*g['excess_return']>=flags['minimum_stock_and_etf_excess_pp'] for g in groups['groups'].values())
                and 100*groups['groups']['stock']['median_excess_return']>=flags['minimum_median_stock_excess_pp'])
            phases[phase]=dict(training=groups,development=cell[phase],training_comparability_flag=comparable,
                model_sha256=checksum(model_path),model_identity=before,optimizer_sha256=opt,
                optimizer=optimizer_record(model),derived_report_sha256=checksum(directory/'training.json'),
                per_symbol={r['symbol']:dict(policy_return=r['metrics']['net_return'],holding_return=r['baselines']['buy_hold']['metrics']['net_return']) for r in results.values()})
            print(arm,phase,'fitted excess pp',{g:round(100*v['excess_return'],3) for g,v in groups['groups'].items()},'comparable',comparable,flush=True)
        records[arm]=phases
    if any(checksum(Path(p))!=h for p,h in protected.items()):raise ValueError('source changed')
    result=dict(protocol_sha256=PROTOCOL_SHA256,records=records,source_sha256=protected,
        training_window={k:spec[k] for k in ('anchor_session','first_execution_session','last_session')},
        episodes_per_model=len(view['episodes']),execution_rows_per_model=sum(len(e['bars'])-1 for e in view['episodes'].values()),
        normalization_sha256=normalizer,common_training_episode_sha256=digest(views['full']['episodes']),holding_path_sha256=baseline,source_files_unchanged=True,
        weights_optimizer_normalization_unchanged=True,training_performed=False,dataset_opened=True,provider_requested=False,colab_allocated=False,
        test_evaluated=False,checkpoint_selected=False,objective_achieved=False,interpretation=spec['interpretation'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
