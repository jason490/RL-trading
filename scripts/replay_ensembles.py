"""Fixed frozen PPO/SAC ensembles on a shared development-only account path."""
import argparse
import csv
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import runpy
from statistics import mean
import subprocess
import torch

from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.provenance import repository_state,source_identity
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import evaluate
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.inference.bundle import FrozenPolicy,checksum
from rl_stocks.inference.ensemble import FixedMember,UniformEnsemble,input_contract
from rl_stocks.reporting.initialization import group_metrics,screen_gate
from rl_stocks.rl.registry.factory import CLASSES


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_bytes(canonical(value));temporary.replace(path)


def prepare(protocol,state,data_root):
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete source study and verified cleanup required')
    if protocol['algorithms']!=['ppo','sac'] or protocol['seeds']!=[11,23,37]:raise ValueError('fixed six-member source panel required')
    if protocol['evaluation_dataset']!='672a21621091a56c1c7beb30557464e7edd8df38874494659bab2c83738e1bf9' or protocol['source_dataset']!='dfd600f62d5f12f7aa1c62b64a3fe2a1d00a48b3264f2a7fff31354958a57059':
        raise ValueError('locked development/source dataset identities required before data access')
    expected={f'{a}-{s}' for a in protocol['algorithms'] for s in protocol['seeds']}
    if protocol['candidates']!={'ppo_mean':[f'ppo-{s}' for s in (11,23,37)],'sac_mean':[f'sac-{s}' for s in (11,23,37)],'combined_mean':[f'{a}-{s}' for a in ('ppo','sac') for s in (11,23,37)]}:
        raise ValueError('fixed uniform candidates required')
    # Only the development-only evaluation dataset is opened. The older source
    # dataset (whose request includes retired dates) is never read.
    dataset_path=data_root/protocol['evaluation_dataset'];d=read_dataset(dataset_path)
    if any(e['split']=='test' for e in d['episodes'].values()):raise ValueError('development-only data required')
    validation={k:e for k,e in d['episodes'].items() if e['split']=='validation'}
    if len(validation)!=26 or any(not '2021'<=b['session'][:4]<='2023' for e in validation.values() for b in e['bars']):raise ValueError('fixed development panel required')
    members={phase:{} for phase in ('selected','final')};references={};protected={dataset_path/'dataset.json':checksum(dataset_path/'dataset.json')};catalog=[];common=None;simulation=None
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one source run per task required')
        path=paths[0];meta=json.loads(path.read_text());identity=f"{meta['algorithm']}-{meta['seed']}"
        if meta['algorithm'] not in protocol['algorithms']:continue
        if (identity not in expected or any(c['identity']==identity for c in catalog) or meta['smoke']
            or meta['status']!='complete' or meta['actual_steps']!=100000 or meta['requested_steps']!=100000
            or meta['dataset']!=protocol['source_dataset']):raise ValueError('unexpected source model')
        if simulation is not None and simulation!=meta['simulation']:raise ValueError('source simulation mismatch')
        simulation=meta['simulation'];bundle=Path(item['root'])/'frozen'/meta['bundle_id']
        frozen=FrozenPolicy(bundle,d['schema'],d['manifest']['provenance']['capabilities']['feed'])
        contract=input_contract(frozen)
        if (contract['names']!=d['names'] or contract['normalizer']!=d['normalizer'] or contract['observation']['profile']!='legacy'
            or contract['adjustments']!=d['manifest']['provenance']['capabilities']['adjustments']):raise ValueError('evaluation input contract differs')
        if common is not None and contract!=common:raise ValueError('member input contracts differ')
        common=contract
        if checksum(path.parent/'best.zip')!=checksum(bundle/'model.zip'):raise ValueError('selected weights differ from frozen bundle')
        for p in bundle.iterdir():protected[p]=checksum(p)
        for phase,weight,report in (('selected','best.zip','selected-validation.json'),('final','latest.zip','validation.json')):
            wp=path.parent/weight;rp=path.parent/report;protected[wp]=checksum(wp);protected[rp]=checksum(rp)
            results=json.loads(rp.read_text())
            if set(results)!=set(validation):raise ValueError('source reference panel differs')
            for key,result in results.items():
                if result['split']!='validation' or [(r['session'],r['price']) for r in result['rows']]!=[(b['session'],b['close']) for b in validation[key]['bars'][1:]]:
                    raise ValueError('source validation date/price bridge failed')
            references[identity,phase]=results
            model=frozen._model if phase=='selected' else CLASSES[meta['algorithm']].load(wp,device='cpu')
            members[phase][identity]=FixedMember(model,contract,identity,checksum(wp))
        protected[path]=checksum(path)
        catalog.append({'identity':identity,'algorithm':meta['algorithm'],'seed':meta['seed'],'run':path.parent.name,
            'bundle_id':meta['bundle_id'],'source_dataset':meta['dataset'],'source_steps':meta['actual_steps'],
            'source_git':meta['git'],'source_hash':meta['source_hash'],
            'phase_weights_sha256':{p:members[p][identity].weights_sha256 for p in ('selected','final')}})
    if {c['identity'] for c in catalog}!=expected:raise ValueError('incomplete source panel')
    return d,members,references,protected,{'passed':True,'catalog':catalog,'input_contract':common,'simulation':simulation,
        'evaluation_dataset':protocol['evaluation_dataset'],'source_dataset_not_opened':protocol['source_dataset'],
        'source_validation_dates_and_closes_exact':True,'normalization_feature_feed_state_contracts_match':True,
        'data_provenance_identical':False,'source_data_identity_not_claimed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,default=Path('configs/experiments/frozen_ensembles.json'))
    p.add_argument('--state',type=Path,default=Path('.runtime/algorithm-stability-state.json'));p.add_argument('--data-root',type=Path,default=Path('data/datasets'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',type=Path,default=Path('results/frozen-ensembles-2026-10-20/preflight.json'));p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());state=json.loads(a.state.read_text());torch.set_num_threads(1)
    if not a.preflight_only:
        source=repository_state(Path.cwd())
        if source['dirty']:raise ValueError('commit protocol and source before replay')
        if subprocess.check_output(['git','show','HEAD:'+a.protocol.resolve().relative_to(Path.cwd().resolve()).as_posix()])!=a.protocol.read_bytes():raise ValueError('committed protocol required')
    if protocol['phases']!=['selected','final'] or protocol['maximum_labeled_replays']!=18:raise ValueError('fixed phase/replay budget required')
    d,members,references,protected,proof=prepare(protocol,state,a.data_root)
    if a.preflight_only:
        a.output.parent.mkdir(parents=True,exist_ok=True);save(a.output,{**proof,'protocol_sha256':digest(protocol),'training_performed':False,'performance_evaluated':False});print('Six members and development bridge verified; no performance evaluated');return
    receipt=json.loads(a.preflight.read_text())
    if receipt['protocol_sha256']!=digest(protocol) or any(receipt.get(k)!=v for k,v in proof.items()):raise ValueError('source members/contracts changed since committed preflight')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'reports').mkdir()
    scope={'status':'running','protocol':protocol,'protocol_sha256':digest(protocol),'source':source,
           'package_source_hash':source_identity(),'preflight':proof,'cells':[],
           'training_performed':False,'colab_allocated':False,'test_evaluated':False,'objective_achieved':False}
    target=a.output/'summary.json';save(target,scope)
    universe=json.loads(Path(protocol['universe']).read_text());stocks={v['symbol'] for v in universe['instruments'] if v['kind']=='stock'}
    helper=runpy.run_path('scripts/replay_directions.py');fingerprint=helper['policy_fingerprint'];difference=helper['reference_differences']
    fingerprints={ (phase,k):fingerprint(m.model) for phase,group in members.items() for k,m in group.items() }
    from rl_stocks.environment.trading.env import TradingEnv
    from rl_stocks.data.generation.datasets import normalizer_from
    ep=next(e for e in d['episodes'].values() if e['split']=='validation')
    probe_env=TradingEnv([ep],normalizer_from(d['normalizer']),d['names']);probe_observation,_=probe_env.reset(seed=11);probe_env.close()
    records=[];baseline=None;cache={}
    normalizer_before=canonical(d['normalizer'])
    try:
        for phase in ('selected','final'):
            cases=[('reference:'+key,[key]) for key in sorted(members[phase])]+list(protocol['candidates'].items())
            for name,identities in cases:
                models=[members[phase][k] for k in identities];ensemble=UniformEnsemble(models)
                cache_key=tuple((k,fingerprints[phase,k]) for k in identities)
                reused=cache_key in cache
                results,oldcell=cache[cache_key] if reused else (evaluate(ensemble,d,config=SimulationConfig(**proof['simulation'])),None)
                report=a.output/'reports'/f'{phase}-{name.replace(":","-")}.json.gz'
                if reused:report=a.output/oldcell['report']
                else:report.write_bytes(gzip.compress(canonical(results),mtime=0))
                cell={'name':name,'phase':phase,'member_ids':identities,'recipe':ensemble.recipe(),
                      'metrics':group_metrics(results,stocks),'declining_stocks':declining_stock_summary(results,stocks),
                      'mean_costs':mean(r['metrics']['costs'] for r in results.values()),
                      'report':str(report.relative_to(a.output)),'report_sha256':checksum(report),'reused':reused}
                expected=references.get((identities[0],phase)) if name.startswith('reference:') else None
                if expected is not None:
                    drift=difference(results,expected);cell['source_reference_drift']=drift
                    if (not drift['panel_matches'] or not drift['row_counts_match'] or drift['session_mismatch_rows']
                        or any(drift[k]>limit for k,limit in protocol['reference_drift_limits'].items())):
                        scope['failed_case']=cell;raise ValueError('source/local reference drift exceeds declared financial limits')
                current=digest({k:r['baselines'] for k,r in results.items()})
                if baseline is not None and baseline!=current:raise ValueError('holding baseline drift')
                baseline=current
                if any(checksum(path)!=h for path,h in protected.items()):raise ValueError('source artifact mutation')
                if any(fingerprint(m.model)!=fingerprints[p,k] for p,g in members.items() for k,m in g.items()):raise ValueError('inference changed model tensors')
                restored=UniformEnsemble.from_recipe(cell['recipe'],members[phase])
                import numpy as np
                np.testing.assert_array_equal(ensemble.predict(probe_observation)[0],restored.predict(probe_observation)[0])
                if canonical(d['normalizer'])!=normalizer_before:raise ValueError('normalization mutation')
                cell['recipe_sha256']=digest(cell['recipe']);cell['recipe_prediction_parity']=True
                scope['cells'].append(cell);cache[cache_key]=(results,cell);save(target,scope)
                for r in results.values():records.append({'name':name,'phase':phase,'symbol':r['symbol'],
                    'kind':'stock' if r['symbol'] in stocks else 'etf',**r['metrics'],
                    'buy_hold_return':r['baselines']['buy_hold']['metrics']['net_return'],**r['directional']})
                print(json.dumps({'phase':phase,'case':name,'return_pct':100*cell['metrics']['score'],'reused':reused}),flush=True)
        assert len(scope['cells'])==protocol['maximum_labeled_replays']
        comparisons={}
        for candidate,ids in protocol['candidates'].items():
            chosen={p:next(c for c in scope['cells'] if c['name']==candidate and c['phase']==p) for p in ('selected','final')}
            reference={}
            for phase in ('selected','final'):
                refs=[next(c for c in scope['cells'] if c['name']=='reference:'+key and c['phase']==phase) for key in ids]
                reference[phase]={'score':mean(c['metrics']['score'] for c in refs),
                    'mean_max_drawdown':mean(c['metrics']['mean_max_drawdown'] for c in refs),
                    'groups':{g:{'net_return':mean(c['metrics']['groups'][g]['net_return'] for c in refs)} for g in ('stock','etf')}}
            pair={'mean_individual':reference,'ensemble':{p:c['metrics'] for p,c in chosen.items()}}
            relative=screen_gate(pair,{'arms':['mean_individual','ensemble'],'reference_arm':'mean_individual','screen_gate':protocol['relative_gate']})
            absolute={}
            limits=protocol['absolute_diagnostic']
            for phase,cell in chosen.items():
                groups=cell['metrics']['groups']
                absolute[phase]=all(100*g['excess_return']>=limits['group_excess_pp'] and
                    100*(g['buy_hold_max_drawdown']-g['max_drawdown'])<=limits['maximum_drawdown_worsening_vs_hold_pp'] for g in groups.values()) and 100*groups['stock']['median_excess_return']>=limits['median_stock_excess_pp']
            comparisons[candidate]={'relative_to_mean_individual_reference':relative,'absolute_comparable_by_phase':absolute,'absolute_comparable':all(absolute.values()),'not_independent_confirmation':True}
        scope['diagnostic_comparisons']=comparisons
        scope.update(status='complete',source_artifacts_and_tensors_unchanged=True,validation_baseline_sha256=baseline,
                     unique_replays=len(cache),ensemble_repetitions=1,interpretation='One fixed ensemble per candidate, not independent ensemble seeds; no prospective evidence')
        save(target,scope)
        with (a.output/'per-symbol.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig=make_subplots(rows=2,cols=1,subplot_titles=('Selected/final mean development return (%)','Final stock/ETF returns (%)'))
        for name in protocol['candidates']:
            cells=[c for c in scope['cells'] if c['name']==name]
            fig.add_trace(go.Bar(x=[c['phase'] for c in cells],y=[100*c['metrics']['score'] for c in cells],name=name),row=1,col=1)
            final=next(c for c in cells if c['phase']=='final')
            fig.add_trace(go.Bar(x=['stock','etf'],y=[100*final['metrics']['groups'][g]['net_return'] for g in ('stock','etf')],name=name,showlegend=False),row=2,col=1)
        holding=next(iter(scope['cells']))['metrics']['groups'];overall=(22*holding['stock']['buy_hold_return']+4*holding['etf']['buy_hold_return'])/26
        fig.add_hline(y=100*overall,row=1,col=1,line_dash='dash',annotation_text='Matched holding')
        fig.update_layout(height=900,barmode='group',title='Fixed uniform seed ensembles: shared account, development only')
        fig.write_html(str(a.output/'comparison.html'),include_plotlyjs=True)
    except BaseException as error:
        scope.update(status='failed',error_type=type(error).__name__,error=str(error));save(target,scope);raise


if __name__=='__main__':main()
