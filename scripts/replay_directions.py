"""Replay universally restricted fixed policies; no fitting or model selection."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from statistics import mean
import csv
import numpy as np
import torch

from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.provenance import source_identity,repository_state
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.evaluation.runner import evaluate
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.inference.bundle import FrozenPolicy,checksum
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.rl.registry.factory import CLASSES


def policy_fingerprint(model):
    """Only used within one run with identical architecture/configuration/spaces."""
    h=hashlib.sha256()
    h.update(type(model.policy).__qualname__.encode())
    for name,tensor in sorted(model.policy.state_dict().items()):
        array=tensor.detach().cpu().numpy()
        h.update(canonical([name,str(array.dtype),list(array.shape)]));h.update(array.tobytes())
    for space in (model.observation_space,model.action_space):
        h.update(canonical([list(space.shape),str(space.dtype)]))
        h.update(space.low.tobytes());h.update(space.high.tobytes())
    return h.hexdigest()


def reference_differences(actual,expected):
    result={'panel_matches':set(actual)==set(expected),'row_counts_match':True,
            'session_mismatch_rows':0,'quantity_mismatch_rows':0,'position_sign_mismatch_rows':0,
            'maximum_quantity_difference':0,'maximum_equity_difference_dollars':0.,
            'maximum_action_difference':0.,'maximum_return_difference':0.,
            'maximum_drawdown_difference':0.,'maximum_cost_difference_dollars':0.,'quantity_examples':[]}
    for key in sorted(set(actual)&set(expected)):
        new,old=actual[key],expected[key]
        result['row_counts_match'] &= len(new['rows'])==len(old['rows'])
        for metric,out in (('net_return','maximum_return_difference'),('max_drawdown','maximum_drawdown_difference'),('costs','maximum_cost_difference_dollars')):
            result[out]=max(result[out],abs(new['metrics'][metric]-old['metrics'][metric]))
        for a,b in zip(new['rows'],old['rows']):
            result['session_mismatch_rows']+=a['session']!=b['session']
            difference=abs(a['quantity']-b['quantity'])
            result['quantity_mismatch_rows']+=difference!=0
            result['position_sign_mismatch_rows']+=int(np.sign(a['quantity'])!=np.sign(b['quantity']))
            result['maximum_quantity_difference']=max(result['maximum_quantity_difference'],difference)
            result['maximum_equity_difference_dollars']=max(result['maximum_equity_difference_dollars'],abs(a['equity']-b['equity']))
            result['maximum_action_difference']=max(result['maximum_action_difference'],max(abs(x-y) for x,y in zip(a['raw_action'],b['raw_action'])))
            if difference and len(result['quantity_examples'])<10:
                result['quantity_examples'].append({'episode':key,'session':a['session'],'actual':a['quantity'],'reference':b['quantity'],'equity_difference':a['equity']-b['equity']})
    return result


def verify_reference(actual,expected,tolerance):
    if set(actual)!=set(expected):raise ValueError('reference episode panel mismatch')
    exact=digest(actual)==digest(expected)
    for key,result in actual.items():
        reference=expected[key]
        if (result['symbol'],result['split'])!=(reference['symbol'],'validation'):
            raise ValueError('reference symbol/split mismatch')
        if len(result['rows'])!=len(reference['rows']):raise ValueError('reference path length mismatch')
        for row,old in zip(result['rows'],reference['rows']):
            if (row['session'],row['quantity'])!=(old['session'],old['quantity']):
                raise ValueError('reference session/quantity path mismatch')
            if abs(row['equity']-old['equity'])>tolerance['equity_dollars']:
                raise ValueError('reference equity path mismatch')
            np.testing.assert_allclose(row['raw_action'],old['raw_action'],rtol=0,atol=tolerance['action_absolute'])
        for name in ('net_return','max_drawdown'):
            if abs(result['metrics'][name]-reference['metrics'][name])>tolerance['return_absolute']:
                raise ValueError('reference metric mismatch')
        if abs(result['metrics']['costs']-reference['metrics']['costs'])>tolerance['cost_dollars']:
            raise ValueError('reference cost mismatch')
        if digest(result['baselines'])!=digest(reference['baselines']):raise ValueError('reference holding mismatch')
    return {'passed':True,'exact_report_digest':exact,'tolerance':tolerance}


def action_statistics(results):
    rows=[row for result in results.values() for row in result['rows']]
    requested=[row.get('policy_action',row['raw_action'])[0] for row in rows]
    return {'weighting':'session-weighted across all validation episodes','sessions':len(rows),
            'requested_long_fraction':mean(a>0 for a in requested),
            'requested_short_fraction':mean(a<0 for a in requested),
            'blocked_signal_fraction':mean(a!=row['raw_action'][0] for a,row in zip(requested,rows)),
            'end_long_fraction':mean(row['quantity']>0 for row in rows),
            'end_short_fraction':mean(row['quantity']<0 for row in rows),
            'mean_absolute_end_exposure':mean(abs(row['exposure']) for row in rows)}


def render(records,cells,path):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig=make_subplots(rows=3,cols=1,subplot_titles=('Final model mean net return by group (%)',
        'Final model mean maximum drawdown (%)','Final per-stock return (%)'))
    for cell in cells:
        if cell['phase']!='final':continue
        name=cell['profile']+'/'+cell['mode']
        fig.add_trace(go.Bar(x=['stock','etf'],y=[100*cell['metrics']['groups'][g]['net_return'] for g in ('stock','etf')],name=name,legendgroup=name),row=1,col=1)
        fig.add_trace(go.Bar(x=['stock','etf'],y=[100*cell['metrics']['groups'][g]['max_drawdown'] for g in ('stock','etf')],name=name,legendgroup=name,showlegend=False),row=2,col=1)
        rows=[r for r in records if (r['profile'],r['mode'],r['phase'],r['kind'])==(cell['profile'],cell['mode'],'final','stock')]
        fig.add_trace(go.Scatter(x=[r['symbol'] for r in rows],y=[100*r['net_return'] for r in rows],name=name,legendgroup=name,showlegend=False,mode='lines+markers'),row=3,col=1)
    fig.update_layout(height=1250,title='Universal exposure restrictions: fixed-policy development replay; no training',barmode='group')
    fig.write_html(str(path),include_plotlyjs=True)


def validate_protocol(protocol):
    profiles=protocol['profiles']
    if (not profiles or profiles!=[p for p in ('legacy','execution_masked','execution_full') if p in profiles]
        or protocol['modes']!=['unrestricted','long_only','short_only']
        or protocol['phases']!=['selected','final'] or protocol['maximum_cells']!=6*len(profiles)
        or type(protocol['source_seed']) is not int or protocol['source_seed'] not in (11,23,37)
        or protocol.get('source_initialization','wide')!='wide'):
        raise ValueError('unsupported fixed replay protocol')


def replay(protocol_path,state_path,output,data_root,reference_only=False):
    protocol_path,state_path,output,data_root=map(Path,(protocol_path,state_path,output,data_root))
    protocol=json.loads(protocol_path.read_text());root=Path(__file__).resolve().parents[1]
    revision=repository_state(root)
    if revision['dirty']:raise ValueError('commit source and protocol before replay')
    committed=subprocess.check_output(['git','show','HEAD:'+protocol_path.resolve().relative_to(root).as_posix()],cwd=root)
    if committed!=protocol_path.read_bytes():raise ValueError('committed protocol required')
    validate_protocol(protocol)
    state=json.loads(state_path.read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('completed source study and verified cleanup required')
    dataset_path=data_root/protocol['dataset'];dataset=read_dataset(dataset_path)
    if any(e['split']=='test' for e in dataset['episodes'].values()):raise ValueError('development-only dataset required')
    validation={k:e for k,e in dataset['episodes'].items() if e['split']=='validation'}
    if len(validation)!=26 or any(not '2021'<=b['session'][:4]<='2023' for e in validation.values() for b in e['bars']):
        raise ValueError('original 26-symbol development panel required')
    universe=json.loads((root/protocol['universe']).read_text());stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    runs={};contract=None;protected={dataset_path/'dataset.json':checksum(dataset_path/'dataset.json')}
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one source run per profile required')
        path=paths[0];meta=json.loads(path.read_text())
        if meta['seed']!=protocol['source_seed'] or meta.get('initialization',{}).get('arm')!=protocol.get('source_initialization','wide'):continue
        observed=meta.get('observation',observation_spec())
        profile=observed.get('profile')
        if (profile not in protocol['profiles'] or profile in runs or observed!=observation_spec(profile)
            or meta['status']!='complete' or meta['smoke'] or meta['seed']!=protocol['source_seed'] or meta['algorithm']!='ppo'
            or meta['actual_steps']!=50000 or meta['requested_steps']!=50000 or meta['dataset']!=protocol['dataset']
            or meta['git']!={'revision':protocol['training_source_revision'],'dirty':''}):
            raise ValueError('unexpected fixed source run')
        current={k:meta[k] for k in ('dataset','simulation','source_hash','versions','resolved_parameters','initialization')}
        current['start_sampling']=meta.get('start_sampling',{'mode':'legacy_uniform'})
        if current['start_sampling']!={'mode':'legacy_uniform'}:raise ValueError('legacy start sampling required')
        if contract is not None and current!=contract:raise ValueError('source run configuration mismatch')
        contract=current;bundle=Path(item['root'])/'frozen'/meta['bundle_id']
        frozen=FrozenPolicy(bundle,dataset['schema'],dataset['manifest']['provenance']['capabilities']['feed'])
        if (frozen.manifest['normalizer']!=dataset['normalizer'] or frozen.manifest['data_manifest']!=dataset['manifest']
            or frozen.manifest['names']!=dataset['names'] or frozen.manifest['metadata'].get('observation',observation_spec())!=observed):
            raise ValueError('frozen data/profile mismatch')
        if checksum(path.parent/'best.zip')!=checksum(bundle/'model.zip'):raise ValueError('selected weights mismatch')
        for p in bundle.iterdir():protected[p]=checksum(p)
        for name in ('metadata.json','best.zip','latest.zip','selected-validation.json','validation.json'):
            p=path.parent/name;protected[p]=checksum(p)
        runs[profile]=(path,meta,frozen)
    if set(runs)!=set(protocol['profiles']):raise ValueError('complete fixed source panel required')
    output.mkdir(parents=True,exist_ok=False);(output/'reports').mkdir()
    source={'revision':revision,'source_hash':source_identity()}
    started={'status':'running','protocol_sha256':digest(protocol),'source':source,'source_run_contract':contract,
             'reference_only':reference_only,'source_seed':protocol['source_seed'],'training_performed':False,'test_evaluated':False,'colab_allocated':False,'cells':[]}
    target=output/'summary.json';target.write_text(json.dumps(started,indent=2)+'\n')
    cells,records,baseline_hash=[],[],None;torch.set_num_threads(1)
    try:
        for profile in protocol['profiles']:
            path,meta,frozen=runs[profile];cache={};normalizer_before=canonical(asdict(frozen.normalizer))
            for phase,weights,reference in (('selected','best.zip','selected-validation.json'),('final','latest.zip','validation.json')):
                model=frozen._model if phase=='selected' else CLASSES['ppo'].load(path.parent/weights,device='cpu')
                model.policy.set_training_mode(False);fingerprint=policy_fingerprint(model)
                for mode in (['unrestricted'] if reference_only else protocol['modes']):
                    key=(fingerprint,mode)
                    if key in cache:
                        previous,results=cache[key]
                        cell={**previous,'phase':phase,'weights_sha256':checksum(path.parent/weights),
                              'reused_evaluation_from':previous['phase']}
                        if mode=='unrestricted':
                            expected=json.loads((path.parent/reference).read_text())
                            try:cell['unrestricted_parity']={**verify_reference(results,expected,protocol['parity_tolerance']),'differences':reference_differences(results,expected)}
                            except (ValueError,AssertionError) as error:
                                if not reference_only:raise
                                cell['unrestricted_parity']={'passed':False,'differences':reference_differences(results,expected),'error':str(error)}
                    else:
                        results=evaluate(model,dataset,config=SimulationConfig(**meta['simulation']),
                                         observation_profile=profile,exposure_mode=mode)
                        panel_hash=digest({k:r['baselines'] for k,r in results.items()})
                        if baseline_hash is not None and baseline_hash!=panel_hash:raise ValueError('baseline drift')
                        baseline_hash=panel_hash
                        report=output/'reports'/f'{profile}-{phase}-{mode}.json.gz'
                        report.write_bytes(gzip.compress(canonical(results),mtime=0))
                        parity=None
                        if mode=='unrestricted':
                            expected=json.loads((path.parent/reference).read_text())
                            differences=reference_differences(results,expected)
                            try:parity={**verify_reference(results,expected,protocol['parity_tolerance']),'differences':differences}
                            except (ValueError,AssertionError) as error:
                                started['failed_case']={'profile':profile,'phase':phase,'mode':mode,
                                    'report':str(report.relative_to(output)),'report_sha256':checksum(report),
                                    'reference_differences':differences,'error':str(error)}
                                if not reference_only:raise
                                parity={'passed':False,'differences':differences,'error':str(error)}
                        cell={'profile':profile,'seed':meta['seed'],'phase':phase,'mode':mode,'run':path.parent.name,
                              'weights_sha256':checksum(path.parent/weights),'policy_fingerprint':fingerprint,
                              'metrics':group_metrics(results,stocks),'declining_stocks':declining_stock_summary(results,stocks),
                              'action_statistics':action_statistics(results),
                              'mean_costs':mean(r['metrics']['costs'] for r in results.values()),
                              'mean_turnover':mean(r['metrics']['turnover'] for r in results.values()),
                              'unrestricted_parity':parity,'report':str(report.relative_to(output)),
                              'report_sha256':checksum(report),'reused_evaluation_from':None}
                        cache[key]=(cell,results)
                    cells.append(cell)
                    for r in results.values():
                        records.append({'profile':profile,'phase':phase,'mode':mode,'symbol':r['symbol'],
                            'kind':'stock' if r['symbol'] in stocks else 'etf',**r['metrics'],
                            'buy_hold_return':r['baselines']['buy_hold']['metrics']['net_return'],**r['directional']})
                    if normalizer_before!=canonical(asdict(frozen.normalizer)):raise ValueError('normalizer mutation')
                    if policy_fingerprint(model)!=fingerprint:raise ValueError('inference mutated policy tensors')
                    if any(checksum(p)!=h for p,h in protected.items()):raise ValueError('source artifact mutation')
                    started['cells']=cells;target.write_text(json.dumps(started,indent=2,allow_nan=False)+'\n')
                    print(json.dumps({'profile':profile,'phase':phase,'mode':mode,'return_pct':100*cell['metrics']['score'],'reused':cell['reused_evaluation_from']}),flush=True)
        assert len(cells)==(2*len(protocol['profiles']) if reference_only else protocol['maximum_cells'])
        started.update(status='calibrated' if reference_only else 'complete',validation_baseline_sha256=baseline_hash,source_artifacts_unchanged=True,
                       unique_replays=sum(c['reused_evaluation_from'] is None for c in cells),objective_achieved=False)
        target.write_text(json.dumps(started,indent=2,allow_nan=False)+'\n')
        with (output/'per-symbol.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
        render(records,cells,output/'comparison.html')
        return target
    except BaseException as error:
        started.update(status='failed',error_type=type(error).__name__);target.write_text(json.dumps(started,indent=2,allow_nan=False)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/experiments/direction_replay.json'))
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('data/datasets'))
    p.add_argument('--reference-only',action='store_true',help='Measure unrestricted reference drift only; does not complete or relax the protocol')
    a=p.parse_args();print(replay(a.protocol,a.state,a.output,a.data_root,a.reference_only))
