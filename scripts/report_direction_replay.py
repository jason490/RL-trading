"""Pair fixed universal-filter diagnostics across the three declared legacy seeds."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import mean,stdev
from rl_stocks.core.contracts import digest
from rl_stocks.core.provenance import repository_state
from rl_stocks.inference.bundle import checksum

PROTOCOLS=('direction_replay.json','direction_replay_seed23.json','direction_replay_seed37_calibrated.json')


def aggregate(summaries,protocols):
    cells={};contract=None;baseline=None;sources=[];cohort_symbols=None
    for summary,protocol in zip(summaries,protocols,strict=True):
        if (summary['status']!='complete' or summary.get('reference_only',False)
            or not summary['source_artifacts_unchanged'] or summary['training_performed']
            or summary['test_evaluated'] or summary['colab_allocated']
            or summary['protocol_sha256']!=digest(protocol)):
            raise ValueError('complete immutable diagnostic matching committed protocol required')
        seed=protocol['source_seed']
        if seed in cells:raise ValueError('duplicate seed')
        current={k:v for k,v in summary['source_run_contract'].items() if k!='source_hash'}
        if contract is not None and current!=contract:raise ValueError('training configuration mismatch')
        contract=current
        if baseline is not None and baseline!=summary['validation_baseline_sha256']:raise ValueError('holding panel mismatch')
        baseline=summary['validation_baseline_sha256'];selected={}
        for cell in summary['cells']:
            if cell['profile']!='legacy':continue
            key=(cell['phase'],cell['mode'])
            if key in selected:raise ValueError('duplicate phase/mode')
            if cell['mode']=='unrestricted' and not cell['unrestricted_parity']['passed']:
                raise ValueError('unverified unrestricted path')
            cohort=cell['declining_stocks']['records'];symbols={r['symbol'] for r in cohort}
            if len(symbols)!=len(cohort) or cohort_symbols is not None and symbols!=cohort_symbols:
                raise ValueError('declining reporting panel mismatch')
            cohort_symbols=symbols
            if any(r['buy_hold_return']>=0 or r['positive_policy_and_short_pnl']!=(r['policy_return']>0 and r['short_net_pnl']>0) for r in cohort):
                raise ValueError('inconsistent declining-stock profit attribution')
            selected[key]=cell
        if set(selected)!={(p,m) for p in ('selected','final') for m in ('unrestricted','long_only','short_only')}:
            raise ValueError('complete mode/phase panel required')
        for phase in ('selected','final'):
            if len({selected[phase,m]['weights_sha256'] for m in ('unrestricted','long_only','short_only')})!=1:
                raise ValueError('filter modes must use identical phase weights')
        cells[seed]=selected;sources.append({'seed':seed,'training_source_revision':protocol['training_source_revision'],
              'replay_source':summary['source'],'protocol_sha256':summary['protocol_sha256']})
    if set(cells)!={11,23,37}:raise ValueError('all three fixed seeds required')
    results=[]
    for phase in ('selected','final'):
        for mode in ('unrestricted','long_only','short_only'):
            chosen=[cells[s][phase,mode] for s in (11,23,37)]
            gains=[100*(c['metrics']['score']-cells[s][phase,'unrestricted']['metrics']['score']) for s,c in zip((11,23,37),chosen)]
            cohort=[r for c in chosen for r in c['declining_stocks']['records']]
            results.append({'phase':phase,'mode':mode,'seeds':[11,23,37],
                'mean_return':mean(c['metrics']['score'] for c in chosen),
                'seed_return_standard_deviation':stdev(c['metrics']['score'] for c in chosen),
                'mean_gain_vs_paired_unrestricted_pp':mean(gains),'positive_gain_seeds':sum(g>0 for g in gains),
                'groups':{g:{name:mean(c['metrics']['groups'][g][name] for c in chosen)
                    for name in ('net_return','buy_hold_return','excess_return','max_drawdown','buy_hold_max_drawdown','median_excess_return')} for g in ('stock','etf')},
                'mean_costs':mean(c['mean_costs'] for c in chosen),
                'declining_stock_seed_observations':len(cohort),
                'positive_total_and_short_count':sum(r['positive_policy_and_short_pnl'] for r in cohort),
                'mean_declining_stock_return':mean(r['policy_return'] for r in cohort),
                'declining_per_symbol':{symbol:{'mean_policy_return':mean(r['policy_return'] for r in cohort if r['symbol']==symbol),
                     'buy_hold_return':next(r['buy_hold_return'] for r in cohort if r['symbol']==symbol),
                     'positive_total_and_short_seeds':sum(r['positive_policy_and_short_pnl'] for r in cohort if r['symbol']==symbol)}
                     for symbol in sorted({r['symbol'] for r in cohort})}})
    return {'scope':'Fixed-policy development diagnostic; no strategy selection or independent confirmation',
        'sources':sources,'matched_training_contract':contract,'validation_baseline_sha256':baseline,'results':results,
        'objective_achieved':False,'training_performed':False,'colab_allocated':False,'test_evaluated':False,
        'seed37_selected_bounded_reference_exception':not cells[37]['selected','unrestricted']['unrestricted_parity']['quantity_path_exact']}


def action_feedback(reference,restricted):
    deltas=[];flips=0
    for key in sorted(set(reference)&set(restricted)):
        a={r['session']:r for r in reference[key]['rows']}
        b={r['session']:r for r in restricted[key]['rows']}
        for day in sorted(set(a)&set(b)):
            original=a[day].get('policy_action',a[day]['raw_action'])[0]
            request=b[day].get('policy_action',b[day]['raw_action'])[0]
            deltas.append(abs(original-request))
            flips+=(original>0 and request<0) or (original<0 and request>0)
    if not deltas:raise ValueError('matching decision sessions required')
    return {'common_decisions':len(deltas),'requested_direction_sign_flips':flips,
            'requested_direction_sign_flip_fraction':flips/len(deltas),
            'mean_absolute_requested_exposure_change':mean(deltas),
            'maximum_absolute_requested_exposure_change':max(deltas),
            'comparison':'Original policy requests before restriction; same model and fixed market inputs, different simulated account paths'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs=3)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocols=[json.loads((Path('configs/experiments')/name).read_text()) for name in PROTOCOLS]
    summaries=[json.loads((root/'summary.json').read_text()) for root in a.roots]
    source=repository_state(Path(__file__).resolve().parents[1])
    if source['dirty']:raise ValueError('commit reporter and inputs before final aggregation')
    result=aggregate(summaries,protocols)
    result['reporter']={'git':source,'sha256':checksum(Path(__file__))}
    bridge=json.loads(Path('results/execution-observations-2026-10-19/legacy-screen-parity.json').read_text())
    if not bridge['passed'] or not bridge['all_checkpoint_scores_equal']:raise ValueError('training source bridge required')
    first=next(c for c in summaries[0]['cells'] if c['profile']=='legacy' and c['phase']=='selected')
    if first['run']!=bridge['new_run']:raise ValueError('source bridge run mismatch')
    result['seed11_exact_training_bridge']=bridge
    failure=json.loads(Path('results/direction-replay-2026-10-19/seed37-initial-failure.json').read_text())
    if failure['status']!='failed' or failure['cells']:raise ValueError('original strict seed37 failure receipt required')
    result['seed37_selected_strict_reference_failed']=True
    records=[];reports={}
    for root,protocol,summary in zip(a.roots,protocols,summaries):
        for cell in summary['cells']:
            path=root/cell['report']
            if checksum(path)!=cell['report_sha256']:raise ValueError('replay report checksum mismatch')
            reports[str(path)]={'sha256':cell['report_sha256'],'size':path.stat().st_size}
        with (root/'per-symbol.csv').open() as f:
            records.extend({'seed':protocol['source_seed'],**r} for r in csv.DictReader(f) if r['profile']=='legacy')
    result['unique_source_reports_verified']=len(reports)
    feedback=[]
    for root,protocol,summary in zip(a.roots,protocols,summaries):
        cache={}
        def load(cell):
            path=root/cell['report']
            if path not in cache:cache[path]=json.loads(gzip.decompress(path.read_bytes()))
            return cache[path]
        for profile in protocol['profiles']:
            for phase in ('selected','final'):
                chosen={c['mode']:c for c in summary['cells'] if c['profile']==profile and c['phase']==phase}
                if len({c['weights_sha256'] for c in chosen.values()})!=1:raise ValueError('feedback model mismatch')
                for mode in ('long_only','short_only'):
                    feedback.append({'seed':protocol['source_seed'],'profile':profile,'phase':phase,'mode':mode,
                        **action_feedback(load(chosen['unrestricted']),load(chosen[mode]))})
    result['account_path_feedback']=feedback
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig=make_subplots(rows=3,cols=1,subplot_titles=('Mean return; error bars are seed standard deviation (%)',
       'Final stock and ETF return (%)','Short-only declining-stock returns by seed (%)'))
    for mode in ('unrestricted','long_only','short_only'):
        group=[r for r in result['results'] if r['mode']==mode]
        fig.add_trace(go.Bar(x=[r['phase'] for r in group],y=[100*r['mean_return'] for r in group],error_y={'array':[100*r['seed_return_standard_deviation'] for r in group]},name=mode),row=1,col=1)
        final=next(r for r in group if r['phase']=='final')
        fig.add_trace(go.Bar(x=['stock','etf'],y=[100*final['groups'][g]['net_return'] for g in ('stock','etf')],name=mode,showlegend=False),row=2,col=1)
    for seed in (11,23,37):
        rows=[r for r in records if r['seed']==seed and r['phase']=='final' and r['mode']=='short_only' and float(r['buy_hold_return'])<0 and r['kind']=='stock']
        fig.add_trace(go.Scatter(x=[r['symbol'] for r in rows],y=[100*float(r['net_return']) for r in rows],mode='lines+markers',name=f'short-only seed {seed}'),row=3,col=1)
    fig.update_layout(height=1150,barmode='group',title='Three-seed universal direction replay; retrospective development evidence')
    fig.write_html(str(a.output.with_suffix('.html')),include_plotlyjs=True)
    print(json.dumps([{'phase':r['phase'],'mode':r['mode'],'return_pct':100*r['mean_return'],'declining_positive':r['positive_total_and_short_count']} for r in result['results']],indent=2))


if __name__=='__main__':main()
