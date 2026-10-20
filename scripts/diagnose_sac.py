"""Inspect exactly three saved SAC runs; never open datasets or train."""
import argparse,csv,json
from pathlib import Path
from statistics import mean
import torch
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import FrozenPolicy,checksum
from rl_stocks.environment.observations.state import EXECUTION_FIELDS
from rl_stocks.reporting.offpolicy import saved_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new diagnostic output required')
    state=json.loads(Path('.runtime/algorithm-stability-state.json').read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('completed source study required')
    universe=json.loads(Path('configs/universes/sector_research.json').read_text());stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'};symbols={i['symbol'] for i in universe['instruments'] if i['role']=='tradable'}
    records=[];protected={};baseline=contract=None;torch.set_num_threads(1)
    for seed in (11,23,37):
        item=state['completed'][f'sac-{seed}'];root=Path(item['root']);paths=list((root/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one run per source cell required')
        path=paths[0];m=json.loads(path.read_text());protected[str(path)]=checksum(path)
        if m['algorithm']!='sac' or m['seed']!=seed or m['actual_steps']!=100000 or m['smoke'] or m['status']!='complete':raise ValueError('fixed100k SAC sources required')
        current={k:m[k] for k in ('dataset','source_hash','git','simulation','versions','resolved_parameters')}
        if contract is not None and current!=contract:raise ValueError('same source/configuration required')
        contract=current;folder=root/'frozen'/m['bundle_id'];manifest=json.loads((folder/'manifest.json').read_text())
        for f in folder.iterdir():protected[str(f)]=checksum(f)
        frozen=FrozenPolicy(folder,manifest['feature_schema'],manifest['data_manifest']['provenance']['capabilities']['feed'])
        observation=dict(profile=frozen.observation_profile,input_size=frozen.observation_size,model_shape=list(frozen._model.observation_space.shape),omitted_execution_fields=list(EXECUTION_FIELDS) if frozen.observation_profile=='legacy' else [])
        for phase,filename in [('selected','selected-validation.json'),('final','validation.json')]:
            rp=path.parent/filename;protected[str(rp)]=checksum(rp);results=json.loads(rp.read_text())
            if {v['symbol'] for v in results.values()}!=symbols:raise ValueError('configured symbol panel required')
            bh=digest({k:v['baselines']['buy_hold'] for k,v in results.items()})
            if baseline is not None and bh!=baseline:raise ValueError('identical holding paths required')
            baseline=bh;per_symbol=[saved_episode(v) for v in results.values()];groups={}
            for group in ('stock','etf'):
                rows=[r for r in per_symbol if (r['symbol'] in stocks)==(group=='stock')]
                keys=['net_return','buy_hold_return','net_excess_pp','incremental_cost_drag_pp','saved_path_noncost_excess_pp','reported_costs','mean_abs_exposure','long_fraction','short_fraction','protective_reentries_within_one_session','fills']
                groups[group]={k:mean(r[k] for r in rows) for k in keys}
            records.append(dict(seed=seed,phase=phase,run=path.parent.name,bundle_id=m['bundle_id'],observation=observation,groups=groups,per_symbol=per_symbol,report_sha256=protected[str(rp)]))
    assert all(checksum(Path(p))==h for p,h in protected.items())
    result=dict(records=records,source_hashes=protected,source_unchanged=True,matched_source_contract=contract,validation_baseline_sha256=baseline,
        training_performed=False,dataset_opened=False,test_evaluated=False,colab_allocated=False,objective_achieved=False,
        interpretation='Saved path accounting only. Cost add-back is not a zero-cost replay. Close short fractions do not establish short profit; directional P&L is not reconstructed without corporate-action bars. Legacy12-input proof identifies omitted execution state, not its performance effect.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        rows=[dict(seed=r['seed'],phase=r['phase'],group=g,**v) for r in records for g,v in r['groups'].items()];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Audited3 frozen SAC models and6 saved phases; no fitting or dataset access.')


if __name__=='__main__':main()
