"""Audit existing development paths; never train or open a dataset/test partition."""
import argparse
import csv
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.participation import diagnose_groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,action='append',required=True)
    p.add_argument('--universe',type=Path,default=Path('configs/universes/sector_research.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithm', choices=('ppo','sac','td3'), default='ppo')
    p.add_argument('--steps', type=int, default=50000)
    a=p.parse_args()
    if a.steps <= 0:raise ValueError('positive declared transition budget required')
    if a.output.exists():raise ValueError('new output required')
    universe=json.loads(a.universe.read_text());stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    expected_symbols={i['symbol'] for i in universe['instruments'] if i['role']=='tradable'}
    records=[];seen=set();baseline=None;protected={}
    for state_path in a.state:
        state=json.loads(state_path.read_text())
        if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('completed study with verified cleanup required')
        for arm,item in state['completed'].items():
            if len(seen)>=5:raise ValueError('at most five source runs required')
            paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
            if len(paths)!=1:raise ValueError('one run per task required')
            path=paths[0];m=json.loads(path.read_text());protected[str(path)]=checksum(path)
            if m['status']!='complete' or m['smoke'] or m['actual_steps']!=a.steps or m['algorithm']!=a.algorithm:raise ValueError('declared completed algorithm/budget required')
            if path.parent.name in seen:raise ValueError('duplicate source run')
            seen.add(path.parent.name)
            for phase,file in [('selected','selected-validation.json'),('final','validation.json')]:
                rp=path.parent/file;protected[str(rp)]=checksum(rp);results=json.loads(rp.read_text())
                if {r['symbol'] for r in results.values()}!=expected_symbols:raise ValueError('configured validation universe required')
                bh=digest({k:{'metrics':r['baselines']['buy_hold']['metrics'],'sessions':[x['session'] for x in r['rows']]} for k,r in results.items()})
                if baseline is not None and bh!=baseline:raise ValueError('matched holding panel required')
                baseline=bh
                records.append({'study':state_path.stem,'arm':arm,'phase':phase,'run':path.parent.name,
                    'dataset':m['dataset'],'seed':m['seed'],'source_revision':m['git']['revision'],
                    'source_hash':m['source_hash'],'reward_config':m['reward_config'],
                    'source_report_sha256':protected[str(rp)],**diagnose_groups(results,stocks)})
    if not records or len(seen)>5:raise ValueError('one to five unique source runs required')
    assert all(checksum(Path(p))==h for p,h in protected.items())
    sources=['scripts/diagnose_participation.py','src/rl_stocks/reporting/participation.py','src/rl_stocks/evaluation/behavior.py']
    result={'diagnostic_source_sha256':{p:checksum(Path(p)) for p in sources},'records':records,'validation_baseline_sha256':baseline,'universe_sha256':checksum(a.universe),
        'source_files_unchanged':True,'split':'2021-2023 development only','training_performed':False,
        'dataset_opened':False,'test_evaluated':False,'colab_allocated':False,'objective_achieved':False,'declared_algorithm':a.algorithm,'declared_steps':a.steps}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[{'study':r['study'],'arm':r['arm'],'phase':r['phase'],'kind':kind,**g['equal_weight_means']} for r in records for kind,g in r['groups'].items()]
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Audited',len(seen),'source runs and',len(records),'saved phases; no training/data access/Colab.')


if __name__=='__main__':main()
