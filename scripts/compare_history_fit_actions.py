"""Compare accepted history-policy training/development paths without fitting."""
import argparse,json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.action_execution import diagnose_groups


def compare(output):
    if output.exists():raise ValueError('new output required')
    fit_path=Path('results/history-fit-2026-10-23/summary.json')
    dev_path=Path('results/training-history-2026-10-23/summary.json')
    state_path=Path('.runtime/recency-screen-state.json')
    universe_path=Path('configs/universes/sector_research.json')
    protected={str(p):checksum(p) for p in [fit_path,dev_path,state_path,universe_path,Path(__file__)]}
    fit=json.loads(fit_path.read_text());dev=json.loads(dev_path.read_text());state=json.loads(state_path.read_text())
    if (fit['protocol_sha256']!='e319c26f6cddecfaf3b2b3e847941c77c2bdf0e0ce8d7f8e466353f6b14a3143'
        or not fit['source_files_unchanged'] or not fit['weights_optimizer_normalization_unchanged']
        or fit['training_performed'] or fit['checkpoint_selected'] or fit['test_evaluated']
        or state['status']!='complete' or not state['cleanup_verified'] or dev['test_evaluated']
        or set(fit['records'])!=set(dev['cells']) or set(dev['cells'])!={'full','recent'}):
        raise ValueError('exact accepted cleaned inference-only sources required')
    universe=json.loads(universe_path.read_text())
    stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    panel={i['symbol'] for i in universe['instruments'] if i['role']=='tradable'}
    records=[]
    for arm in ['full','recent']:
        cell=dev['cells'][arm];item=state['completed']['recency-'+arm+'-11']
        run=Path(item['root'])/'runs'/cell['run']
        for phase in ['selected','final']:
            fc=fit['records'][arm][phase];model=run/('best.zip' if phase=='selected' else 'latest.zip')
            if checksum(model)!=fc['model_sha256'] or fc['development']!=cell[phase]:
                raise ValueError('same accepted models and development metrics required')
            protected[str(model)]=checksum(model)
            paths={'training':(Path('artifacts/history-fit-2026-10-23')/arm/phase/'training.json',fc['derived_report_sha256']),
                   'development':(run/cell['reports'][phase]['file'],cell['reports'][phase]['sha256'])}
            for period,(path,expected) in paths.items():
                protected[str(path)]=checksum(path)
                if protected[str(path)]!=expected:raise ValueError('accepted report hash mismatch')
                values=json.loads(path.read_text());split='train' if period=='training' else 'validation'
                if {v['symbol'] for v in values.values()}!=panel or any(v['split']!=split for v in values.values()):
                    raise ValueError('exact panel and chronological source split required')
                records.append(dict(arm=arm,phase=phase,period=period,source_report_sha256=expected,
                    groups=diagnose_groups(values,stocks,period='training-window' if period=='training' else 'development')['groups']))
    if any(checksum(Path(p))!=h for p,h in protected.items()):raise ValueError('source changed')
    result=dict(records=records,source_files_sha256=protected,source_files_unchanged=True,
        training_performed=False,dataset_opened=False,provider_requested=False,colab_allocated=False,
        test_evaluated=False,checkpoint_selected=False,objective_achieved=False,
        interpretation='Observed saved paths; periods/regimes differ. No causal action effect, fitted-bar visitation claim or model promotion.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    compare(a.output);print('Compared8 accepted paths,208 symbol episodes; no training/data access/Colab.')
