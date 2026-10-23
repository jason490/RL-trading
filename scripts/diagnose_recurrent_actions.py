"""Audit exact accepted recurrent-study saved reports, without fitting or data access."""
import argparse,importlib,json
from pathlib import Path
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.action_execution import diagnose_groups


STUDIES={'recurrent-state':('recurrent-state','2026-10-21'),
         'alignment':('recurrent-alignment','2026-10-21'),
         'cash':('recurrent-cash','2026-10-22'),
         'guard':('recurrent-guard','2026-10-22'),
         'capacity':('recurrent-capacity','2026-10-22')}


def audit(study):
    if study not in STUDIES:raise ValueError('registered accepted study required')
    result_study,date=STUDIES[study]
    state_path=Path('.runtime/guard-screen-v2-state.json' if study=='guard' else f'.runtime/{study}-screen-state.json')
    accepted_path=Path(f'results/{result_study}-{date}/summary.json')
    state=json.loads(state_path.read_text())
    accepted=json.loads(accepted_path.read_text())
    module={'recurrent-state':'recurrent_state','alignment':'recurrent_alignment','cash':'recurrent_cash','guard':'recurrent_guard','capacity':'recurrent_capacity'}[study]
    required=set(importlib.import_module('rl_stocks.training.'+module).protocol()['arms'])
    if set(accepted.get('cells',{}))!=required:raise ValueError('all registered study arms required')
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete verified shutdown required')
    universe=json.loads(Path('configs/universes/sector_research.json').read_text())
    stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    panel={i['symbol'] for i in universe['instruments'] if i['role']=='tradable'}
    records=[];protected={str(p):checksum(p) for p in (state_path,accepted_path,Path('configs/universes/sector_research.json'))};seen=set()
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('exactly one accepted run per task required')
        path=paths[0];m=json.loads(path.read_text());protected[str(path)]=checksum(path)
        matches=[arm for arm,c in accepted['cells'].items() if c['run']==path.parent.name]
        if len(matches)!=1 or matches[0] in seen:raise ValueError('exact accepted study runs required')
        arm=matches[0];seen.add(arm)
        if (m.get('status')!='complete' or m.get('bundle_id')!=accepted['cells'][arm]['bundle_id']
                or m.get('dataset')!=accepted['cells'][arm]['dataset']):
            raise ValueError('accepted metadata identity mismatch')
        for phase in ('selected','final'):
            source=accepted['cells'][arm]['reports'][phase]
            expected='selected-validation.json' if phase=='selected' else 'validation.json'
            if source['file']!=expected:raise ValueError('accepted phase report filename required')
            rp=path.parent/source['file'];protected[str(rp)]=checksum(rp)
            if checksum(rp)!=source['sha256']:raise ValueError('accepted report hash mismatch')
            results=json.loads(rp.read_text())
            if {r['symbol'] for r in results.values()}!=panel:raise ValueError('exact original panel required')
            records.append(dict(arm=arm,phase=phase,source_report_sha256=checksum(rp),**diagnose_groups(results,stocks)))
    if seen!=set(accepted['cells']) or len(records)!=2*len(accepted['cells']):raise ValueError('all accepted phases required')
    if any(checksum(Path(p))!=h for p,h in protected.items()):raise ValueError('source changed')
    source_root=Path(__file__).resolve().parents[1]
    result=dict(study=study,accepted_summary_sha256=checksum(accepted_path),records=records,source_files_unchanged=True,source_files_sha256=protected,training_performed=False,dataset_opened=False,
        colab_allocated=False,test_evaluated=False,objective_achieved=False,
        source_sha256={p:checksum(source_root/p) for p in ['scripts/diagnose_recurrent_actions.py','src/rl_stocks/reporting/action_execution.py']})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=tuple(STUDIES),default='recurrent-state')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output required')
    result=audit(a.study)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"Audited {len(result['records'])} accepted saved phases; no fitting/data access/Colab.")


if __name__=='__main__':main()
