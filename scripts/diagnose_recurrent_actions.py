"""Audit exact accepted recurrent-study saved reports, without fitting or data access."""
import argparse,json
from pathlib import Path
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.action_execution import diagnose_groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=('recurrent-state','alignment'),default='recurrent-state')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result_study='recurrent-state' if a.study=='recurrent-state' else 'recurrent-alignment'
    if a.output.exists():raise ValueError('new output required')
    state=json.loads(Path(f'.runtime/{a.study}-screen-state.json').read_text())
    accepted=json.loads(Path(f'results/{result_study}-2026-10-21/summary.json').read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete verified shutdown required')
    universe=json.loads(Path('configs/universes/sector_research.json').read_text())
    stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    panel={i['symbol'] for i in universe['instruments'] if i['role']=='tradable'}
    records=[];protected={};seen=set()
    for item in state['completed'].values():
        path=next((Path(item['root'])/'runs').glob('*/metadata.json'));m=json.loads(path.read_text());protected[str(path)]=checksum(path)
        matches=[arm for arm,c in accepted['cells'].items() if c['run']==path.parent.name]
        if len(matches)!=1 or matches[0] in seen:raise ValueError('exact accepted study runs required')
        arm=matches[0];seen.add(arm)
        for phase in ('selected','final'):
            source=accepted['cells'][arm]['reports'][phase];rp=path.parent/source['file'];protected[str(rp)]=checksum(rp)
            if checksum(rp)!=source['sha256']:raise ValueError('accepted report hash mismatch')
            results=json.loads(rp.read_text())
            if {r['symbol'] for r in results.values()}!=panel:raise ValueError('exact original panel required')
            records.append(dict(arm=arm,phase=phase,source_report_sha256=checksum(rp),**diagnose_groups(results,stocks)))
    if seen!=set(accepted['cells']) or len(records)!=6:raise ValueError('all six phases required')
    if any(checksum(Path(p))!=h for p,h in protected.items()):raise ValueError('source changed')
    result=dict(study=a.study,records=records,source_files_unchanged=True,training_performed=False,dataset_opened=False,
        colab_allocated=False,test_evaluated=False,objective_achieved=False,
        source_sha256={p:checksum(Path(p)) for p in ['scripts/diagnose_recurrent_actions.py','src/rl_stocks/reporting/action_execution.py']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Audited six accepted saved phases; no fitting/data access/Colab.')


if __name__=='__main__':main()
