"""Restore accepted cash-study optimizers read-only; never fit or open datasets."""
import argparse,importlib,json
from pathlib import Path
import torch
from rl_stocks.inference.bundle import checksum
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.reporting.optimizer import optimizer_record,training_diagnostics,parameter_drift
from rl_stocks.training.recurrent_cash import protocol,model_record,validate_record


def audit(study="cash"):
    if study not in ("cash","training-history"):raise ValueError("registered accepted optimizer study required")
    module=importlib.import_module("rl_stocks.training."+("recurrent_cash" if study=="cash" else "training_history"))
    protocol,model_record,validate_record=module.protocol,module.model_record,module.validate_record
    prefix="cash" if study=="cash" else "recency"
    folder="recurrent-cash-2026-10-22" if study=="cash" else "training-history-2026-10-23"
    root=Path.cwd().resolve();torch.set_num_threads(1)
    state_path=root/f'.runtime/{prefix}-screen-state.json'
    summary_path=root/f'results/{folder}/summary.json'
    inventory_path=root/f'.runtime/{prefix}-retained-hashes.json'
    release_path=root/f'results/{folder}/release-verification.json'
    inventory=json.loads(inventory_path.read_text());protected={}
    def protect(path,published=False):
        relative=str(path.resolve().relative_to(root));h=checksum(path)
        if published and inventory.get(relative)!=h:raise ValueError('retained accepted source hash mismatch')
        protected[relative]=h
    protect(state_path,published=study=="cash")
    for path in (summary_path,release_path):protect(path,True)
    protect(inventory_path)
    state=json.loads(state_path.read_text());accepted=json.loads(summary_path.read_text());release=json.loads(release_path.read_text())
    if (state['status']!='complete' or not state['cleanup_verified'] or not release['verified']
            or set(accepted['cells'])!=set(protocol()['arms'])):
        raise ValueError('complete accepted study and verified cleanup/publication required')
    if next(a['digest'] for a in release['assets'] if a['name']=='summary.json')!='sha256:'+checksum(summary_path):
        raise ValueError('published summary mismatch')
    cells={}
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one accepted run per task required')
        path=paths[0];protect(path,True);m=json.loads(path.read_text())
        arms=[a for a,c in accepted['cells'].items() if c['run']==path.parent.name]
        if len(arms)!=1 or arms[0] in cells:raise ValueError('exact accepted run required')
        arm=arms[0];cell=accepted['cells'][arm]
        if (m['status']!='complete' or m['smoke'] or m['seed']!=11 or m['bundle_id']!=cell['bundle_id']
                or m['dataset']!=cell['dataset'] or m['source_hash']!=accepted['matched_contract']['source_hash']):
            raise ValueError('accepted run metadata mismatch')
        records={};models={}
        for phase,filename,key in [('selected','best.zip',prefix+'_selected'),('final','latest.zip',prefix+'_final')]:
            model_path=path.parent/filename;protect(model_path,True)
            model=CLASSES[m['algorithm']].load(model_path,device='cpu')
            actual=model_record(model)
            validate_record(actual,arm,selected_steps=m['selected_step'] if phase=='selected' else None)
            if actual!=m[key]:raise ValueError('restored model identity mismatch')
            record=optimizer_record(model)
            expected=record['sb3_nominal_epoch_counter']*(model.n_steps*model.n_envs//model.batch_size)
            if (record['target_kl'] is not None or record['optimizer_state_count']!=record['optimizer_parameter_count']
                    or record['adam_step_min']!=expected or record['adam_step_max']!=expected):
                raise ValueError('actual fixed-study Adam step/coverage mismatch')
            records[phase]=dict(model_sha256=checksum(model_path),**record)
            models[phase]={n:p.detach().cpu().clone() for n,p in model.policy.named_parameters()}
        cells[arm]=dict(run=path.parent.name,metadata_sha256=checksum(path),
            selected_step=m['selected_step'],models=records,selected_final_parameter_drift=parameter_drift(models['selected'],models['final']),training=training_diagnostics(m['validation_checkpoints']))
    if set(cells)!=set(protocol()['arms']):raise ValueError('all four optimizer pairs required')
    for p in ['scripts/diagnose_cash_optimizer.py','src/rl_stocks/reporting/optimizer.py']:protect(root/p)
    if any(checksum(root/p)!=h for p,h in protected.items()):raise ValueError('source changed during read-only audit')
    return dict(study=study,cells=cells,accepted_summary_sha256=checksum(summary_path),source_files_sha256=protected,
        source_files_unchanged=True,training_performed=False,dataset_opened=False,colab_allocated=False,
        test_evaluated=False,objective_achieved=False,
        interpretation='Saved finite optimizer states and sparse last-rollout logs. No policy update, replay, causal optimizer diagnosis or new selection. SB3 nominal epoch counter is not Adam steps and can overcount executed epochs when KL stopping is enabled.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--study',choices=['cash','training-history'],default='cash');a=p.parse_args()
    if a.output.exists():p.error('new output required')
    result=audit(a.study);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Audited',len(result['cells'])*2,'saved models; no fitting/data access/Colab.')
