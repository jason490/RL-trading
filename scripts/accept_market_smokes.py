"""Accept fixed market smokes only after retrieval and verified shutdown."""
from rl_stocks.training.market_context import validate_sampling
import argparse
import json
from pathlib import Path

from rl_stocks.inference.bundle import checksum
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.checkpoint_state import read_checkpoint_state
from rl_stocks.training.diversity import validate_counts
from rl_stocks.training.market_context import protocol, PROTOCOL_SHA256, model_record, validate_record, modes, validate_execution_counts, validate_execution_report,validate_initial_records,identify_arm,initialization_spec,initialization_entry


def accept(state_path, restoration_path, source_proof):
    state=json.loads(state_path.read_text());restoration=json.loads(restoration_path.read_text())
    proof=json.loads(source_proof.read_text());spec=protocol()
    if state['status']!='complete' or not state['cleanup_verified']:
        raise ValueError('complete smokes and verified owned shutdown required')
    if not restoration['passed'] or restoration['training_performed'] or restoration['test_evaluated']:
        raise ValueError('inference-only frozen acceptance required')
    records={}
    for arm in spec['arms']:
        item=state['completed'][f'market-{arm}-11-smoke']
        root=Path(item['root']);paths=list((root/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('exactly one smoke run required')
        path=paths[0];m=json.loads(path.read_text())
        if (m['study_protocol_sha256']!=PROTOCOL_SHA256 or m['git']['revision']!=proof['source_revision']
                or m['source_hash']!=proof['source_hash'] or m['git']['dirty'] or not m['smoke'] or m['seed']!=11):
            raise ValueError('fixed clean smoke source required')
        if identify_arm(m)!=arm:raise ValueError('market smoke task/arm mismatch')
        from rl_stocks.training.shared_initialization import read_initialization
        init_proof=initialization_spec(arm,11,True)
        if m.get('market_shared_initialization')!=init_proof:raise ValueError('loaded shared initialization metadata required')
        read_initialization((path.parent/'market-initialization.npz').read_bytes(),initialization_entry(arm,11,True))
        validate_record(m['market_initialization'],arm,True,initial=True)
        if m.get('action_execution')!=modes(arm):raise ValueError('actual market execution modes mismatch')
        validate_execution_counts(m['training_execution'],arm,512)
        for filename in ('selected-validation.json','validation.json'):
            validate_execution_report(json.loads((path.parent/filename).read_text()),arm)
        for name,key in [('latest.zip','market_final'),('best.zip','market_selected')]:
            restored=model_record(CLASSES[m['algorithm']].load(path.parent/name,device='cpu'))
            if restored['policy_tensor_sha256']==m['market_initialization']['policy_tensor_sha256']:
                raise ValueError('actual training must change model weights')
            validate_record(restored,arm,True,selected_steps=m['selected_step'] if name=='best.zip' else None)
            if restored!=m[key]:raise ValueError('actual saved model differs from metadata')
        checkpoint=read_checkpoint_state(path.parent)
        if not checkpoint['final_post_update'] or checkpoint['latest']['steps']!=512 or checkpoint['selected']['steps']!=m['selected_step']:
            raise ValueError('smoke checkpoint identity mismatch')
        matches=[b for b in restoration['bundles'] if b['bundle_id']==m['bundle_id']]
        if len(matches)!=1:raise ValueError('one restored frozen record required')
        frozen=matches[0]
        if (frozen['predictions_checked']!=78 or len(set(frozen['symbols_checked']))!=26
                or not frozen['exact_prediction_parity'] or not frozen['deterministic_repeat']
                or not frozen['normalization_unchanged'] or not frozen['normalization_and_data_match_dataset']
                or not frozen['schema_feed_nonfinite_rejection']
                or frozen['recurrent_sequence_and_reset_parity']!=True
                or frozen.get('action_execution')!=modes(arm)):
            raise ValueError('complete frozen sequence/reset acceptance required')
        if arm=='masked' and frozen.get('market_mask_rejection') is not True:raise ValueError('actual masked bundle must reject nonzero context')
        records[arm]=dict(session=item['session'],run=path.parent.name,bundle_id=m['bundle_id'],
            initial=m['market_initialization'],sampling=m['sampling_diagnostics'],
            training_execution=m['training_execution'],final_tensor_sha256=m['market_final']['policy_tensor_sha256'],
            sampling_checked=validate_sampling(m['sampling_diagnostics'],arm,True,512),
            checkpoint_identity=checkpoint,metadata_sha256=checksum(path),python=m['python'],actual_saved_models_restored=True,duration_seconds=m['duration_seconds'],
            final_optimizer=m['market_final']['optimizer'],selected_optimizer=m['market_selected']['optimizer'])
        if arm=='masked':records[arm]['market_mask_rejection']=True
    initialization_checks=validate_initial_records({a:r['initial'] for a,r in records.items()},smoke=True)
    if any(r['sampling']!=records['legacy']['sampling'] for r in records.values()):
        raise ValueError('actual training sampling differs')
    if len(restoration['bundles'])!=3:raise ValueError('exactly three frozen smoke records required')
    return dict(passed=True,protocol_sha256=PROTOCOL_SHA256,source_proof=proof,records=records,
                frozen_predictions_exact=234,initialization_checks=initialization_checks,actual_sampling_equal=True,
                cleanup_verified=True,screen_allocated=False,test_evaluated=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','restoration','source-proof','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('output must be new')
    result=accept(a.state,a.restoration,a.source_proof)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Accepted three512 CPU smokes,234 exact frozen predictions and verified shutdown.')
