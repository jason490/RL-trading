"""Fixed training-prefix Gaussian probabilities; no fitting or performance scoring."""
import argparse,json
from pathlib import Path
import torch
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.generation.datasets import read_dataset,normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.rl.registry.factory import create,CLASSES
from rl_stocks.reporting.action_boundaries import recurrent_parameters
from rl_stocks.reporting.optimizer import optimizer_fingerprint
from rl_stocks.training.input_clipping import (overrides,apply_initialization,model_record,
    validate_record,validate_payload,initialization_entry,ROOT)

PROTOCOL_SHA256='f49c54441bec512c6b8170c0d8e437987aba08a503b22eab5cc31559e4036ba4'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output required')
    protocol_path=Path('configs/experiments/action_boundary_diagnostic.json');spec=json.loads(protocol_path.read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed action-boundary protocol changed')
    accepted=Path(spec['accepted_summary'])
    if checksum(accepted)!=spec['accepted_summary_sha256']:raise ValueError('accepted summary changed')
    summary=json.loads(accepted.read_text());cell=summary['cells']['control']
    if cell['run']!=spec['source_run']:raise ValueError('exact accepted control required')
    state_path=Path('.runtime/clipping-screen-state.json');state=json.loads(state_path.read_text())
    if state['status']!='complete' or state.get('cleanup_verified') is not True:raise ValueError('owned shutdown required')
    root=Path(state['completed']['clipping-control-11']['root'])/'runs'/spec['source_run']
    metadata=root/'metadata.json'
    if checksum(metadata)!=spec['metadata_sha256']:raise ValueError('pinned model metadata changed')
    saved_metadata=json.loads(metadata.read_text())
    data_path=Path('data/datasets')/spec['dataset']
    data=validate_payload(read_dataset(data_path),'control');normalizer=normalizer_from(data['normalizer'])
    episodes=[e for e in data['episodes'].values() if e['split']=='train']
    if len(episodes)!=111 or len({e['symbol'] for e in episodes})!=111:raise ValueError('exact training panel required')
    files=[protocol_path,accepted,state_path,metadata,data_path/'dataset.json',ROOT/initialization_entry('control',11,False)['file']]
    for source in spec['models'].values():
        path=root/source['filename']
        if checksum(path)!=source['sha256']:raise ValueError('pinned saved model changed')
        files.append(path)
    protected={str(path):checksum(path) for path in files};normalizer_before=digest(data['normalizer'])
    torch.set_num_threads(1);records={}
    for phase in ['untrained','selected','final']:
        if phase=='untrained':
            env=TradingEnv([episodes[0]],normalizer,data['names'],observation_profile='execution_masked')
            try:model=create('recurrent_ppo',env,seed=11,device='cpu',smoke=False,overrides=overrides('control'))
            finally:env.close()
            apply_initialization(model,'control',11,False)
        else:model=CLASSES['recurrent_ppo'].load(root/spec['models'][phase]['filename'],device='cpu')
        before=model_record(model);fingerprint=optimizer_fingerprint(model);rng=torch.get_rng_state().clone()
        validate_record(before,'control',initial=phase=='untrained',selected_steps=saved_metadata['selected_step'] if phase=='selected' else None)
        result=recurrent_parameters(model,episodes,normalizer,data['names'],spec['selection']['decisions_per_symbol'])
        if before!=model_record(model) or fingerprint!=optimizer_fingerprint(model) or not torch.equal(rng,torch.get_rng_state()):
            raise ValueError('diagnostic changed model/Adam/RNG')
        records[phase]=dict(**result,model_record_unchanged=True,optimizer_unchanged=True,rng_unchanged=True,
            policy_tensor_sha256=before['policy_tensor_sha256'],optimizer_fingerprint=fingerprint)
    if digest(data['normalizer'])!=normalizer_before or any(checksum(Path(p))!=h for p,h in protected.items()):
        raise ValueError('source or normalization changed')
    result=dict(protocol_sha256=PROTOCOL_SHA256,records=records,protected_sha256=protected,
        source_files_unchanged=True,normalization_unchanged=True,training_performed=False,optimizer_steps=False,
        policy_actions_executed=False,validation_performance_evaluated=False,test_evaluated=False,colab_allocated=False,
        objective_achieved=False,interpretation=spec['constraints']['interpretation'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Audited3 Gaussian models on111 fixed training prefixes; no fitting or return scoring.')


if __name__=='__main__':main()
