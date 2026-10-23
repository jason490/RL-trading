from copy import deepcopy
import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.market_initialization import parameters,map_initial_tensors,validate_mapping,column_fingerprint
from rl_stocks.training.recurrent_capacity import initialization_entry,ROOT
from rl_stocks.training.shared_initialization import read_initialization,write_initialization,load_initialization,tensor_fingerprint


class ModelSpaces(Env):
    observation_space=spaces.Box(-np.inf,np.inf,(24,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(4,),dtype=np.float32)


@pytest.mark.parametrize('seed',[11,23,37])
def test_all_untrained_seeds_map_common_columns_retain_native_new_columns_and_roundtrip_rng(tmp_path,seed):
    torch.set_num_threads(1);entry=initialization_entry('control',seed,False)
    old,rng=read_initialization((ROOT/entry['file']).read_bytes(),entry)
    model=create('recurrent_ppo',ModelSpaces(),seed=seed,device='cpu',overrides=parameters())
    native={k:v.detach().numpy().copy() for k,v in model.policy.state_dict().items()}
    native_sha=tensor_fingerprint(native);new_sha=column_fingerprint(native)
    before=deepcopy(old);proof=map_initial_tensors(model,old)
    assert proof['new_columns_sha256']==new_sha and proof['native_draw_sha256']==native_sha
    assert all(np.array_equal(before[k],old[k]) for k in old)
    assert not model.policy.optimizer.state and model.num_timesteps==model._n_updates==0
    torch.set_rng_state(torch.from_numpy(rng));path=tmp_path/'init.npz';spec=write_initialization(model,path)
    arrays,restored_rng=read_initialization(path.read_bytes(),spec)
    assert np.array_equal(rng,restored_rng);validate_mapping(old,arrays,new_sha)
    twin=create('recurrent_ppo',ModelSpaces(),seed=999,device='cpu',overrides=parameters(True))
    load_initialization(twin,path,spec)
    assert all(torch.equal(model.policy.state_dict()[k],twin.policy.state_dict()[k]) for k in arrays)
    assert np.array_equal(torch.get_rng_state().numpy(),rng) and not twin.policy.optimizer.state
    for key,column in [('lstm_actor.weight_ih_l0',0),('lstm_actor.weight_ih_l0',7),('lstm_critic.weight_ih_l0',10)]:
        bad={k:a.copy() for k,a in arrays.items()};bad[key][0,column]+=1
        with pytest.raises(ValueError):validate_mapping(old,bad,new_sha)
    bad={k:a.copy() for k,a in arrays.items()};bad['log_std'][0]+=1
    with pytest.raises(ValueError):validate_mapping(old,bad,new_sha)


def test_actual_manifest_assets_and_final_protocol_preserve_fixed_preregistration():
    import json,hashlib
    from rl_stocks.core.contracts import digest
    from rl_stocks.training.market_initialization import preregistration,PREREGISTRATION_SHA256
    original=preregistration();final=json.loads((ROOT/'configs/experiments/market_context.json').read_text())
    assert final['preregistration_sha256']==PREREGISTRATION_SHA256
    for k,v in original.items():
        if k not in ('name','shared_initialization','implementation_status'):assert final[k]==v
    for k,v in original['shared_initialization'].items():assert final['shared_initialization'][k]==v
    path=ROOT/final['shared_initialization']['manifest'];raw=path.read_bytes();manifest=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==final['shared_initialization']['manifest_sha256']
    assert manifest['generated_without_fitting'] and manifest['no_dataset_or_provider_access']
    assert manifest['original_assets_unchanged'] and manifest['preregistration_sha256']==PREREGISTRATION_SHA256
    for path,sha in {**manifest['original_assets_sha256'],**manifest['implementation_sha256']}.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    for seed in [11,23,37]:
        for smoke in ([False,True] if seed==11 else [False]):
            phase='smoke' if smoke else 'screen';key=lambda arm:f'{arm}-{phase}-{seed}'
            entries={arm:manifest['artifacts'][key(arm)] for arm in ['legacy','masked','full']}
            assert entries['masked']==entries['full']
            arrays={};rngs={}
            for arm,e in entries.items():
                arrays[arm],rngs[arm]=read_initialization((ROOT/e['file']).read_bytes(),e)
                assert e['seed']==seed and e['smoke'] is smoke
            assert all(np.array_equal(rngs['legacy'],r) for r in rngs.values())
            validate_mapping(arrays['legacy'],arrays['full'],manifest['mapping_proofs'][str(seed)]['new_columns_sha256'])
    assert final['datasets']['legacy']==original['source_dataset']
    assert final['dataset_contracts']['masked']['extra_normalizer_fit_rows']==0
    assert final['dataset_contracts']['full']['extra_normalizer_fit_rows']==298946
