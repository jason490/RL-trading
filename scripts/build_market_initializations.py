"""Pin mapped24-input untrained policies and original sampling RNG; no dataset access."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from gymnasium import Env,spaces
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.market_initialization import preregistration,parameters,map_initial_tensors,validate_mapping,PREREGISTRATION_SHA256
from rl_stocks.training.shared_initialization import write_initialization,read_initialization


class ModelSpaces(Env):
    observation_space=spaces.Box(-np.inf,np.inf,(24,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(4,),dtype=np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new initialization directory required')
    spec=preregistration();source=Path(spec['shared_initialization']['source_manifest'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(source)!=spec['shared_initialization']['source_manifest_sha256']:raise ValueError('pinned original manifest required')
    old=json.loads(source.read_text());torch.set_num_threads(1);records={};proofs={};protected={str(source):sha(source)}
    a.output.mkdir(parents=True)
    for seed in [11,23,37]:
        control=old['artifacts']['control-screen-'+str(seed)];path=Path(control['file']);protected[str(path)]=sha(path)
        original,rng=read_initialization(path.read_bytes(),control)
        model=create('recurrent_ppo',ModelSpaces(),seed=seed,device='cpu',overrides=parameters())
        proof=map_initial_tensors(model,original)
        torch.set_rng_state(torch.from_numpy(rng));target=a.output/f'market24-seed{seed}.npz'
        record=dict(file=str(target),seed=seed,smoke=False,**write_initialization(model,target))
        restored,restored_rng=read_initialization(target.read_bytes(),record)
        validate_mapping(original,restored,proof['new_columns_sha256'])
        if not np.array_equal(rng,restored_rng):raise ValueError('original CPU sampling RNG must remain exact')
        records['legacy-screen-'+str(seed)]=control
        for arm in ['masked','full']:records[arm+'-screen-'+str(seed)]=record
        proofs[str(seed)]=proof
        if seed==11:
            for arm in ['legacy','masked','full']:records[arm+'-smoke-11']=dict(records[arm+'-screen-11'],smoke=True)
    if any(sha(Path(p))!=h for p,h in protected.items()):raise ValueError('original initialization changed')
    code=['scripts/build_market_initializations.py','src/rl_stocks/training/market_initialization.py']
    result=dict(version=1,preregistration_sha256=PREREGISTRATION_SHA256,generated_without_fitting=True,
        no_dataset_or_provider_access=True,artifacts=records,mapping_proofs=proofs,
        original_assets_unchanged=True,original_assets_sha256=protected,implementation_sha256={p:sha(Path(p)) for p in code},
        scope='Same full64 architecture for smoke/screen, common original tensors/account columns exact, new columns native untrained draw, original CPU RNG for all arms. No policy trajectory parity claim across21/24 inputs.')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Generated3 mapped untrained24-input snapshots; common tensors/new columns/RNG verified. No data/fitting.')


if __name__=='__main__':main()
