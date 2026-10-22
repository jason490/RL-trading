"""Pin one larger untrained recurrent width, with paired CPU sampling RNG."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gymnasium import Env,spaces
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.recurrent_state import overrides
from rl_stocks.training.shared_initialization import write_initialization,read_initialization


class ModelSpaces(Env):
    observation_space=spaces.Box(-np.inf,np.inf,(21,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(4,),dtype=np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new initialization directory required')
    a.output.mkdir(parents=True);torch.set_num_threads(1);records={}
    old=json.loads(Path('configs/initializations/recurrent_guard_v2/manifest.json').read_text())
    for seed in [11,23,37]:
        control=old['artifacts']['screen-'+str(seed)]
        _,rng=read_initialization(Path(control['file']).read_bytes(),control)
        params=overrides('recurrent_masked',False)
        params['policy_kwargs'].update(net_arch=[128,128],lstm_hidden_size=128)
        model=create('recurrent_ppo',ModelSpaces(),seed=seed,device='cpu',overrides=dict(params,target_kl=None))
        # Weights remain the once-generated seed draw. Align only future sampling RNG.
        torch.set_rng_state(torch.from_numpy(rng))
        path=a.output/f'width128-seed{seed}.npz';record=write_initialization(model,path)
        candidate=dict(file=str(path),seed=seed,smoke=False,**record)
        assert candidate['rng_sha256']==control['rng_sha256']
        records[f'control-screen-{seed}']=control;records[f'larger-screen-{seed}']=candidate
        if seed==11:
            records['control-smoke-11']=dict(control,smoke=True)
            records['larger-smoke-11']=dict(candidate,smoke=True)
    manifest=dict(version=1,generated_without_fitting=True,no_dataset_or_provider_access=True,
        rng_alignment='After generating larger weights, restore pinned control CPU RNG; no weights/performance selection',
        smoke_architecture='Same full widths64/128 as screens; only rollout128/batch64/episode128 differ',artifacts=records)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated three larger untrained snapshots; paired sampling RNG, no data or learning.')


if __name__=='__main__':main()
