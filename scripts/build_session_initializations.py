"""Generate new25-field untrained artifacts; reuse pinned21-field legacy starts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gymnasium import Env,spaces
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.recurrent_state import overrides
from rl_stocks.training.shared_initialization import write_initialization


class ModelSpaces(Env):
    observation_space=spaces.Box(-np.inf,np.inf,(25,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(4,),dtype=np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new initialization directory required')
    a.output.mkdir(parents=True);torch.set_num_threads(1)
    old=json.loads(Path('configs/initializations/recurrent_guard_v2/manifest.json').read_text())
    records={}
    for smoke,seed in [(True,11),(False,11),(False,23),(False,37)]:
        key=('smoke' if smoke else 'screen')+'-'+str(seed)
        records['legacy-'+key]=old['artifacts'][key]
        model=create('recurrent_ppo',ModelSpaces(),seed=seed,device='cpu',smoke=smoke,
                     overrides={**overrides('recurrent_masked',smoke),'target_kl':None})
        path=a.output/(key+'.npz');record=write_initialization(model,path)
        records['augmented-'+key]=dict(file=str(path),seed=seed,smoke=smoke,**record)
    (a.output/'manifest.json').write_text(json.dumps(dict(version=1,generated_without_fitting=True,
        no_dataset_or_provider_access=True,legacy_reuses_untrained_pinned_guard_initializations=True,
        artifacts=records),indent=2)+'\n')
    print('Generated four untrained25-field artifacts; reused four pinned21-field starts. No fitting.')


if __name__=='__main__':main()
