"""Real dataset environment/model construction acceptance; no learning or allocation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check
from rl_stocks.data.generation.datasets import read_dataset,normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.input_clipping import (protocol,PROTOCOL_SHA256,validate_payload,
    overrides,apply_initialization,model_record,validate_record,validate_initial_records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output required')
    torch.set_num_threads(1);spec=protocol();records={};sizes={}
    for arm in spec['arms']:
        data=validate_payload(read_dataset(Path('data/datasets')/spec['datasets'][arm]),arm)
        episodes=[e for e in data['episodes'].values() if e['split']=='train']
        env=TradingEnv(episodes,normalizer_from(data['normalizer']),data['names'],SimulationConfig(),
            random_start=True,episode_steps=128,start_sampling=spec['start_sampling'],observation_profile='execution_masked')
        try:
            gym_check(env,skip_render_check=True);sb3_check(env)
            model=create('recurrent_ppo',env,seed=11,device='cpu',smoke=True,overrides=overrides(arm,True))
            apply_initialization(model,arm,11,True);record=model_record(model)
            validate_record(record,arm,True,initial=True)
            obs,_=env.reset(seed=11)
            action,state=model.predict(obs,state=None,episode_start=np.array([True]),deterministic=True)
            if action.shape!=(4,) or not np.isfinite(action).all() or any(not np.isfinite(v).all() for v in state):
                raise ValueError('finite causal recurrent prediction required')
            if model_record(model)!=record:raise ValueError('prediction changed untrained model or optimizer')
            records[arm]=record;sizes[arm]=list(env.observation_space.shape)
        finally:env.close()
    validate_initial_records(records,True)
    result=dict(passed=True,protocol_sha256=PROTOCOL_SHA256,observation_sizes=sizes,initial_records=records,
        gymnasium_and_sb3_checkers_passed=True,actual_recurrent_forward_passed=True,pinned_per_arm_initialization=True,sampling_rng_identical=True,cross_arm_initial_tensors_identical=True,
        training_performed=False,test_evaluated=False,sessions_allocated=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Both actual full-width dataset environments/checkers and untrained recurrent forwards pass; no learning.')


if __name__=='__main__':main()
