"""Read owned saved SAC models/replay; no fitting, data fetch or test access."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.sac_learning import moments,target_parts
from rl_stocks.training.sac_state import PROTOCOL_SHA256


def weights(model):
    h=hashlib.sha256()
    for key,value in sorted(model.policy.state_dict().items()):h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    h.update(model.log_ent_coef.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output required')
    state=json.loads(Path('.runtime/sac-state-screen-state.json').read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete owned study required')
    records=[];protected={};torch.set_num_threads(1)
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one owned run per source required')
        path=paths[0];m=json.loads(path.read_text());run=path.parent
        if (m['algorithm']!='sac' or m['seed']!=11 or m['smoke'] or m['actual_steps']!=50000
            or m['study_protocol_sha256']!=PROTOCOL_SHA256 or m['reward_config']!={'kind':'net_log_equity','scale':1.}):raise ValueError('fixed source contract required')
        for f in [path,run/'latest.zip',run/'replay.pkl']:protected[str(f)]=checksum(f)
        model=SAC.load(run/'latest.zip',device='cpu')
        # Only this task's own locally retained replay is deserialized, never downloads.
        model.load_replay_buffer(run/'replay.pkl');buffer=model.replay_buffer
        if type(buffer).__name__!='ReplayBuffer' or buffer.n_envs!=1 or buffer.full or buffer.pos!=50000 or buffer.optimize_memory_usage or model._vec_normalize_env is not None:
            raise ValueError('fixed complete one-step training replay required')
        model.policy.set_training_mode(False);model.critic_target.eval();before=weights(model);updates=model._n_updates
        rewards=buffer.rewards[:buffer.pos,0];alpha=float(model.log_ent_coef.detach().exp().cpu().item())
        indices=np.random.default_rng(11).choice(buffer.pos,size=4096,replace=False)
        tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
        obs=tensor(buffer.observations[indices,0]);nxt=tensor(buffer.next_observations[indices,0]);actions=tensor(buffer.actions[indices,0])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            with torch.no_grad():
                next_actions,logp=model.actor.action_log_prob(nxt)
                qnext=torch.cat(model.critic_target(nxt,next_actions),dim=1).min(dim=1).values
                qcurrent=torch.cat(model.critic(obs,actions),dim=1)
                parts=target_parts(rewards[indices],qnext.numpy(),logp.numpy(),buffer.dones[indices,0],buffer.timeouts[indices,0],alpha,model.gamma)
                current=qcurrent.numpy();residual=current-parts['target'][:,None]
        if weights(model)!=before or model._n_updates!=updates:raise ValueError('diagnostic mutated weights or updates')
        records.append(dict(arm=m['observation']['profile'],run=run.name,replay_rows=buffer.pos,observation_size=model.observation_space.shape[0],alpha=alpha,
            all_replay_rewards=moments(rewards),reward_chunks={str(start):moments(rewards[start:start+10000]) for start in range(0,50000,10000)},
            sample_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),sample_rows=len(indices),
            sampled_target_components={k:moments(v) for k,v in parts.items()},next_log_prob=moments(logp.numpy()),
            replay_action_q=[moments(current[:,i]) for i in range(2)],critic_rmse=[float(np.sqrt(np.mean(residual[:,i]**2))) for i in range(2)],
            timeout_rows=int(buffer.timeouts[:buffer.pos,0].sum()),done_rows=int(buffer.dones[:buffer.pos,0].sum()),
            updates=updates,weights_unchanged=True,weight_state_sha256=before,
            training_log=[dict(steps=r['steps'],phase=r['phase'],values=r.get('training',{})) for r in m['validation_checkpoints']]))
    assert all(checksum(Path(p))==h for p,h in protected.items())
    result=dict(records=records,protected_sha256=protected,source_files_unchanged=True,training_performed=False,optimizer_steps=0,
        dataset_opened=False,test_evaluated=False,colab_allocated=False,objective_achieved=False,
        interpretation='Final saved training replay and critics only; no early checkpoint critic/replay reconstruction. Fixed sampled one-step soft targets include timeout bootstrapping. Reward/Q/entropy scale mismatch is descriptive, not proof of causal failure or profitable trades. No future decline cohorts used.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('Audited3 final SAC/replay pairs with zero optimizer steps; weights/files unchanged.')


if __name__=='__main__':main()
