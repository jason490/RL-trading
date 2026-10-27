"""Bounded actual Colab CPU profiling; no performance selection or local fitting."""
from pathlib import Path
import hashlib
import time

import numpy as np
from stable_baselines3.common.logger import configure
import torch

from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.execution_provenance import execution_backend
from rl_stocks.core.provenance import source_identity,repository_state
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.recurrent_sac import evaluate_recurrent_sac
from rl_stocks.evaluation.runner import evaluate
from rl_stocks.reporting.html import report
from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training.recurrent_collector import RecurrentCollector
from rl_stocks.training.recurrent_comparison import protocol,PROTOCOL_SHA256,profile_admission
from rl_stocks.training.recurrent_comparison_starts import load_start,lineage
from rl_stocks.training.recurrent_loop import validate_dataset,_atomic_json
from rl_stocks.training.recurrent_payload import RecurrentPayloadStore
from rl_stocks.training.recurrent_sac_checkpoint import save_checkpoint
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.sac_update_worker import preserved_rng


def validate_panel(dataset):
    spec=protocol()
    normalizer,training=validate_dataset(dataset,spec['dataset'])
    validation=[ep for ep in dataset['episodes'].values() if ep['split']=='validation']
    if (sorted(ep['symbol'] for ep in training)!=sorted(spec['training_symbols'])
            or sorted(ep['symbol'] for ep in validation)!=sorted(spec['validation_symbols'])):
        raise ValueError('exact full training/development panels required')
    return normalizer,training


def publish_profile_checkpoint(arm,model,output,timings):
    """Publish learned state before later profiling/evaluation can fail."""
    if arm not in protocol()['arms']:
        raise ValueError('registered profile arm required')
    output=Path(output)
    if arm=='gru_sac':
        save_checkpoint(model,output/arm,lineage=lineage())
    else:
        model.save(output/(arm+'.zip'))
    _atomic_json(output/'partial-profile.json',timings)
    export_artifacts(output.parent)


def execute_colab(dataset,starts,output):
    require_colab_cpu()  # before model construction, output or any optimizer call
    from rl_stocks.training.recurrent_comparison_worker import NativeAudit
    normalizer,training=validate_panel(dataset)
    output=Path(output)
    if output.parent.name!='artifacts' or output.exists():
        raise ValueError('fresh owned artifacts/profile output required')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    timings=dict(protocol_sha256=PROTOCOL_SHA256,dataset=digest(dataset),device='cpu',
                 default_python_family=[3,13],candidate_updates=0,native_sac_updates=0,
                 recurrent_ppo_transitions=0,recurrent_ppo_adam_steps=0,
                 candidate_collection_seconds=[],actor_refresh512_seconds=[],
                 candidate_sample_update_seconds=[],native_sac_update_seconds=[],
                 recurrent_ppo_rollout1000_seconds=[],validation_seconds=[],checkpoint_io_seconds=[])
    envs=[]
    counters=[]
    started=time.monotonic()
    def measured(operation):
        before=time.monotonic()
        value=operation()
        return value,time.monotonic()-before
    def training_env(profile):
        env=TradingEnv(training,normalizer,dataset['names'],random_start=True,episode_steps=512,
                       start_sampling='stock_etf_cycle_windows',observation_profile=profile)
        envs.append(env)
        return env
    try:
        candidate=load_start(starts,'gru_sac',11)
        candidate.max_updates=64  # bounded profile allowance, unchanged optimizer/model recipe
        store=RecurrentPayloadStore(100000,21)
        collector=RecurrentCollector(training_env('execution_full'),candidate.networks,store,
                                     seed=11,warmup_steps=64,max_episode_steps=512)
        rng=np.random.default_rng(11)
        rows=[]
        update_records=[]
        for step in range(1,513):
            row,elapsed=measured(lambda:collector.collect(policy_revision=candidate.updates))
            rows.append(row)
            timings['candidate_collection_seconds'].append(elapsed)
            if step>64 and (step-64)%7==0:
                def update():
                    windows=store.index.sample(batch_size=8,burn_in=16,unroll=16,rng=rng)
                    diagnostics=candidate.update(store.gather(windows))
                    update_records.append(dict(transitions=step,diagnostics=diagnostics))
                _,elapsed=measured(update)
                timings['candidate_sample_update_seconds'].append(elapsed)
        timings['candidate_updates']=candidate.updates
        if candidate.completed_steps!=dict.fromkeys(('actor','critic','entropy'),64):
            raise ValueError('actual GRU-SAC profile optimizer counts mismatch')
        np.savez_compressed(output/'collected-replay.npz',
            observations=np.stack([row.observation for row in rows]),
            next_observations=np.stack([row.next_observation for row in rows]),
            actions=np.stack([row.action for row in rows]),
            rewards=np.array([row.reward for row in rows]),
            terminated=np.array([row.terminated for row in rows]),
            truncated=np.array([row.truncated for row in rows]))
        (output/'candidate-updates.json').write_bytes(canonical(update_records))
        state_io={}
        _,state_io['gru_sac']=measured(lambda:publish_profile_checkpoint('gru_sac',candidate,output,timings))
        # Time the exact full-prefix refresh work, including conversion, without gradients.
        observations=np.stack([row.observation for row in rows])
        for _ in range(64):
            def refresh():
                with torch.no_grad():
                    candidate.networks.actor.memory(torch.tensor(observations.copy()).unsqueeze(0))
            _,elapsed=measured(refresh)
            timings['actor_refresh512_seconds'].append(elapsed)
        models={'gru_sac':candidate}
        native=load_start(starts,'native_sac',11)
        native.set_logger(configure(folder=None,format_strings=[]))
        for row in rows:
            native.replay_buffer.add(row.observation[None],row.next_observation[None],row.action[None],
                np.array([row.reward]),np.array([row.terminated or row.truncated]),
                [dict(**{'TimeLimit.truncated':row.truncated and not row.terminated})])
        native_audit=NativeAudit(native,'native_sac')
        native_audit.maximum=dict.fromkeys(native_audit.maximum,64)
        counters.append(native_audit)
        for _ in range(64):
            _,elapsed=measured(lambda:native.train(gradient_steps=1,batch_size=256))
            timings['native_sac_update_seconds'].append(elapsed)
        if native_audit.counts!=dict.fromkeys(('actor','critic','entropy'),64):
            raise ValueError('actual native SAC profile optimizer counts mismatch')
        timings['native_sac_updates']=native._n_updates
        models['native_sac']=native
        _,state_io['native_sac']=measured(lambda:publish_profile_checkpoint('native_sac',native,output,timings))
        ppo=load_start(starts,'recurrent_ppo_reference',11,env=training_env('execution_masked'))
        ppo_audit=NativeAudit(ppo,'recurrent_ppo_reference')
        ppo_audit.maximum=dict(policy=600)
        counters.append(ppo_audit)
        collect_rollouts,train=ppo.collect_rollouts,ppo.train
        def timed_rollout(*args,**kwargs):
            result,elapsed=measured(lambda:collect_rollouts(*args,**kwargs))
            timings['recurrent_ppo_rollout1000_seconds'].append(elapsed)
            return result
        def timed_train(*args,**kwargs):
            result,elapsed=measured(lambda:train(*args,**kwargs))
            timings['recurrent_ppo_rollout1000_seconds'][-1]+=elapsed
            return result
        ppo.collect_rollouts=timed_rollout
        ppo.train=timed_train
        try:
            ppo.learn(total_timesteps=3000)
        finally:
            del ppo.collect_rollouts
            del ppo.train
        timings['recurrent_ppo_transitions']=ppo.num_timesteps
        timings['recurrent_ppo_adam_steps']=ppo_audit.counts['policy']
        models['recurrent_ppo_reference']=ppo
        _,state_io['recurrent_ppo_reference']=measured(lambda:publish_profile_checkpoint('recurrent_ppo_reference',ppo,output,timings))
        # Common order for evaluation and complete checkpoint/report/archive timings.
        for arm in protocol()['arms']:
            model=models[arm]
            with preserved_rng():
                operation=(lambda:evaluate_recurrent_sac(model.networks,dataset,exposure_mode='unit_magnitude')) if arm=='gru_sac' else (
                    lambda:evaluate(model,dataset,split='validation',
                        observation_profile=protocol()['arm_contracts'][arm]['observation_profile'],
                        exposure_mode='unit_magnitude'))
                results,elapsed=measured(operation)
            timings['validation_seconds'].append(elapsed)
            def write_artifacts():
                (output/(arm+'-validation.json')).write_bytes(canonical(results))
                report(results,output/(arm+'-validation.html'),'Profiling only; no performance selection')
                export_artifacts(output.parent)
            _,elapsed=measured(write_artifacts)
            timings['checkpoint_io_seconds'].append(state_io[arm]+elapsed)
        (output/'profile.json').write_bytes(canonical(timings))
        projection=profile_admission(timings)
        (output/'projection.json').write_bytes(canonical(projection))
        (output/'provenance.json').write_bytes(canonical(dict(backend=execution_backend('cpu'),
            source_sha256=source_identity(),repository=repository_state(Path.cwd()),
            initialization_manifest_sha256=hashlib.sha256((Path(starts)/'manifest.json').read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-started,profile_only=True,
            performance_interpretation_allowed=False,prospective_confirmation_required=True)))
        return timings
    finally:
        try:
            (output/'partial-profile.json').write_bytes(canonical(timings))
        finally:
            try:
                for counter in counters:
                    counter.close()
                for env in envs:
                    env.close()
            finally:
                export_artifacts(output.parent)
