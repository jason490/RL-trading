from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import random
import shutil
from pathlib import Path
import sys
import time
import tarfile
import uuid
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env as sb3_check
from gymnasium.utils.env_checker import check_env as gym_check
from rl_stocks.data.generation.datasets import read_dataset, normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.rl.registry.factory import create,resolved_parameters,ON_POLICY
from rl_stocks.evaluation.runner import evaluate,selection_score
from rl_stocks.reporting.html import report
from rl_stocks.inference.bundle import freeze,FrozenPolicy
from rl_stocks.core.provenance import source_identity,repository_state
from rl_stocks.training.curves import curve_record,append_record,render_curve
from rl_stocks.training.initialization import initialization_spec,initialize_policy

class ValidationCheckpoint(BaseCallback):
    def __init__(self,dataset,config,path,every,compact=False,target_steps=None,sampling_env=None,observation_profile="legacy",exposure_mode="unrestricted"):
        super().__init__();self.dataset=dataset;self.config=config;self.path=path
        self.every=every;self.best=-float("inf");self.history=[]
        self.best_step=None;self.best_phase=None;self.best_metrics={}
        self.compact=compact;self.target_steps=target_steps;self.started=time.monotonic()
        self.sampling_env=sampling_env;self.observation_profile=observation_profile;self.exposure_mode=exposure_mode
    def validate(self,phase='on_step'):
        if self.sampling_env is not None and self.sampling_env.start_sampler:
            (self.path/'training-sampling.json').write_text(json.dumps(self.sampling_env.sampling_statistics(),indent=2))
        if self.sampling_env is not None:
            (self.path/'training-execution.json').write_text(json.dumps(self.sampling_env.execution_statistics(),indent=2))
        results=evaluate(self.model,self.dataset,"validation",self.config,self.observation_profile,self.exposure_mode)
        score=selection_score(results)
        training={}
        for key in ('train/ent_coef','train/actor_loss','train/critic_loss','train/ent_coef_loss',
                    'train/entropy_loss','train/approx_kl','train/clip_fraction','train/value_loss',
                    'train/policy_gradient_loss','train/explained_variance','train/std','rollout/ep_rew_mean'):
            value=self.model.logger.name_to_value.get(key)
            if value is not None and np.isfinite(float(value)):training[key]=float(value)
        item=curve_record(results,self.history,self.num_timesteps,phase,time.monotonic()-self.started,training)
        self.history.append(item)
        append_record(self.path/'learning-curve.jsonl',item)
        render_curve(self.history,self.path/'learning-curve.html')
        filename=f"checkpoint-{self.num_timesteps}" if not self.compact or phase=='post_update' else 'latest-checkpoint'
        self.model.save(self.path/filename)
        if hasattr(self.model,"save_replay_buffer"):
            self.model.save_replay_buffer(self.path/"checkpoint-replay.pkl")
        improved=score>self.best
        if improved:
            self.best=score;self.model.save(self.path/"best")
            self.best_step=self.num_timesteps;self.best_phase=phase
            self.best_metrics={key:{"symbol":value["symbol"],"policy":value["metrics"],
                "baselines":{name:baseline["metrics"] for name,baseline in value["baselines"].items()}}
                for key,value in results.items()}
        (self.path/"checkpoints.json").write_text(json.dumps(self.history,indent=2))
        if improved or not self.compact or phase=='post_update' and self.num_timesteps==self.target_steps:
            report(results,self.path/"validation.html","Validation; selection only",self.history)
            if improved:
                for suffix in ("html","json"):
                    shutil.copyfile(self.path/f"validation.{suffix}",self.path/f"selected-validation.{suffix}")
        from rl_stocks.training.checkpoint_state import write_checkpoint_state
        write_checkpoint_state(self.path,results,filename+'.zip',self.num_timesteps,phase,
            self.best_step,self.best_phase,
            final_post_update=phase=='post_update' and self.num_timesteps==self.target_steps)
        snapshot=os.environ.get("RL_CHECKPOINT_ARCHIVE")
        if snapshot:
            target=Path(snapshot)
            temporary=target.with_suffix(".tmp")
            with tarfile.open(temporary,"w:gz") as archive:
                archive.add(self.path.parent.parent,arcname="artifacts")
            temporary.replace(target)
    def _on_step(self):
        if not np.isfinite(self.locals["rewards"]).all():raise ValueError("nonfinite reward")
        if self.num_timesteps%self.every==0:self.validate()
        for name,value in self.model.logger.name_to_value.items():
            if "loss" in name and not np.isfinite(value):raise ValueError("nonfinite loss")
        return True

def git_state():
    return repository_state(Path(__file__).resolve().parents[3])

def train(dataset_path,algorithm,output="artifacts",seed=11,device="cpu",smoke=True,steps=None,final_test=False,overrides=None,
          evaluation_every=None,compact_checkpoints=False,milestones=None,reward_scale=1.,initialization='standard',start_sampling='legacy_uniform',observation_profile='legacy',policy_variant='native',reward_kind='net_log_equity',study_protocol_sha256=None,training_exposure_mode='unrestricted',evaluation_exposure_mode='unrestricted'):
    initialization_spec(initialization,algorithm)
    from rl_stocks.risk.exposure import execution_contract,filter_exposure
    action_execution=execution_contract(training_exposure_mode,evaluation_exposure_mode)
    from rl_stocks.training.dependencies import verify_environment
    versions=verify_environment()
    dataset=read_dataset(dataset_path)
    from rl_stocks.training import treasury_context,treasury_initialization
    treasury_study=treasury_context.validate_dispatch(dataset,study_protocol_sha256)
    if treasury_study:
        treasury_arm=treasury_context.validate_request(Path(dataset_path).name,algorithm,overrides,
            seed=seed,smoke=smoke,steps=steps if steps is not None else 512,
            every=evaluation_every if evaluation_every is not None else 256 if smoke else 10000,
            initialization=initialization,policy_variant=policy_variant,observation_profile=observation_profile,
            start_sampling=start_sampling,reward_kind=reward_kind,reward_scale=reward_scale,
            training_exposure_mode=training_exposure_mode,evaluation_exposure_mode=evaluation_exposure_mode,
            final_test=final_test,compact_checkpoints=compact_checkpoints,milestones=milestones,device=device)
        treasury_context.validate_payload(dataset,treasury_arm)
    from rl_stocks.training import vix_context
    vix_study=vix_context.validate_dispatch(dataset,study_protocol_sha256)
    if vix_study:
        vix_arm=vix_context.validate_request(Path(dataset_path).name,algorithm,overrides,
            seed=seed,smoke=smoke,steps=steps if steps is not None else 512,
            every=evaluation_every if evaluation_every is not None else 256 if smoke else 10000,
            initialization=initialization,policy_variant=policy_variant,observation_profile=observation_profile,
            start_sampling=start_sampling,reward_kind=reward_kind,reward_scale=reward_scale,
            training_exposure_mode=training_exposure_mode,evaluation_exposure_mode=evaluation_exposure_mode,
            final_test=final_test,compact_checkpoints=compact_checkpoints,milestones=milestones)
        vix_context.validate_payload(dataset,vix_arm)
    from rl_stocks.training import gae_context,gae_initialization
    gae_study=study_protocol_sha256==gae_context.PROTOCOL_SHA256
    if gae_study:
        gae_arm=gae_context.validate_request(Path(dataset_path).name,algorithm,overrides,
            seed=seed,smoke=smoke,steps=steps if steps is not None else 512,
            every=evaluation_every if evaluation_every is not None else 256 if smoke else 10000,
            initialization=initialization,policy_variant=policy_variant,observation_profile=observation_profile,
            start_sampling=start_sampling,reward_kind=reward_kind,reward_scale=reward_scale,
            training_exposure_mode=training_exposure_mode,evaluation_exposure_mode=evaluation_exposure_mode,
            final_test=final_test,compact_checkpoints=compact_checkpoints,milestones=milestones,device=device)
        gae_context.validate_payload(dataset)
    config=SimulationConfig()
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    episodes=[e for e in dataset["episodes"].values() if e["split"]=="train"]
    env=TradingEnv(episodes,normalizer_from(dataset["normalizer"]),dataset["names"],config,random_start=True,episode_steps=128 if smoke else 512,reward_scale=reward_scale,reward_kind=reward_kind,start_sampling=start_sampling,observation_profile=observation_profile,exposure_execution=training_exposure_mode)
    env.action_space.seed(seed);env.observation_space.seed(seed)
    gym_check(env,skip_render_check=True);sb3_check(env)
    env.reset_sampling_statistics();env.reset_execution_statistics()
    if treasury_study:
        from rl_stocks.training.treasury_sampling import SamplingPaths
        treasury_paths=SamplingPaths(env)
    model=create(algorithm,env,seed,device,smoke,overrides,policy_variant)
    if treasury_study:
        treasury_shared=treasury_initialization.apply_initialization(model,treasury_arm,seed,smoke)
    if vix_study:
        vix_shared=vix_context.apply_initialization(model,vix_arm,seed,smoke)
    if gae_study:
        gae_shared=gae_initialization.apply_initialization(model,gae_arm,seed,smoke)
    from rl_stocks.training.recurrent_guard import (PROTOCOL_SHA256 as GUARD_PROTOCOL,
        model_record as guard_record,apply_shared_initialization,initialization_path)
    shared_initialization=apply_shared_initialization(model,seed,smoke) if study_protocol_sha256==GUARD_PROTOCOL else None
    from rl_stocks.training.session_context import (PROTOCOL_SHA256 as SESSION_PROTOCOL,protocol as session_protocol,
        apply_initialization as apply_session_initialization,model_record as session_record,initialization_entry as session_initialization_entry,
        validate_payload as validate_session_payload,ROOT as SESSION_ROOT)
    if study_protocol_sha256==SESSION_PROTOCOL:
        session_arm=next(a for a,d in session_protocol()['datasets'].items() if d==Path(dataset_path).name)
        validate_session_payload(dataset,session_arm)
        session_shared=apply_session_initialization(model,session_arm,seed,smoke)
    from rl_stocks.training.recurrent_capacity import (PROTOCOL_SHA256 as CAPACITY_PROTOCOL,identify_parameters as capacity_parameters,
        apply_initialization as apply_capacity_initialization,model_record as capacity_record,initialization_entry as capacity_entry,
        validate_payload as validate_capacity_payload,ROOT as CAPACITY_ROOT)
    if study_protocol_sha256==CAPACITY_PROTOCOL:
        capacity_arm=capacity_parameters(overrides,smoke);validate_capacity_payload(dataset)
        capacity_shared=apply_capacity_initialization(model,capacity_arm,seed,smoke)
    from rl_stocks.training.squashed_actions import (PROTOCOL_SHA256 as SQUASHED_PROTOCOL,identify_parameters as squashed_parameters,
        apply_initialization as apply_squashed_initialization,model_record as squashed_record,initialization_entry as squashed_entry,
        validate_payload as validate_squashed_payload,ROOT as SQUASHED_ROOT)
    if study_protocol_sha256==SQUASHED_PROTOCOL:
        squashed_arm=squashed_parameters(overrides,smoke,algorithm);validate_squashed_payload(dataset)
        squashed_shared=apply_squashed_initialization(model,squashed_arm,seed,smoke)
    from rl_stocks.training.recurrent_regularization import (PROTOCOL_SHA256 as L2_PROTOCOL,identify_parameters as l2_parameters,
        apply_initialization as apply_l2_initialization,model_record as l2_record,initialization_entry as l2_entry,
        validate_payload as validate_l2_payload,ROOT as L2_ROOT)
    if study_protocol_sha256==L2_PROTOCOL:
        l2_arm=l2_parameters(overrides,smoke);validate_l2_payload(dataset)
        l2_shared=apply_l2_initialization(model,l2_arm,seed,smoke)
    from rl_stocks.training.market_context import (PROTOCOL_SHA256 as MARKET_PROTOCOL,identify_parameters as market_parameters,
        apply_initialization as apply_market_initialization,model_record as market_record,initialization_entry as market_entry,
        validate_payload as validate_market_payload,ROOT as MARKET_ROOT)
    if study_protocol_sha256==MARKET_PROTOCOL:
        market_arm=market_parameters(overrides,smoke,Path(dataset_path).name);validate_market_payload(dataset,market_arm)
        market_shared=apply_market_initialization(model,market_arm,seed,smoke)
    from rl_stocks.training.input_clipping import (PROTOCOL_SHA256 as CLIPPING_PROTOCOL,identify_parameters as clipping_parameters,
        apply_initialization as apply_clipping_initialization,model_record as clipping_record,initialization_entry as clipping_entry,
        validate_payload as validate_clipping_payload,ROOT as CLIPPING_ROOT)
    if study_protocol_sha256==CLIPPING_PROTOCOL:
        clipping_arm=clipping_parameters(overrides,smoke,Path(dataset_path).name);validate_clipping_payload(dataset,clipping_arm)
        clipping_shared=apply_clipping_initialization(model,clipping_arm,seed,smoke)
    from rl_stocks.training.training_history import (PROTOCOL_SHA256 as HISTORY_PROTOCOL,identify_parameters as history_parameters,
        apply_initialization as apply_history_initialization,model_record as history_record,initialization_entry as history_entry,
        validate_payload as validate_history_payload,ROOT as HISTORY_ROOT)
    if study_protocol_sha256==HISTORY_PROTOCOL:
        recency_arm=history_parameters(overrides,smoke,Path(dataset_path).name);validate_history_payload(dataset,recency_arm)
        history_shared=apply_history_initialization(model,recency_arm,seed,smoke)
    from rl_stocks.training.distributional_critics import (PROTOCOL_SHA256 as CRITIC_PROTOCOL,
        identify_parameters as critic_parameters, validate_payload as validate_critic_payload,
        apply_initialization as apply_critic_initialization)
    if study_protocol_sha256==CRITIC_PROTOCOL:
        if (initialization!='standard' or policy_variant!='native' or observation_profile!='execution_full'
                or start_sampling!='legacy_uniform' or reward_kind!='net_log_equity' or reward_scale!=1.
                or training_exposure_mode!='unrestricted' or evaluation_exposure_mode!='unrestricted' or final_test):
            raise ValueError('fixed critic-study input/reward/execution contract required')
        critic_arm=critic_parameters(overrides,algorithm,smoke)
        validate_critic_payload(dataset)
        critic_shared=apply_critic_initialization(model,critic_arm,seed,smoke)
    initialized=initialize_policy(model,initialization,algorithm)
    from rl_stocks.rl.registry.distribution import model_distribution
    action_distribution=model_distribution(model)
    steps=steps if steps is not None else 512 if algorithm in ON_POLICY else 1024
    if steps<=0:raise ValueError("positive training steps required")
    milestones=list(milestones) if milestones is not None else [steps]
    if not milestones or milestones!=sorted(set(milestones)) or milestones[-1]!=steps or milestones[0]<=0:
        raise ValueError('ordered positive milestones ending at total steps required')
    if len(milestones)>1 and algorithm!='sac':raise ValueError('segmented learning study currently requires SAC')
    every=evaluation_every if evaluation_every is not None else 256 if smoke else 10000
    if not isinstance(every,int) or every<=0:raise ValueError('positive evaluation interval required')
    if study_protocol_sha256==CRITIC_PROTOCOL:
        from rl_stocks.training.distributional_critics import protocol as critic_protocol
        spec=critic_protocol()
        if steps!=spec['smoke_steps' if smoke else 'steps_per_cell'] or every!=spec['smoke_evaluation_every' if smoke else 'evaluation_every']:
            raise ValueError('fixed critic-study transitions/checkpoint cadence required')
    if algorithm in ON_POLICY and steps%model.n_steps:raise ValueError("steps must be an exact PPO rollout multiple")
    run=Path(output)/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+f"-{algorithm}-{seed}-"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    from rl_stocks.rl.ppo.architecture import architecture_spec
    metadata={"algorithm":algorithm,"seed":seed,"device":device,"requested_steps":steps,"smoke":smoke,
        "python":sys.version.split()[0],
        "resolved_parameters":resolved_parameters(algorithm,smoke,overrides,policy_variant,len(dataset["names"])),"simulation":asdict(config),
        "dataset":Path(dataset_path).name,"git":git_state(),"source_hash":source_identity(),
        "selection_metric":"Equal-weight mean validation net return", "status":"running",
        "paper_eligible":False,"dependency_lock_present":(Path(__file__).resolve().parents[3]/"requirements.lock").exists(),
        "reward_config":{"kind":reward_kind,"scale":reward_scale},"action_execution":action_execution,
        "action_distribution":action_distribution,
        "initialization":initialized,"observation":env.observation_spec,
        "policy_architecture":architecture_spec(policy_variant,len(dataset["names"])),
        "research_variant":initialized['arm']+('' if start_sampling=='legacy_uniform' else ':'+start_sampling)+('' if observation_profile=='legacy' else ':'+observation_profile)+('' if policy_variant=='native' else ':'+policy_variant)+('' if reward_kind=='net_log_equity' else ':'+reward_kind),
        "start_sampling":env.start_sampler.summary if env.start_sampler else {'mode':'legacy_uniform'},
        "evaluation_every":every,"milestones":milestones,"compact_checkpoints":compact_checkpoints,
        "versions":versions,
        "nondeterminism":"Seeds fixed; hardware/backend floating point may differ", "resume":"Replay buffer saved; exact RNG/env restoration not claimed"}
    if algorithm=='tqc':
        from rl_stocks.rl.tqc.geometry import model_spec as tqc_spec
        metadata['critic_geometry']=tqc_spec(model)
    if treasury_study:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'treasury-initialization.npz'
        shutil.copyfile(treasury_initialization.portable_path(treasury_shared['artifact_file']),snapshot)
        if checksum(snapshot)!=treasury_shared['artifact_sha256']:
            raise ValueError('Treasury initialization snapshot checksum mismatch')
        metadata['treasury_arm']=treasury_arm
        metadata['treasury_binding_sha256']=treasury_initialization.BINDING_SHA256
        metadata['treasury_shared_initialization']=treasury_shared
        metadata['treasury_initialization']=treasury_initialization.model_record(model)
        treasury_initialization.validate_record(metadata['treasury_initialization'],treasury_arm,smoke,initial=True,seed=seed)
    if gae_study:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'gae-initialization.npz'
        shutil.copyfile(gae_initialization.ROOT/gae_shared['artifact_file'],snapshot)
        if checksum(snapshot)!=gae_shared['artifact_sha256']:
            raise ValueError('GAE initialization snapshot checksum mismatch')
        metadata['gae_arm']=gae_arm
        metadata['gae_shared_initialization']=gae_shared
        metadata['gae_initialization']=gae_initialization.model_record(model)
        gae_initialization.validate_record(metadata['gae_initialization'],gae_arm,smoke,initial=True,seed=seed)
    if vix_study:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'vix-initialization.npz'
        shutil.copyfile(vix_context.ROOT/vix_shared['artifact_file'],snapshot)
        if checksum(snapshot)!=vix_shared['artifact_sha256']:
            raise ValueError('VIX initialization snapshot checksum mismatch')
        metadata['vix_arm']=vix_arm
        metadata['vix_binding_sha256']=vix_context.BINDING_SHA256
        metadata['vix_shared_initialization']=vix_shared
        metadata['vix_initialization']=vix_context.model_record(model)
        vix_context.validate_record(metadata['vix_initialization'],vix_arm,smoke,initial=True,seed=seed)
    if study_protocol_sha256==CRITIC_PROTOCOL:
        from rl_stocks.training.distributional_records import model_record as critic_record,validate_record as validate_critic_record
        from rl_stocks.training.distributional_assets import initialization_entry as critic_entry
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'critic-initialization.npz'
        shutil.copyfile(Path(__file__).resolve().parents[3]/critic_entry(critic_arm,seed,smoke)[0]['file'],snapshot)
        if checksum(snapshot)!=critic_shared['artifact_sha256']:
            raise ValueError('critic initialization snapshot checksum mismatch')
        metadata['critic_arm']=critic_arm
        metadata['critic_shared_initialization']=critic_shared
        metadata['critic_initialization']=critic_record(model)
        validate_critic_record(metadata['critic_initialization'],critic_arm,smoke,initial=True,seed=seed)
    if study_protocol_sha256 is not None:metadata['study_protocol_sha256']=study_protocol_sha256
    from rl_stocks.training.recurrent_state import PROTOCOL_SHA256 as RECURRENT_PROTOCOL,model_record as recurrent_record
    if study_protocol_sha256==RECURRENT_PROTOCOL:
        metadata['recurrent_state_initialization']=recurrent_record(model)
    from rl_stocks.training.recurrent_alignment import PROTOCOL_SHA256 as ALIGNMENT_PROTOCOL
    if study_protocol_sha256==ALIGNMENT_PROTOCOL:
        metadata['alignment_initialization']=recurrent_record(model)
    from rl_stocks.training.recurrent_cash import PROTOCOL_SHA256 as CASH_PROTOCOL,model_record as cash_record
    if study_protocol_sha256==CASH_PROTOCOL:
        metadata['cash_initialization']=cash_record(model)
    if study_protocol_sha256==GUARD_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'guard-initialization.npz'
        shutil.copyfile(initialization_path(seed,smoke),snapshot)
        if checksum(snapshot)!=shared_initialization['artifact_sha256']:raise ValueError('initialization snapshot checksum mismatch')
        metadata['guard_shared_initialization']=shared_initialization
        metadata['guard_initialization']=guard_record(model)
    if study_protocol_sha256==SESSION_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'session-initialization.npz'
        shutil.copyfile(SESSION_ROOT/session_initialization_entry(session_arm,seed,smoke)['file'],snapshot)
        if checksum(snapshot)!=session_shared['artifact_sha256']:raise ValueError('session snapshot checksum mismatch')
        metadata['session_context_arm']=session_arm
        metadata['session_context_shared_initialization']=session_shared
        metadata['session_context_initialization']=session_record(model)
    if study_protocol_sha256==CAPACITY_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'capacity-initialization.npz'
        shutil.copyfile(CAPACITY_ROOT/capacity_entry(capacity_arm,seed,smoke)['file'],snapshot)
        if checksum(snapshot)!=capacity_shared['artifact_sha256']:raise ValueError('capacity snapshot checksum mismatch')
        metadata['capacity_arm']=capacity_arm
        metadata['capacity_shared_initialization']=capacity_shared
        metadata['capacity_initialization']=capacity_record(model)
    if study_protocol_sha256==SQUASHED_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'squashed-initialization.npz'
        shutil.copyfile(SQUASHED_ROOT/squashed_entry(squashed_arm,seed,smoke)['file'],snapshot)
        if checksum(snapshot)!=squashed_shared['artifact_sha256']:raise ValueError('squashed snapshot checksum mismatch')
        metadata['squashed_arm']=squashed_arm
        metadata['squashed_shared_initialization']=squashed_shared
        metadata['squashed_initialization']=squashed_record(model)
    if study_protocol_sha256==L2_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'l2-initialization.npz'
        shutil.copyfile(L2_ROOT/l2_entry(l2_arm,seed,smoke)['file'],snapshot)
        if checksum(snapshot)!=l2_shared['artifact_sha256']:raise ValueError('l2 snapshot checksum mismatch')
        metadata['l2_arm']=l2_arm
        metadata['l2_shared_initialization']=l2_shared
        metadata['l2_initialization']=l2_record(model)
    if study_protocol_sha256==MARKET_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'market-initialization.npz'
        shutil.copyfile(MARKET_ROOT/market_entry(market_arm,seed,smoke)['file'],snapshot)
        if checksum(snapshot)!=market_shared['artifact_sha256']:raise ValueError('market snapshot checksum mismatch')
        metadata['market_arm']=market_arm
        metadata['market_shared_initialization']=market_shared
        metadata['market_initialization']=market_record(model)
    if study_protocol_sha256==CLIPPING_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'clipping-initialization.npz'
        shutil.copyfile(CLIPPING_ROOT/clipping_entry(clipping_arm,seed,smoke)['file'],snapshot)
        if checksum(snapshot)!=clipping_shared['artifact_sha256']:raise ValueError('clipping snapshot checksum mismatch')
        metadata['clipping_arm']=clipping_arm
        metadata['clipping_shared_initialization']=clipping_shared
        metadata['clipping_initialization']=clipping_record(model)
    if study_protocol_sha256==HISTORY_PROTOCOL:
        from rl_stocks.inference.bundle import checksum
        snapshot=run/'recency-initialization.npz'
        shutil.copyfile(HISTORY_ROOT/history_entry(recency_arm,seed,smoke)['file'],snapshot)
        if checksum(snapshot)!=history_shared['artifact_sha256']:raise ValueError('history snapshot checksum mismatch')
        metadata['recency_arm']=recency_arm
        metadata['recency_shared_initialization']=history_shared
        metadata['recency_initialization']=history_record(model)
    if algorithm=='td3':
        from rl_stocks.training.td3_state import PROTOCOL_SHA256 as TD3_PROTOCOL,initialization_record as td3_initialization_record
        if study_protocol_sha256==TD3_PROTOCOL:metadata['td3_initialization']=td3_initialization_record(model)
    if algorithm=='sac':
        from rl_stocks.training.sac_temperature import PROTOCOL_SHA256 as TEMP_PROTOCOL,initialization_record
        if study_protocol_sha256==TEMP_PROTOCOL:metadata['temperature_initialization']=initialization_record(model)
        from rl_stocks.training.diversity import PROTOCOL_SHA256 as DIVERSITY_PROTOCOL
        if study_protocol_sha256==DIVERSITY_PROTOCOL:metadata['diversity_initialization']=initialization_record(model)
        from rl_stocks.training.sac_regularization import PROTOCOL_SHA256 as REG_PROTOCOL,optimizer_record
        if study_protocol_sha256==REG_PROTOCOL:metadata['regularization_initialization']=optimizer_record(model)
        from rl_stocks.training.sac_discount import PROTOCOL_SHA256 as DISCOUNT_PROTOCOL,model_record as discount_record
        if study_protocol_sha256==DISCOUNT_PROTOCOL:metadata['discount_initialization']=discount_record(model)
    if 'history_augmentation' in dataset['manifest']:
        from rl_stocks.training.history.spec import PROTOCOL_SHA256
        metadata['history_augmentation']=dataset['manifest']['history_augmentation']
        metadata['history_protocol_sha256']=PROTOCOL_SHA256
    (run/"metadata.json").write_text(json.dumps(metadata,indent=2))
    start=time.monotonic();callback=ValidationCheckpoint(dataset,config,run,every,compact_checkpoints,steps,sampling_env=env,observation_profile=observation_profile,exposure_mode=evaluation_exposure_mode)
    callbacks=callback
    if gae_study:
        from rl_stocks.training.gae_rollouts import GaeRolloutAudit
        gae_audit=GaeRolloutAudit(run,gae_arm,smoke)
        callbacks=[callback,gae_audit]
    try:
        previous=0
        for milestone in milestones:
            model.learn(total_timesteps=milestone-previous,callback=callbacks,reset_num_timesteps=previous==0)
            # On-step evaluations precede the optimizer update; always capture
            # the completed milestone weights as a distinct post-update record.
            callback.validate(phase='post_update')
            previous=milestone
        if not all(torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
            raise ValueError("nonfinite model weights")
        model.save(run/"latest")
        if hasattr(model,'save_replay_buffer'):model.save_replay_buffer(run/"replay.pkl")
        from rl_stocks.rl.registry.factory import CLASSES
        selected=CLASSES[algorithm].load(run/"best.zip",device=device)
        sampling_diagnostics=env.sampling_statistics()
        observation,_=env.reset(seed=seed)
        from rl_stocks.rl.registry.prediction import recurrent
        if recurrent(selected):
            original,original_state=selected.predict(observation,state=None,episode_start=np.array([True]),deterministic=True)
        else:original=selected.predict(observation,deterministic=True)[0]
        original=filter_exposure(original,evaluation_exposure_mode)
        metadata.update(status="complete",actual_steps=model.num_timesteps,duration_seconds=time.monotonic()-start,
            validation_checkpoints=callback.history,selected_score=callback.best)
        metadata.update(selected_step=callback.best_step,selected_metrics=callback.best_metrics)
        metadata['sampling_diagnostics']=sampling_diagnostics
        metadata['training_execution']=env.execution_statistics()
        if treasury_study:
            from rl_stocks.training.treasury_sampling import validate_paths
            metadata['treasury_sampling_paths']=validate_paths(treasury_paths.snapshot(),sampling_diagnostics,model.num_timesteps)
            metadata['treasury_final']=treasury_initialization.model_record(model)
            metadata['treasury_selected']=treasury_initialization.model_record(selected)
            treasury_initialization.validate_record(metadata['treasury_final'],treasury_arm,smoke,seed=seed)
            treasury_initialization.validate_record(metadata['treasury_selected'],treasury_arm,smoke,
                selected_steps=metadata['selected_step'],seed=seed)
        if gae_study:
            metadata['gae_rollouts']=gae_audit.summary()
            metadata['gae_final']=gae_initialization.model_record(model)
            metadata['gae_selected']=gae_initialization.model_record(selected)
            gae_initialization.validate_record(metadata['gae_final'],gae_arm,smoke,seed=seed)
            gae_initialization.validate_record(metadata['gae_selected'],gae_arm,smoke,
                selected_steps=metadata['selected_step'],seed=seed)
        if vix_study:
            metadata['vix_final']=vix_context.model_record(model)
            metadata['vix_selected']=vix_context.model_record(selected)
            vix_context.validate_record(metadata['vix_final'],vix_arm,smoke,seed=seed)
            vix_context.validate_record(metadata['vix_selected'],vix_arm,smoke,
                selected_steps=metadata['selected_step'],seed=seed)
        if algorithm=='sac' and study_protocol_sha256==REG_PROTOCOL:
            metadata['regularization_final']=optimizer_record(model)
            metadata['regularization_selected']=optimizer_record(selected)
        if algorithm=='sac' and study_protocol_sha256==DISCOUNT_PROTOCOL:
            metadata['discount_final']=discount_record(model)
            metadata['discount_selected']=discount_record(selected)
        if study_protocol_sha256==RECURRENT_PROTOCOL:
            metadata['recurrent_state_final']=recurrent_record(model)
            metadata['recurrent_state_selected']=recurrent_record(selected)
        if study_protocol_sha256==ALIGNMENT_PROTOCOL:
            metadata['alignment_final']=recurrent_record(model)
            metadata['alignment_selected']=recurrent_record(selected)
        if study_protocol_sha256==CASH_PROTOCOL:
            metadata['cash_final']=cash_record(model)
            metadata['cash_selected']=cash_record(selected)
        if study_protocol_sha256==GUARD_PROTOCOL:
            metadata['guard_final']=guard_record(model)
            metadata['guard_selected']=guard_record(selected)
        if study_protocol_sha256==SESSION_PROTOCOL:
            metadata['session_context_final']=session_record(model)
            metadata['session_context_selected']=session_record(selected)
        if study_protocol_sha256==CAPACITY_PROTOCOL:
            metadata['capacity_final']=capacity_record(model)
            metadata['capacity_selected']=capacity_record(selected)
        if study_protocol_sha256==SQUASHED_PROTOCOL:
            metadata['squashed_final']=squashed_record(model)
            metadata['squashed_selected']=squashed_record(selected)
        if study_protocol_sha256==L2_PROTOCOL:
            metadata['l2_final']=l2_record(model)
            metadata['l2_selected']=l2_record(selected)
        if study_protocol_sha256==MARKET_PROTOCOL:
            metadata['market_final']=market_record(model)
            metadata['market_selected']=market_record(selected)
        if study_protocol_sha256==CLIPPING_PROTOCOL:
            metadata['clipping_final']=clipping_record(model)
            metadata['clipping_selected']=clipping_record(selected)
        if study_protocol_sha256==HISTORY_PROTOCOL:
            metadata['recency_final']=history_record(model)
            metadata['recency_selected']=history_record(selected)
        if study_protocol_sha256==CRITIC_PROTOCOL:
            metadata['critic_final']=critic_record(model)
            metadata['critic_selected']=critic_record(selected)
            validate_critic_record(metadata['critic_final'],critic_arm,smoke,seed=seed)
            validate_critic_record(metadata['critic_selected'],critic_arm,smoke,selected_steps=metadata['selected_step'],seed=seed)
        bundle=freeze(run/"best.zip",dataset,metadata,Path(output)/"frozen")
        frozen=FrozenPolicy(bundle,dataset["schema"],dataset["manifest"]["provenance"]["capabilities"]["feed"],device)
        if recurrent(selected):
            action,state=frozen.predict_recurrent(observation,state=None,episode_start=True)
            np.testing.assert_array_equal(original,action)
            for a,b in zip(original_state,state):np.testing.assert_array_equal(a,b)
        else:np.testing.assert_array_equal(original,frozen.predict(observation))
        metadata["bundle_id"]=bundle.name
        if final_test:
            results=evaluate(selected,dataset,"test",config,observation_profile,evaluation_exposure_mode)
            report(results,run/"final-test.html","Held-out audit of frozen validation selection")
        (run/"metadata.json").write_text(json.dumps(metadata,indent=2))
        return run,bundle
    except BaseException as exc:
        metadata.update(status="failed",failure_type=type(exc).__name__,actual_steps=model.num_timesteps,duration_seconds=time.monotonic()-start)
        (run/"metadata.json").write_text(json.dumps(metadata,indent=2))
        model.save(run/"interrupted")
        raise
    finally:env.close()
