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
    config=SimulationConfig()
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    episodes=[e for e in dataset["episodes"].values() if e["split"]=="train"]
    env=TradingEnv(episodes,normalizer_from(dataset["normalizer"]),dataset["names"],config,random_start=True,episode_steps=128 if smoke else 512,reward_scale=reward_scale,reward_kind=reward_kind,start_sampling=start_sampling,observation_profile=observation_profile,exposure_execution=training_exposure_mode)
    env.action_space.seed(seed);env.observation_space.seed(seed)
    gym_check(env,skip_render_check=True);sb3_check(env)
    env.reset_sampling_statistics()
    model=create(algorithm,env,seed,device,smoke,overrides,policy_variant)
    initialized=initialize_policy(model,initialization,algorithm)
    steps=steps if steps is not None else 512 if algorithm in ON_POLICY else 1024
    if steps<=0:raise ValueError("positive training steps required")
    milestones=list(milestones) if milestones is not None else [steps]
    if not milestones or milestones!=sorted(set(milestones)) or milestones[-1]!=steps or milestones[0]<=0:
        raise ValueError('ordered positive milestones ending at total steps required')
    if len(milestones)>1 and algorithm!='sac':raise ValueError('segmented learning study currently requires SAC')
    every=evaluation_every if evaluation_every is not None else 256 if smoke else 10000
    if not isinstance(every,int) or every<=0:raise ValueError('positive evaluation interval required')
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
        "initialization":initialized,"observation":env.observation_spec,
        "policy_architecture":architecture_spec(policy_variant,len(dataset["names"])),
        "research_variant":initialized['arm']+('' if start_sampling=='legacy_uniform' else ':'+start_sampling)+('' if observation_profile=='legacy' else ':'+observation_profile)+('' if policy_variant=='native' else ':'+policy_variant)+('' if reward_kind=='net_log_equity' else ':'+reward_kind),
        "start_sampling":env.start_sampler.summary if env.start_sampler else {'mode':'legacy_uniform'},
        "evaluation_every":every,"milestones":milestones,"compact_checkpoints":compact_checkpoints,
        "versions":versions,
        "nondeterminism":"Seeds fixed; hardware/backend floating point may differ", "resume":"Replay buffer saved; exact RNG/env restoration not claimed"}
    if study_protocol_sha256 is not None:metadata['study_protocol_sha256']=study_protocol_sha256
    from rl_stocks.training.recurrent_state import PROTOCOL_SHA256 as RECURRENT_PROTOCOL,model_record as recurrent_record
    if study_protocol_sha256==RECURRENT_PROTOCOL:
        metadata['recurrent_state_initialization']=recurrent_record(model)
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
    try:
        previous=0
        for milestone in milestones:
            model.learn(total_timesteps=milestone-previous,callback=callback,reset_num_timesteps=previous==0)
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
        if algorithm=='sac' and study_protocol_sha256==REG_PROTOCOL:
            metadata['regularization_final']=optimizer_record(model)
            metadata['regularization_selected']=optimizer_record(selected)
        if algorithm=='sac' and study_protocol_sha256==DISCOUNT_PROTOCOL:
            metadata['discount_final']=discount_record(model)
            metadata['discount_selected']=discount_record(selected)
        if study_protocol_sha256==RECURRENT_PROTOCOL:
            metadata['recurrent_state_final']=recurrent_record(model)
            metadata['recurrent_state_selected']=recurrent_record(selected)
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
