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
from rl_stocks.rl.registry.factory import create,resolved_parameters
from rl_stocks.evaluation.runner import evaluate,selection_score
from rl_stocks.reporting.html import report
from rl_stocks.inference.bundle import freeze,FrozenPolicy
from rl_stocks.core.provenance import source_identity,repository_state

class ValidationCheckpoint(BaseCallback):
    def __init__(self,dataset,config,path,every):
        super().__init__();self.dataset=dataset;self.config=config;self.path=path
        self.every=every;self.best=-float("inf");self.history=[]
        self.best_step=None;self.best_metrics={}
    def validate(self):
        results=evaluate(self.model,self.dataset,"validation",self.config)
        score=selection_score(results)
        item={"steps":self.num_timesteps,"score":score}
        self.history.append(item)
        self.model.save(self.path/f"checkpoint-{self.num_timesteps}")
        if hasattr(self.model,"save_replay_buffer"):
            self.model.save_replay_buffer(self.path/"checkpoint-replay.pkl")
        improved=score>self.best
        if improved:
            self.best=score;self.model.save(self.path/"best")
            self.best_step=self.num_timesteps
            self.best_metrics={key:{"symbol":value["symbol"],"policy":value["metrics"],
                "baselines":{name:baseline["metrics"] for name,baseline in value["baselines"].items()}}
                for key,value in results.items()}
        (self.path/"checkpoints.json").write_text(json.dumps(self.history,indent=2))
        report(results,self.path/"validation.html","Validation; selection only",self.history)
        if improved:
            for suffix in ("html","json"):
                shutil.copyfile(self.path/f"validation.{suffix}",self.path/f"selected-validation.{suffix}")
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

def train(dataset_path,algorithm,output="artifacts",seed=11,device="cpu",smoke=True,steps=None,final_test=False,overrides=None):
    from rl_stocks.training.dependencies import verify_environment
    versions=verify_environment()
    dataset=read_dataset(dataset_path)
    config=SimulationConfig()
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    episodes=[e for e in dataset["episodes"].values() if e["split"]=="train"]
    env=TradingEnv(episodes,normalizer_from(dataset["normalizer"]),dataset["names"],config,random_start=True,episode_steps=128 if smoke else 512)
    env.action_space.seed(seed);env.observation_space.seed(seed)
    gym_check(env,skip_render_check=True);sb3_check(env)
    model=create(algorithm,env,seed,device,smoke,overrides)
    steps=steps if steps is not None else 512 if algorithm=="ppo" else 1024
    if steps<=0:raise ValueError("positive training steps required")
    if algorithm=="ppo" and steps%model.n_steps:raise ValueError("steps must be an exact PPO rollout multiple")
    run=Path(output)/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+f"-{algorithm}-{seed}-"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    metadata={"algorithm":algorithm,"seed":seed,"device":device,"requested_steps":steps,"smoke":smoke,
        "python":sys.version.split()[0],
        "resolved_parameters":resolved_parameters(algorithm,smoke,overrides),"simulation":asdict(config),
        "dataset":Path(dataset_path).name,"git":git_state(),"source_hash":source_identity(),
        "selection_metric":"Equal-weight mean validation net return", "status":"running",
        "paper_eligible":False,"dependency_lock_present":(Path(__file__).resolve().parents[3]/"requirements.lock").exists(),
        "versions":versions,
        "nondeterminism":"Seeds fixed; hardware/backend floating point may differ", "resume":"Replay buffer saved; exact RNG/env restoration not claimed"}
    (run/"metadata.json").write_text(json.dumps(metadata,indent=2))
    start=time.monotonic();callback=ValidationCheckpoint(dataset,config,run,256 if smoke else 10000)
    try:
        model.learn(total_timesteps=steps,callback=callback)
        if not all(torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
            raise ValueError("nonfinite model weights")
        # The last on-step callback precedes PPO's rollout update; validate the
        # final updated weights even when the transition count matches.
        callback.validate()
        model.save(run/"latest")
        if algorithm!="ppo":model.save_replay_buffer(run/"replay.pkl")
        from rl_stocks.rl.registry.factory import CLASSES
        selected=CLASSES[algorithm].load(run/"best.zip",device=device)
        observation,_=env.reset(seed=seed)
        original=selected.predict(observation,deterministic=True)[0]
        metadata.update(status="complete",actual_steps=model.num_timesteps,duration_seconds=time.monotonic()-start,
            validation_checkpoints=callback.history,selected_score=callback.best)
        metadata.update(selected_step=callback.best_step,selected_metrics=callback.best_metrics)
        bundle=freeze(run/"best.zip",dataset,metadata,Path(output)/"frozen")
        frozen=FrozenPolicy(bundle,dataset["schema"],dataset["manifest"]["provenance"]["capabilities"]["feed"],device)
        np.testing.assert_array_equal(original,frozen.predict(observation))
        metadata["bundle_id"]=bundle.name
        if final_test:
            results=evaluate(selected,dataset,"test",config)
            report(results,run/"final-test.html","Held-out audit of frozen validation selection")
        (run/"metadata.json").write_text(json.dumps(metadata,indent=2))
        return run,bundle
    except BaseException as exc:
        metadata.update(status="failed",failure_type=type(exc).__name__,actual_steps=model.num_timesteps,duration_seconds=time.monotonic()-start)
        (run/"metadata.json").write_text(json.dumps(metadata,indent=2))
        model.save(run/"interrupted")
        raise
    finally:env.close()
