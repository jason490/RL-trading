from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from rl_stocks.core.contracts import ACTION_SCHEMA, canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from

def checksum(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(model_path,dataset,metadata,output):
    manifest={"feature_schema":dataset["schema"],"names":dataset["names"],
        "action_schema":ACTION_SCHEMA,"normalizer":dataset["normalizer"],
        "data_manifest":dataset["manifest"],"metadata":metadata,
        "model_sha256":checksum(model_path)}
    identity=digest(manifest)
    target=Path(output)/identity
    target.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(model_path,target/"model.zip")
    (target/"manifest.json").write_bytes(canonical(manifest))
    dependency_root=Path(__file__).resolve().parents[3]
    lock=dependency_root/"requirements.lock"
    if lock.exists():shutil.copyfile(lock,target/"requirements.lock")
    elif (dependency_root/"requirements.txt").exists():
        shutil.copyfile(dependency_root/"requirements.txt",target/"requirements.txt")
    else:
        (target/"dependencies.json").write_bytes(canonical(metadata.get("versions",{})))
    hashes={p.name:checksum(p) for p in target.iterdir()}
    (target/"checksums.json").write_bytes(canonical(hashes))
    return target

class FrozenPolicy:
    def __init__(self,path,feature_schema,feed,device="cpu"):
        from rl_stocks.rl.registry.factory import CLASSES
        self.path=Path(path)
        hashes=json.loads((self.path/"checksums.json").read_text())
        if set(hashes)!={p.name for p in self.path.iterdir() if p.name!="checksums.json"}:
            raise ValueError("bundle file set mismatch")
        for name,value in hashes.items():
            if Path(name).name!=name or checksum(self.path/name)!=value:
                raise ValueError("bundle checksum mismatch")
        self.manifest=json.loads((self.path/"manifest.json").read_text())
        if digest(self.manifest)!=self.path.name:raise ValueError("bundle identity mismatch")
        if self.manifest["feature_schema"]!=feature_schema or self.manifest["action_schema"]!=ACTION_SCHEMA:
            raise ValueError("bundle schema mismatch")
        expected=self.manifest["data_manifest"]["provenance"]["capabilities"]["feed"]
        if expected!=feed:raise ValueError("bundle feed mismatch")
        from rl_stocks.environment.observations.state import observation_spec
        self.observation_spec=self.manifest['metadata'].get('observation',observation_spec())
        if self.observation_spec!=observation_spec(self.observation_spec.get('profile')):
            raise ValueError('bundle observation schema mismatch')
        self.observation_profile=self.observation_spec['profile']
        self.observation_size=len(self.manifest['names'])+len(self.observation_spec['portfolio_fields'])
        self.normalizer=normalizer_from(self.manifest["normalizer"])
        from rl_stocks.risk.exposure import execution_contract,validate_execution_contract
        self.action_execution=validate_execution_contract(self.manifest['metadata'].get('action_execution',execution_contract()))
        self.exposure_execution=self.action_execution['evaluation_exposure_mode']
        algorithm=self.manifest["metadata"]["algorithm"]
        self._model=CLASSES[algorithm].load(self.path/"model.zip",device=device)
        if self._model.observation_space.shape!=(self.observation_size,):
            raise ValueError('model observation schema mismatch')
        from rl_stocks.rl.ppo.architecture import architecture_spec,validate_loaded_architecture
        self.policy_architecture=self.manifest['metadata'].get('policy_architecture',architecture_spec())
        validate_loaded_architecture(self._model,self.policy_architecture,len(self.manifest['names']))
        self._model.policy.set_training_mode(False)
    def _observation(self,observation):
        import numpy as np
        observation=np.asarray(observation,dtype=np.float32)
        if observation.shape!=(self.observation_size,) or not np.isfinite(observation).all():
            raise ValueError("observation shape/nonfinite mismatch")
        return observation
    def predict(self,observation):
        from rl_stocks.rl.registry.prediction import recurrent
        if recurrent(self._model):raise ValueError('recurrent policy requires explicit state and episode_start')
        from rl_stocks.risk.exposure import filter_exposure
        return filter_exposure(self._model.predict(self._observation(observation),deterministic=True)[0],self.exposure_execution)
    def predict_recurrent(self,observation,*,state,episode_start):
        """Return action and new memory; callers own a separate state per stream."""
        import numpy as np
        from rl_stocks.rl.registry.prediction import recurrent
        if not recurrent(self._model):raise ValueError('policy is not recurrent')
        if not isinstance(episode_start,(bool,np.bool_)):raise ValueError('explicit boolean episode_start required')
        if state is None and not episode_start:raise ValueError('initial prediction requires episode_start')
        if state is not None:
            lstm=self._model.policy.lstm_actor
            shape=(lstm.num_layers,1,lstm.hidden_size)
            if not isinstance(state,tuple) or len(state)!=2:raise ValueError('hidden/cell state pair required')
            state=tuple(np.array(s,dtype=np.float32,copy=True) for s in state)
            if any(s.shape!=shape or not np.isfinite(s).all() for s in state):raise ValueError('recurrent state shape/nonfinite mismatch')
        action,new_state=self._model.predict(self._observation(observation),state=state,
            episode_start=np.array([episode_start],dtype=bool),deterministic=True)
        from rl_stocks.risk.exposure import filter_exposure
        return filter_exposure(action,self.exposure_execution),new_state
    def decide(self,features,names,portfolio_state,symbol,decision_time,expiry_time,reference,atr,equity,quantity,
               available_time,now=None,max_age_seconds=86400):
        from rl_stocks.risk.decisions import decode
        from rl_stocks.data.validation.bars import timestamp
        from rl_stocks.core.contracts import utcnow
        if symbol not in self.manifest["data_manifest"]["universe"]:raise ValueError("unknown symbol")
        available,decision_at,expires,current=map(timestamp,(available_time,decision_time,expiry_time,now or utcnow()))
        if not available<=decision_at<=current<expires or (current-available).total_seconds()>max_age_seconds:
            raise ValueError("unavailable/stale/expired decision")
        observation=self.normalizer.transform(features,names)+list(portfolio_state)
        decision=decode(self.predict(observation),symbol,decision_time,expiry_time,reference,atr,equity,quantity)
        return {"bundle_id":self.path.name,**asdict(decision)}
