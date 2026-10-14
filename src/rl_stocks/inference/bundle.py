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
        self.normalizer=normalizer_from(self.manifest["normalizer"])
        algorithm=self.manifest["metadata"]["algorithm"]
        self._model=CLASSES[algorithm].load(self.path/"model.zip",device=device)
        self._model.policy.set_training_mode(False)
    def predict(self,observation):
        import numpy as np
        observation=np.asarray(observation,dtype=np.float32)
        if observation.shape!=(len(self.manifest["names"])+5,) or not np.isfinite(observation).all():
            raise ValueError("observation shape/nonfinite mismatch")
        return self._model.predict(observation,deterministic=True)[0]
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
