"""Read-only research forecasts; this bundle is not an RL trading policy."""
from dataclasses import asdict
import json
from pathlib import Path,PurePosixPath
import torch
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.training.supervised.models import network,design,predict,ridge_predict


class SignalBundle:
    KIND="signal-feasibility-v1"
    def __init__(self,path,names,source_dataset):
        self.path=Path(path);hashes=json.loads((self.path/'checksums.json').read_text())
        if set(hashes)!={p.name for p in self.path.iterdir() if p.is_file() and p.name!='checksums.json'}:
            raise ValueError('signal bundle file set mismatch')
        if any(PurePosixPath(n).name!=n or checksum(self.path/n)!=h for n,h in hashes.items()):raise ValueError('signal bundle checksum mismatch')
        m=json.loads((self.path/'manifest.json').read_text())
        if (digest(m)!=self.path.name or m.get('kind')!=self.KIND
            or list(names)!=m['names'] or source_dataset!=m['source_dataset']):raise ValueError('signal bundle input contract mismatch')
        expected={f'{arm}-{seed}' for arm in ('masked','full') for seed in (11,23,37)}
        if set(m['weights'])!=expected:raise ValueError('complete signal seed set required')
        if any(PurePosixPath(p).name!=p for p in m['weights'].values()):raise ValueError('invalid signal weight name')
        self.manifest=m;self.names=tuple(names);self.normalizers={a:normalizer_from(v) for a,v in m['normalizers'].items()};self.models={}
        if set(self.normalizers)!={'masked','full'} or any(n.names!=self.names for n in self.normalizers.values()):raise ValueError('signal normalizer feature order mismatch')
    def predict(self,features,arm,seed=11,kind='mlp'):
        if arm not in self.normalizers:raise ValueError('unknown signal arm')
        x=design(features,asdict(self.normalizers[arm]),arm)
        if kind=='constant':
            import numpy as np
            return np.full(len(x),self.manifest['constant'])
        if kind=='ridge':return ridge_predict(self.manifest['ridge'][arm],x)
        key=f'{arm}-{seed}'
        if kind!='mlp' or key not in self.manifest['weights']:raise ValueError('unknown predictor')
        if key not in self.models:
            model=network();model.load_state_dict(torch.load(self.path/self.manifest['weights'][key],map_location='cpu',weights_only=True));model.eval();self.models[key]=model
        return predict(self.models[key],x)
