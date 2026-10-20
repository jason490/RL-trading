"""Immutable forecast-only tree bundles; no pickle or trading actions."""
from dataclasses import asdict
from rl_stocks.inference.signal import SignalBundle
from rl_stocks.inference.forest_arrays import load_forest,predict_forest
from rl_stocks.training.supervised.models import design


class ForestSignalBundle(SignalBundle):
    KIND='forest-signal-feasibility-v1'
    def predict(self,features,arm,seed=11,kind='forest'):
        if kind in ('constant','ridge'):return super().predict(features,arm,seed,kind)
        key=f'{arm}-{seed}'
        if kind!='forest' or key not in self.manifest['weights'] or arm not in self.normalizers:raise ValueError('unknown forest predictor')
        x=design(features,asdict(self.normalizers[arm]),arm)
        if key not in self.models:self.models[key]=load_forest(self.path/self.manifest['weights'][key],128)
        return predict_forest(self.models[key],x)
