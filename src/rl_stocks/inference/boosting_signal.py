"""Forecast-only residual boosting with frozen fit-fold preprocessing."""

from dataclasses import asdict
from rl_stocks.inference.signal import SignalBundle
from rl_stocks.inference.boosting_arrays import load_boosting, predict_boosting
from rl_stocks.training.supervised.models import design


class BoostingSignalBundle(SignalBundle):
    KIND = "ridge-residual-boosting-v1"

    def predict(self, features, arm, seed=11, kind="boosting"):
        if kind in ("constant", "ridge"):
            return super().predict(features, arm, seed, kind)
        key = f"{arm}-{seed}"
        if kind != "boosting" or key not in self.manifest["weights"] or arm not in self.normalizers:
            raise ValueError("unknown residual-boosting predictor")
        x = design(features, asdict(self.normalizers[arm]), arm)
        if key not in self.models:
            self.models[key] = load_boosting(self.path / self.manifest["weights"][key])
        return predict_boosting(self.models[key], x)
