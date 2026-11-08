"""Immutable21→64ReLU→64ReLU→4/tanh reference for standard native SAC.

Distinct schema from task-conditioned SAC and Gaussian PPO/TRPO. Reference
arithmetic is explicit; native Torch bit parity is not asserted.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import (
    _Layer, _fields, _hex, _unique, _unscale, OUTPUT, MAX_BYTES,
)
from rl_stocks.inference.canonical_math import CONTRACT, certified_tanh, input32

SCHEMA = "canonical-standard-sac-policy-v1"


@dataclass(frozen=True, init=False)
class CanonicalSACPolicy:
    _encoded: str
    _layers: tuple

    def __init__(self, envelope, *, expected_source=None):
        envelope = deepcopy(envelope)
        _fields(envelope, ("actor", "sha256"))
        body = envelope["actor"]
        if digest(body) != _hex(envelope["sha256"], 64):
            raise ValueError("standard SAC actor checksum mismatch")
        _fields(body, ("schema", "arithmetic", "algorithm", "input_size", "hidden_sizes",
                       "output_size", "output_transform", "source_model_sha256", "layers"))
        source = _hex(body["source_model_sha256"], 64)
        if (body["schema"] != SCHEMA or body["arithmetic"] != CONTRACT
                or body["algorithm"] != "sac" or type(body["input_size"]) is not int
                or body["input_size"] != 21 or body["hidden_sizes"] != [64, 64]
                or any(type(v) is not int for v in body["hidden_sizes"])
                or type(body["output_size"]) is not int or body["output_size"] != 4
                or body["output_transform"] != OUTPUT
                or not isinstance(body["layers"], list) or len(body["layers"]) != 3):
            raise ValueError("fixed standard21-field SAC reference contract required")
        if expected_source is not None and source != expected_source:
            raise ValueError("standard SAC actor source mismatch")
        layers = tuple(_Layer.parse(v, i, o) for v, i, o in zip(
            body["layers"], (21, 64, 64), (64, 64, 4), strict=True))
        object.__setattr__(self, "_layers", layers)
        object.__setattr__(self, "_encoded", json.dumps(envelope, sort_keys=True, allow_nan=False))

    @classmethod
    def load(cls, path, **kwargs):
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("bounded standard SAC reference actor required")
        return cls(json.loads(raw, object_pairs_hook=_unique), **kwargs)

    @property
    def manifest(self):
        return json.loads(self._encoded)

    def predict(self, observation):
        try:
            if len(observation) != 21 or any(isinstance(v, bool) for v in observation):
                raise ValueError("finite21-field standard SAC input required")
            values = tuple(input32(v) for v in observation)
        except (TypeError, OverflowError) as error:
            raise ValueError("finite21-field standard SAC input required") from error
        for layer in self._layers[:2]:
            values = tuple(max(0., v) for v in layer.forward(values))
        return tuple(_unscale(certified_tanh(v)) for v in self._layers[2].forward(values))
