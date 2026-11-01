"""Immutable deterministic task actor under the reference arithmetic contract.

This is a new execution definition, not an assertion of native Torch parity.
Loading and prediction require only the Python standard library.
"""
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, localcontext
import json
from pathlib import Path
import re

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_math import (
    CONTRACT, DOT_PRECISION, _context, bits32, certified_tanh, exact_dot,
    from_bits32, input32, round32,
)

SCHEMA = 'canonical-task-actor-v1'
ROLES = ('masked_shared', 'tagged_shared', 'tagged_routed')
MAX_BYTES = 512 * 1024
OUTPUT = 'sb3-box-minus-one-plus-one-binary32-v1'


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError('actor schema fields mismatch')


def _hex(value, length):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{%d}' % length, value) is None:
        raise ValueError('canonical hexadecimal value required')
    return value


def _decode(value):
    return Decimal.from_float(input32(from_bits32(int(_hex(value, 8), 16))))


@dataclass(frozen=True)
class _Layer:
    weights: tuple
    biases: tuple

    @classmethod
    def parse(cls, value, inputs, outputs):
        _fields(value, ('weights', 'biases'))
        rows, biases = value['weights'], value['biases']
        if (not isinstance(rows, list) or len(rows) != outputs
                or not isinstance(biases, list) or len(biases) != outputs
                or any(not isinstance(row, list) or len(row) != inputs for row in rows)):
            raise ValueError('actor layer geometry mismatch')
        return cls(tuple(tuple(_decode(w) for w in row) for row in rows),
                   tuple(_decode(b) for b in biases))

    def forward(self, values):
        with localcontext(_context(DOT_PRECISION)):
            inputs = tuple(Decimal.from_float(v) for v in values)
            return tuple(round32(sum((w * x for w, x in zip(row, inputs)), bias))
                         for row, bias in zip(self.weights, self.biases))


def _unscale(value):
    # Preserve each operation's binary32 rounding in SB3's Box unscale formula.
    value = exact_dot((1.,), (value,), 1.)
    value = exact_dot((.5,), (value,))
    value = exact_dot((2.,), (value,))
    return exact_dot((1.,), (value,), -1.)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate actor JSON field')
        result[key] = value
    return result


@dataclass(frozen=True, init=False)
class CanonicalActor:
    _encoded: str
    _role: str
    _layers: tuple
    _heads: tuple

    def __init__(self, envelope, *, expected_source=None, expected_role=None):
        envelope = deepcopy(envelope)
        _fields(envelope, ('actor', 'sha256'))
        body = envelope['actor']
        if digest(body) != _hex(envelope['sha256'], 64):
            raise ValueError('actor checksum mismatch')
        _fields(body, ('schema', 'arithmetic', 'role', 'input_size', 'hidden_sizes',
                       'output_size', 'source_model_sha256', 'output_transform', 'layers', 'heads'))
        source = _hex(body['source_model_sha256'], 64)
        role = body['role']
        if (body['schema'] != SCHEMA or body['arithmetic'] != CONTRACT or role not in ROLES
                or type(body['input_size']) is not int or body['input_size'] != 22
                or body['hidden_sizes'] != [64, 64]
                or any(type(n) is not int for n in body['hidden_sizes'])
                or body['output_size'] != 4
                or type(body['output_size']) is not int or body['output_transform'] != OUTPUT):
            raise ValueError('unsupported actor schema or arithmetic')
        if expected_source is not None and source != expected_source:
            raise ValueError('actor source identity mismatch')
        if expected_role is not None and role != expected_role:
            raise ValueError('actor role mismatch')
        layers = body['layers']
        if not isinstance(layers, list) or len(layers) != 2:
            raise ValueError('actor requires two layers')
        names = ('stock', 'etf') if role == 'tagged_routed' else ('shared',)
        _fields(body['heads'], names)
        object.__setattr__(self, '_layers', (_Layer.parse(layers[0], 22, 64),
                                           _Layer.parse(layers[1], 64, 64)))
        object.__setattr__(self, '_heads', tuple(_Layer.parse(body['heads'][k], 64, 4) for k in names))
        object.__setattr__(self, '_role', role)
        object.__setattr__(self, '_encoded', json.dumps(envelope, sort_keys=True, allow_nan=False))

    @classmethod
    def load(cls, path, **kwargs):
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('actor file exceeds size bound')
        return cls(json.loads(raw, object_pairs_hook=_unique), **kwargs)

    @property
    def manifest(self):
        return json.loads(self._encoded)

    def predict(self, observation):
        if len(observation) != 22:
            raise ValueError('actor requires 22 observation coordinates')
        values = tuple(input32(v) for v in observation)
        tag = values[-1]
        if tag not in ((0.,) if self._role == 'masked_shared' else (-1., 1.)):
            raise ValueError('invalid instrument type tag')
        for layer in self._layers:
            values = tuple(max(0., v) for v in layer.forward(values))
        head = self._heads[1 if self._role == 'tagged_routed' and tag == 1 else 0]
        return tuple(_unscale(certified_tanh(v)) for v in head.forward(values))


def encode32(value):
    return f'{bits32(input32(value)):08x}'
