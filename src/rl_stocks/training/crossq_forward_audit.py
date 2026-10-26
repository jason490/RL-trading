"""Observe real native normalization forwards without drawing RNG or fitting."""

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

from sb3_contrib.common.torch_layers import BatchRenorm
import torch

from rl_stocks.rl.crossq.geometry import layers, model_spec
from rl_stocks.training.crossq_spec import expected_updates

SOURCE_SHA256 = 'f69021d9e5d1acbcfbe0ccabbf3a813e5dca22bda5dd9d1dae868d6d2cf93885'
_NATIVE_FORWARD = BatchRenorm.forward


class NormalizationAudit:
    def __init__(self, model):
        model_spec(model)
        if hashlib.sha256(Path(inspect.getfile(BatchRenorm)).read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError('review changed native normalization implementation before training')
        self.model, self.modules = model, layers(model)
        if model.num_timesteps or model._n_updates or any(int(m.steps) for m in self.modules.values()):
            raise ValueError('attach observed audit before any training')
        self.pending, self.handles, self.closed, self.failed = {}, [], False, False
        self.rows = {name:dict(training_calls=0, evaluation_calls=0, renorm_condition_calls=0, counter=0)
                     for name in self.modules}
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_pre_hook(lambda m, args, name=name: self._before(name, m)))
            self.handles.append(module.register_forward_hook(lambda m, args, output, name=name: self._after(name, m)))

    def _before(self, name, module):
        if name in self.pending or self.closed:
            self.failed = True
            raise ValueError('inactive or reentrant normalization audit')
        mode, before = bool(module.training), int(module.steps)
        saved = None if mode else {k:v.detach().clone() for k,v in module.state_dict().items()}
        self.pending[name] = (mode, before, saved)

    def _after(self, name, module):
        mode, before, saved = self.pending.pop(name)
        after = int(module.steps)
        if bool(module.training) != mode or after != before + int(mode):
            self.failed = True
            raise ValueError('native normalization forward counter/mode drift')
        if saved is not None and any(not torch.equal(v, saved[k]) for k,v in module.state_dict().items()):
            self.failed = True
            raise ValueError('evaluation forward changed normalization state')
        row = self.rows[name]
        row['training_calls' if mode else 'evaluation_calls'] += 1
        row['renorm_condition_calls'] += int(mode and before > module.warmup_steps)
        row['counter'] = after

    def snapshot(self):
        if self.failed or self.pending or any(layers(self.model).get(name) is not m for name,m in self.modules.items()):
            self.failed = True
            raise ValueError('unfinished or replaced normalization forward')
        for name, module in self.modules.items():
            if (getattr(module.forward, '__func__', None) is not _NATIVE_FORWARD
                or int(module.steps) != self.rows[name]['training_calls']):
                self.failed = True
                raise ValueError('unobserved normalization mutation or altered implementation')
        return dict(version=1, source_sha256=SOURCE_SHA256, observed=True,
                    warmup_steps=1000, layers=deepcopy(self.rows))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True


def validate_audit(value, steps, phase, smoke=False, normalization=None):
    expected = expected_updates('crossq', steps, phase, smoke)
    if (type(value.get('version')) is not int or value.get('version') != 1 or value.get('observed') is not True
        or value.get('source_sha256') != SOURCE_SHA256 or value.get('warmup_steps') != 1000
        or len(value.get('layers', {})) != 9
        or sum(n.startswith('actor.') for n in value['layers']) != 3
        or sum(n.startswith('critic.') for n in value['layers']) != 6):
        raise ValueError('complete pinned observed normalization evidence required')
    if normalization is not None and set(value['layers']) != set(normalization):
        raise ValueError('observed layer identities must match actual saved normalization')
    for name, row in value['layers'].items():
        if normalization is not None and normalization[name]['steps'] != row['counter']:
            raise ValueError('observed counters must match actual saved normalization')
        branch = 'actor' if name.startswith('actor.') else 'critic'
        if (set(row) != {'training_calls', 'evaluation_calls', 'renorm_condition_calls', 'counter'}
            or any(type(v) is not int or v < 0 for v in row.values())
            or row['training_calls'] != expected[branch + '_normalization']
            or row['counter'] != row['training_calls']
            or row['renorm_condition_calls'] != expected[branch + '_renorm_forward_calls']):
            raise ValueError('observed forwards/renorm condition must match actual update geometry')
    return value
