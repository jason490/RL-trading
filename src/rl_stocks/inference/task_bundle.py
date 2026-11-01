"""Checksummed task SAC inference with caller-independent instrument tagging."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from stable_baselines3 import SAC

from rl_stocks.core.contracts import ACTION_SCHEMA, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.inference.bundle import checksum, freeze
from rl_stocks.rl.sac.task_policy import TaskSACPolicy, check_role
from rl_stocks.risk.exposure import filter_exposure
from rl_stocks.training.task_conditioning import validate_type_manifest


def freeze_task(model_path, dataset, *, task_role, type_manifest, metadata, output):
    identity = validate_type_manifest(dataset, type_manifest)
    role = check_role(task_role)
    if metadata.get('algorithm') != 'sac':
        raise ValueError('task bundle requires SAC')
    metadata = deepcopy(metadata)
    metadata['task_contract'] = {'schema': 'task-inference-v1', 'role': role,
                                 'dataset_sha256': digest(dataset),
                                 'type_manifest_sha256': identity,
                                 'type_manifest': deepcopy(type_manifest),
                                 'base_observations': 21, 'type_encoding': {'stock': -1, 'etf': 1}}
    return freeze(model_path, dataset, metadata, output)


class FrozenTaskPolicy:
    """Research-only policy; input tags are generated from frozen reviewed records."""
    def __init__(self, path, *, feature_schema, feed, device='cpu'):
        self.path = Path(path)
        hashes = json.loads((self.path / 'checksums.json').read_text())
        if set(hashes) != {p.name for p in self.path.iterdir() if p.name != 'checksums.json'}:
            raise ValueError('task bundle file set mismatch')
        for name, value in hashes.items():
            if Path(name).name != name or checksum(self.path / name) != value:
                raise ValueError('task bundle checksum mismatch')
        manifest = json.loads((self.path / 'manifest.json').read_text())
        if digest(manifest) != self.path.name:
            raise ValueError('task bundle identity mismatch')
        if (manifest['feature_schema'] != feature_schema
                or manifest['action_schema'] != ACTION_SCHEMA
                or manifest['model_sha256'] != checksum(self.path / 'model.zip')
                or manifest['data_manifest']['provenance']['capabilities']['feed'] != feed
                or manifest['metadata']['algorithm'] != 'sac'):
            raise ValueError('task bundle schema/feed mismatch')
        contract = manifest['metadata']['task_contract']
        if (contract['schema'] != 'task-inference-v1' or contract['base_observations'] != 21
                or contract['type_encoding'] != {'stock': -1, 'etf': 1}):
            raise ValueError('task observation contract mismatch')
        self.task_role = check_role(contract['role'])
        evidence = contract['type_manifest']
        if (evidence['dataset'] != contract['dataset_sha256']
                or digest(evidence) != contract['type_manifest_sha256']
                or evidence['schema'] != 'historical-instrument-types-v1'
                or evidence['historical_evidence_verified'] is not True):
            raise ValueError('task type provenance mismatch')
        from rl_stocks.data.universe.type_contract import validate_scope, validate_source_point
        validate_scope(evidence)
        records = evidence['records']
        if (len(records) != len({r['symbol'] for r in records})
                or {r['symbol'] for r in records} != set(manifest['data_manifest']['universe'])):
            raise ValueError('task universe mismatch')
        registry = {}
        for record in records:
            validate_source_point(record)
            if (record['kind'] not in ('stock', 'etf') or not record['source']
                    or record['evidence_kind'] != 'historical_primary_source'):
                raise ValueError('task historical evidence required')
            registry[record['symbol']] = (record['kind'], date.fromisoformat(record['known_from']))
        self.registry = MappingProxyType(registry)
        self.normalizer = normalizer_from(manifest['normalizer'])
        if (len(self.normalizer.names) != 7
                or tuple(manifest['names']) != self.normalizer.names):
            raise ValueError('task seven-feature normalization required')
        from rl_stocks.environment.observations.state import observation_spec
        if manifest['metadata']['observation'] != observation_spec('execution_full'):
            raise ValueError('task full execution state required')
        from rl_stocks.risk.exposure import validate_execution_contract
        self.execution = validate_execution_contract(manifest['metadata']['action_execution'])
        self._model = SAC.load(self.path / 'model.zip', device=device)
        policy = self._model.policy
        if (type(policy) is not TaskSACPolicy or policy.task_role != self.task_role
                or self._model.observation_space.shape != (22,)
                or self._model.action_space.shape != (4,)
                or any(member.task_role != self.task_role for member in
                       (policy.actor, policy.critic, policy.critic_target))):
            raise ValueError('task loaded policy contract mismatch')
        self._model.policy.set_training_mode(False)
        self._manifest = deepcopy(manifest)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def predict(self, observation, *, symbol, session):
        if symbol not in self.registry:
            raise ValueError('unknown task instrument')
        kind, known = self.registry[symbol]
        if date.fromisoformat(session) < known:
            raise ValueError('instrument type unavailable at decision')
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (21,) or not np.isfinite(observation).all():
            raise ValueError('task prediction requires finite full21, without caller tag')
        tag = 0 if self.task_role == 'masked_shared' else (1 if kind == 'etf' else -1)
        vector = np.concatenate((observation, np.asarray([tag], dtype=np.float32)))
        with torch.no_grad():
            action = self._model.predict(vector, deterministic=True)[0]
        return filter_exposure(action, self.execution['evaluation_exposure_mode'])
