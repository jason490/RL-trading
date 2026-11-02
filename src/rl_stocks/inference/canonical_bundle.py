"""Checksummed reference actors bound to original frozen research metadata.

The source native bundle remains retained separately. This format defines a new
execution path and never certifies the source native evaluation as accepted.
"""
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import MappingProxyType

from rl_stocks.core.contracts import ACTION_SCHEMA, canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.data.universe.type_contract import validate_scope, validate_source_point
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.inference.canonical_actor import CanonicalActor, ROLES
from rl_stocks.inference.canonical_math import CONTRACT, input32
from rl_stocks.risk.exposure import filter_exposure, validate_execution_contract

SCHEMA = 'canonical-task-bundle-v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked_files(path, expected=None):
    hashes = json.loads((path / 'checksums.json').read_text())
    names = {p.name for p in path.iterdir()} - {'checksums.json'}
    if (not isinstance(hashes, dict) or set(hashes) != names
            or expected is not None and names != set(expected)):
        raise ValueError('reference bundle file set mismatch')
    for name, sha in hashes.items():
        file = path / name
        if (Path(name).name != name or file.is_symlink() or not file.is_file()
                or _sha(file) != sha):
            raise ValueError('reference bundle checksum mismatch')
    return hashes


def _metadata(manifest, *, feature_schema, feed):
    if (manifest['feature_schema'] != feature_schema or manifest['action_schema'] != ACTION_SCHEMA
            or manifest['data_manifest']['provenance']['capabilities']['feed'] != feed
            or manifest['metadata']['algorithm'] != 'sac'):
        raise ValueError('reference bundle schema/feed mismatch')
    contract = manifest['metadata']['task_contract']
    if (contract['schema'] != 'task-inference-v1' or contract['base_observations'] != 21
            or contract['type_encoding'] != {'stock': -1, 'etf': 1}
            or contract['role'] not in ROLES):
        raise ValueError('reference task contract mismatch')
    evidence = contract['type_manifest']
    if (evidence['dataset'] != contract['dataset_sha256']
            or digest(evidence) != contract['type_manifest_sha256']
            or evidence['schema'] != 'historical-instrument-types-v1'
            or evidence['historical_evidence_verified'] is not True):
        raise ValueError('reference type provenance mismatch')
    validate_scope(evidence)
    records = evidence['records']
    if (len(records) != len({r['symbol'] for r in records})
            or {r['symbol'] for r in records} != set(manifest['data_manifest']['universe'])):
        raise ValueError('reference universe mismatch')
    registry = {r['symbol']: (r['kind'], validate_source_point(r)) for r in records}
    normalizer = normalizer_from(manifest['normalizer'])
    if (len(normalizer.names) != 7 or tuple(manifest['names']) != normalizer.names
            or manifest['metadata']['observation'] != observation_spec('execution_full')):
        raise ValueError('reference full21 preprocessing required')
    execution = validate_execution_contract(manifest['metadata']['action_execution'])
    return contract['role'], MappingProxyType(registry), normalizer, deepcopy(execution)


def freeze_canonical_task(source, output):
    """Derive a new bundle from a trusted native bundle without modifying it."""
    from rl_stocks.inference.canonical_export import export_actor
    source, output = Path(source), Path(output)
    source_hashes = _checked_files(source)
    original = json.loads((source / 'manifest.json').read_text())
    if digest(original) != source.name or original['model_sha256'] != _sha(source / 'model.zip'):
        raise ValueError('source native bundle identity mismatch')
    _metadata(original, feature_schema=original['feature_schema'],
              feed=original['data_manifest']['provenance']['capabilities']['feed'])
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.reference-', dir=output) as temporary:
        temporary = Path(temporary)
        export_actor(source / 'model.zip', temporary / 'actor.json')
        manifest = {'schema': SCHEMA, 'arithmetic': CONTRACT, 'source_bundle_id': source.name,
                    'source_checksums': source_hashes, 'source_manifest': original,
                    'actor_sha256': _sha(temporary / 'actor.json')}
        CanonicalActor.load(temporary / 'actor.json', expected_source=original['model_sha256'],
                            expected_role=original['metadata']['task_contract']['role'])
        if _checked_files(source) != source_hashes:
            raise ValueError('native bundle changed during reference export')
        (temporary / 'manifest.json').write_bytes(canonical(manifest))
        hashes = {name: _sha(temporary / name) for name in ('actor.json', 'manifest.json')}
        (temporary / 'checksums.json').write_bytes(canonical(hashes))
        target = output / digest(manifest)
        # Fresh destination; do not merge, overwrite or remove an existing bundle.
        target.mkdir(exist_ok=False)
        try:
            for name in ('actor.json', 'manifest.json', 'checksums.json'):
                shutil.copyfile(temporary / name, target / name)
        except BaseException:
            shutil.rmtree(target)
            raise
    return target


class FrozenCanonicalTaskPolicy:
    """Research inference; preprocessing and causal type provenance stay frozen."""
    def __init__(self, path, *, feature_schema, feed, expected_bundle_id=None):
        self.path = Path(path)
        _checked_files(self.path, ('actor.json', 'manifest.json'))
        manifest = json.loads((self.path / 'manifest.json').read_text())
        if (set(manifest) != {'schema', 'arithmetic', 'source_bundle_id', 'source_checksums',
                             'source_manifest', 'actor_sha256'}
                or manifest['schema'] != SCHEMA or manifest['arithmetic'] != CONTRACT
                or digest(manifest) != self.path.name
                or expected_bundle_id is not None and self.path.name != expected_bundle_id):
            raise ValueError('reference bundle identity/contract mismatch')
        original = manifest['source_manifest']
        if (digest(original) != manifest['source_bundle_id']
                or manifest['source_checksums'].get('manifest.json') != digest(original)
                or manifest['source_checksums'].get('model.zip') != original['model_sha256']
                or manifest['actor_sha256'] != _sha(self.path / 'actor.json')):
            raise ValueError('reference source identity mismatch')
        self.task_role, self.registry, self.normalizer, execution = _metadata(
            original, feature_schema=feature_schema, feed=feed)
        self._execution = execution
        self._actor = CanonicalActor.load(self.path / 'actor.json',
                                         expected_source=original['model_sha256'],
                                         expected_role=self.task_role)
        self._manifest = deepcopy(manifest)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    @property
    def execution(self):
        return deepcopy(self._execution)

    def raw_action(self, observation, *, symbol, session):
        if symbol not in self.registry:
            raise ValueError('unknown reference instrument')
        kind, known = self.registry[symbol]
        if date.fromisoformat(session) < known:
            raise ValueError('instrument type unavailable at decision')
        if len(observation) != 21:
            raise ValueError('reference prediction requires full21 without caller tag')
        values = tuple(input32(v) for v in observation)
        tag = 0. if self.task_role == 'masked_shared' else (1. if kind == 'etf' else -1.)
        return self._actor.predict(values + (tag,))

    def predict(self, observation, *, symbol, session):
        return filter_exposure(self.raw_action(observation, symbol=symbol, session=session),
                               self._execution['evaluation_exposure_mode'])
