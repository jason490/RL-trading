"""Portable generation provenance and independent untrained-array mapping checks."""

import hashlib
from pathlib import Path
import shutil

import numpy as np

from rl_stocks.training.crossq_spec import ROOT


def snapshot_sources(sources, output, root=ROOT):
    """Validate every source before writing; preserve old code across registry edits."""
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError('snapshot output must stay in repository')
    validated = []
    for name, expected in sources.items():
        relative = Path(name)
        source = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not source.is_relative_to(root.resolve()):
            raise ValueError('repository-relative generation source required')
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('original SAC generation source changed before snapshot')
        destination = output / (name + '.txt')
        if destination.exists() or not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError('new contained generation snapshot required')
        validated.append((name, source, expected))
    result = {}
    for name, source, expected in validated:
        destination = output / (name + '.txt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError('generation source snapshot changed')
        result[name] = dict(file=str(destination.relative_to(root)), sha256=expected)
    return result


def verify_arrays(source, target, mapping):
    pairs = mapping['mapping']
    if len(pairs) != 10 or {p['branch'] for p in pairs} != {'actor', 'critic'}:
        raise ValueError('complete ten-module common Linear mapping required')
    used, source_used = set(), set()
    for pair in pairs:
        for suffix in ['weight', 'bias']:
            left = pair['branch'] + '.' + pair['source'] + '.' + suffix
            right = pair['branch'] + '.' + pair['target'] + '.' + suffix
            if right in used or left in source_used or left not in source or right not in target:
                raise ValueError('unique complete archived Linear mapping required')
            if source[left].dtype != target[right].dtype or not np.array_equal(source[left], target[right]):
                raise ValueError('archived source and copied Linear tensors differ')
            used.add(right)
            source_used.add(left)
    normalization = {}
    for name, value in target.items():
        if name in used:
            continue
        suffix = name.rsplit('.', 1)[-1]
        if suffix not in ['scale', 'bias', 'ra_mean', 'ra_var', 'steps']:
            raise ValueError('unexpected non-Linear CrossQ archive tensor')
        expected = 1 if suffix in ['scale', 'ra_var'] else 0
        if not np.isfinite(value).all() or not np.all(value == expected):
            raise ValueError('archived native normalization must be unfitted')
        prefix = name.rsplit('.', 1)[0]
        normalization.setdefault(prefix, set()).add(suffix)
    if len(normalization) != 9 or any(fields != {'scale', 'bias', 'ra_mean', 'ra_var', 'steps'} for fields in normalization.values()):
        raise ValueError('nine archived normalization counters required')
    return dict(common_archived_linear_tensors_exact=20, native_normalization_counters=9,
                native_normalization_zero_one_exact=True)
