"""Fixed synthetic cross-machine reference actor probe; no market outcomes."""
import json
import platform
import sys
from time import perf_counter

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import CanonicalActor, ROLES
from rl_stocks.inference.canonical_math import CONTRACT, bits32

SCHEMA = 'canonical-actor-probe-v1'
COUNT = 128


def execute(actors):
    if set(actors) != set(ROLES):
        raise ValueError('exactly three declared task roles required')
    results = {}
    for role in ROLES:
        actor = CanonicalActor(actors[role], expected_role=role)
        tags = (0.,) if role == 'masked_shared' else (-1., 1.)
        vectors = [tuple(((i+j) % 17-8)/16 for j in range(21)) + (tag,)
                   for tag in tags for i in range(COUNT)]
        for vector in vectors[:4]:
            actor.predict(vector)
        start = perf_counter()
        outputs = [tuple(bits32(x) for x in actor.predict(vector)) for vector in vectors]
        duration = perf_counter() - start
        results[role] = {'actor_sha256': actors[role]['sha256'], 'count': len(vectors),
                         'inputs_sha256': digest(vectors), 'output_bits': outputs,
                         'output_bits_sha256': digest(outputs), 'elapsed_seconds': duration}
    return {'schema': SCHEMA, 'arithmetic': CONTRACT, 'python': platform.python_version(),
            'python_major_minor': list(sys.version_info[:2]), 'platform': platform.platform(),
            'native_packages_imported': sorted(set(sys.modules) & {'torch', 'numpy'}),
            'roles': results, 'training_performed': False, 'market_evaluated': False}


def verify(actual, expected):
    if (actual.get('schema') != SCHEMA or actual.get('arithmetic') != CONTRACT
            or actual.get('python_major_minor') != [3, 13]
            or actual.get('native_packages_imported') != []
            or actual.get('training_performed') is not False
            or actual.get('market_evaluated') is not False
            or set(actual.get('roles', {})) != set(ROLES)):
        raise ValueError('reference probe runtime/schema mismatch')
    for role in ROLES:
        value, reference = actual['roles'][role], expected['roles'][role]
        for key in ('actor_sha256', 'count', 'inputs_sha256', 'output_bits', 'output_bits_sha256'):
            # JSON conversion treats tuple/list as the same portable representation.
            if json.dumps(value[key], sort_keys=True) != json.dumps(reference[key], sort_keys=True):
                raise ValueError('reference probe exact prediction mismatch')
        if digest(value['output_bits']) != value['output_bits_sha256']:
            raise ValueError('reference probe output checksum mismatch')
        if not isinstance(value['elapsed_seconds'], (int, float)) or not 0 < value['elapsed_seconds'] < 120:
            raise ValueError('reference probe latency bound exceeded')
    return True
