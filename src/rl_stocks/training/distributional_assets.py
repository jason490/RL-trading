"""Pinned untrained critic-study assets and native SAC global RNG contract."""
import hashlib
import json
from pathlib import Path

from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.training.distributional_critics import PROTOCOL_SHA256, ROOT, parsed_global_rng, protocol, validate_initial_comparison
from rl_stocks.training.shared_initialization import read_initialization

BINDING_SHA256 = '09e1de58e99689d3807c05e8ba47504afb9aa323c224f69e4a3c1051e2a790f0'


def _path(name):
    path = Path(name)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(ROOT):
        raise ValueError('repository-relative initialization asset path required')
    return resolved


def manifest():
    path = ROOT / 'configs/experiments/distributional_initialization.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BINDING_SHA256:
        raise ValueError('fixed critic initialization binding changed')
    binding = json.loads(raw)
    if (binding['protocol_sha256'] != PROTOCOL_SHA256
            or binding['generated_without_fitting'] is not True or binding['market_data_access'] is not False
            or binding['artifacts'] != 12 or binding['observation_size'] != 21):
        raise ValueError('fixed critic initialization binding required')
    raw = _path(binding['manifest']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['manifest_sha256']:
        raise ValueError('critic initialization manifest checksum mismatch')
    result = json.loads(raw)
    groups = {'smoke-11', 'screen-11', 'screen-23', 'screen-37'}
    expected = {arm + '-' + group for arm in protocol()['arms'] for group in groups}
    if (result['schema'] != 'distributional-critic-initializations-v1'
            or result['protocol_sha256'] != PROTOCOL_SHA256 or result['generated_without_fitting'] is not True
            or result['no_dataset_or_provider_access'] is not True or result['optimizer_steps'] != 0
            or set(result['artifacts']) != expected or set(result['groups']) != groups):
        raise ValueError('complete untrained critic initialization manifest required')
    for key, group in result['groups'].items():
        expected_smoke = key.startswith('smoke-')
        if (group['smoke'] is not expected_smoke or group['native_standard_initialization_unchanged'] is not True
                or group['comparison'] != validate_initial_comparison(group['actual_initial_records'], expected_smoke)):
            raise ValueError('actual generated initialization comparison changed')
    return result


def initialization_entry(arm, seed=11, smoke=False):
    if (arm not in protocol()['arms'] or type(seed) is not int or seed not in (11, 23, 37)
            or type(smoke) is not bool or smoke and seed != 11):
        raise ValueError('fixed critic initialization arm/seed/smoke required')
    result = manifest()
    key = ('smoke' if smoke else 'screen') + '-' + str(seed)
    entry, group = result['artifacts'][arm + '-' + key], result['groups'][key]
    if entry['arm'] != arm or entry['seed'] != seed or entry['smoke'] is not smoke:
        raise ValueError('critic initialization entry identity mismatch')
    _, rng = read_initialization(_path(entry['file']).read_bytes(), entry)
    numpy_state, python_state = parsed_global_rng(group['global_rng'])
    expected = dict(torch=hashlib.sha256(rng.tobytes()).hexdigest(),
                    numpy=fingerprint(numpy_state), python=fingerprint(python_state))
    if expected != entry['global_rng_sha256']:
        raise ValueError('pinned native SAC global sampling RNG mismatch')
    return entry, group['global_rng']


def initialization_spec(arm, seed=11, smoke=False):
    entry, _ = initialization_entry(arm, seed, smoke)
    return dict(version=1, artifact_sha256=entry['sha256'], policy_tensor_sha256=entry['policy_tensor_sha256'],
                rng_sha256=entry['rng_sha256'], torch_version=entry['torch_version'], generated_without_fitting=True,
                arm=arm, seed=seed, smoke=smoke, protocol_sha256=PROTOCOL_SHA256,
                global_rng_sha256=entry['global_rng_sha256'])
