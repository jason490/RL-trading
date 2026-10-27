"""Bounded numerical-diagnostic input contract; no model construction or fitting."""
import json
from pathlib import Path

import numpy as np

from rl_stocks.core.contracts import digest, canonical
from rl_stocks.training.crossq_assets import initialization_spec
from rl_stocks.training.crossq_spec import parameters
from rl_stocks.evaluation.soft_consistency import fingerprint

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = 'aad6c99abe62bd7b9b2659b23b9763f9e4c25a70f8532d88f95d32eca6ecc9a2'
FIELDS = dict(observations=(1001, 1, 21), next_observations=(1001, 1, 21),
              actions=(1001, 1, 4), rewards=(1001, 1), dones=(1001, 1), timeouts=(1001, 1))


def protocol():
    value = json.loads((ROOT / 'configs/experiments/sac_first_update_diagnostic.json').read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed SAC first-update diagnostic protocol required')
    return value


def arm_parameters():
    explicit = json.loads(json.dumps(parameters('sac', False)))
    original = json.loads(json.dumps(explicit))
    original['policy_kwargs'] = dict(net_arch=[64, 64])
    return dict(original_defaults=original, explicit_serialized_defaults=explicit)


def validate_payload(value):
    spec = protocol()
    if (value.get('schema') != 'sac-first-update-input-v1'
            or value.get('protocol_sha256') != PROTOCOL_SHA256
            or type(value.get('rows')) is not int or value['rows'] != spec['replay_rows']
            or value.get('historical_metadata_sha256') != spec['historical_metadata_sha256']
            or value.get('diagnostic_checkpoint_sha256') != spec['screen_diagnostic_checkpoint_sha256']
            or value.get('common_prefix_bitwise_verified') is not True
            or set(value.get('arrays', {})) != set(FIELDS)):
        raise ValueError('fixed owned, matched-prefix SAC diagnostic inputs required')
    if (canonical(value.get('arm_parameters')) != canonical(arm_parameters())
            or canonical(value.get('shared_initialization')) != canonical(initialization_spec('sac', 11, False))
            or value.get('source_kind') != 'retained_training_replay_only_no_new_environment_transitions'):
        raise ValueError('fixed native-default representations and original untrained initializer required')
    arrays = {}
    for name, shape in FIELDS.items():
        row = value['arrays'][name]
        array = np.asarray(row['values'], dtype=np.float32)
        if (row.get('dtype') != 'float32' or array.shape != shape or not np.isfinite(array).all()
                or fingerprint(array) != row.get('sha256')):
            raise ValueError('finite exact filled-prefix diagnostic field required')
        arrays[name] = array
    if (np.abs(arrays['actions']).max() > 1
            or not np.isin(arrays['dones'], [0, 1]).all()
            or not np.isin(arrays['timeouts'], [0, 1]).all()
            or (arrays['timeouts'] > arrays['dones']).any()):
        raise ValueError('normalized actions and valid terminal/timeout masks required')
    observation = np.asarray(value['postupdate_observation'], dtype=np.float32)
    if (observation.shape != (21,) or not np.isfinite(observation).all()
            or fingerprint(observation) != value['postupdate_observation_sha256']):
        raise ValueError('exact same follow-up observation required')
    return arrays
