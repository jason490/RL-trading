"""Pure fixed-input checks, never an optimizer update or model construction."""
from copy import deepcopy

import numpy as np
import pytest

from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.training.crossq_assets import initialization_spec
from rl_stocks.training.sac_update_input import (
    FIELDS, PROTOCOL_SHA256, arm_parameters, protocol, validate_payload,
)


def fixture():
    spec = protocol()
    arrays = {k: np.zeros(shape, dtype=np.float32) for k, shape in FIELDS.items()}
    observation = np.zeros(21, dtype=np.float32)
    return dict(schema='sac-first-update-input-v1', protocol_sha256=PROTOCOL_SHA256, rows=1001,
        historical_metadata_sha256=spec['historical_metadata_sha256'],
        diagnostic_checkpoint_sha256=spec['screen_diagnostic_checkpoint_sha256'],
        common_prefix_bitwise_verified=True,
        arrays={k:dict(dtype='float32', values=v.tolist(), sha256=fingerprint(v)) for k,v in arrays.items()},
        postupdate_observation=observation.tolist(), postupdate_observation_sha256=fingerprint(observation),
        arm_parameters=arm_parameters(), shared_initialization=initialization_spec('sac', 11, False),
        source_kind='retained_training_replay_only_no_new_environment_transitions')


def test_fixed_input_shapes_and_no_grid_budget():
    value = fixture()
    result = validate_payload(value)
    assert {k:v.shape for k,v in result.items()} == FIELDS
    assert protocol()['maximum_total_model_gradient_updates'] == 2
    assert protocol()['maximum_total_adam_steps'] == 6
    assert protocol()['environment_transitions_collected'] == 0


@pytest.mark.parametrize('change', ['rows', 'boolean_rows', 'hash', 'shape', 'nan', 'mask', 'actions', 'params', 'initializer', 'source'])
def test_input_mutations_reject(change):
    value = deepcopy(fixture())
    if change == 'rows':
        value['rows'] = 1002
    elif change == 'boolean_rows':
        value['rows'] = True
    elif change == 'hash':
        value['arrays']['actions']['sha256'] = '0' * 64
    elif change == 'shape':
        value['postupdate_observation'].append(0.)
    elif change == 'nan':
        value['arrays']['rewards']['values'][0][0] = float('nan')
    elif change in ('mask', 'actions'):
        key = 'timeouts' if change == 'mask' else 'actions'
        array = np.asarray(value['arrays'][key]['values'], dtype=np.float32)
        array[0] = 1 if change == 'mask' else 2
        value['arrays'][key].update(values=array.tolist(), sha256=fingerprint(array))
    elif change == 'params':
        value['arm_parameters']['explicit_serialized_defaults']['learning_rate'] = 0.001
    elif change == 'initializer':
        value['shared_initialization']['seed'] = 23
    else:
        value['source_kind'] = 'validation'
    with pytest.raises(ValueError):
        validate_payload(value)
