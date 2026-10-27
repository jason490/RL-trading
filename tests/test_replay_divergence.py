import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('diagnosis', Path(__file__).parents[1] / 'scripts/diagnose_crossq_reproduction.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_prefix_and_four_ulp_action_difference():
    left = np.full((5, 1, 4), -0.19503617, dtype=np.float32)
    right = left.copy()
    for _ in range(4):
        right[2, 0, 2] = np.nextafter(right[2, 0, 2], np.float32(np.inf))
    result = module.field_difference(left, right)
    assert result['first_different_zero_based_row'] == 2
    assert result['common_leading_rows'] == 2 and result['differing_rows'] == 1
    assert result['first_row_maximum_float32_ulp'] == 4
    assert result['first_row_different_components'] == [2]
    assert module.field_difference(left, left)['common_leading_rows'] == 5


@pytest.mark.parametrize('change', ['shape', 'dtype', 'nan', 'empty'])
def test_invalid_prefixes_reject(change):
    a = np.ones((3, 1, 4), dtype=np.float32)
    b = a.copy()
    if change == 'shape':
        b = b[:2]
    elif change == 'dtype':
        b = b.astype(np.float64)
    elif change == 'nan':
        b[0] = np.nan
    else:
        a, b = a[:0], b[:0]
    with pytest.raises(ValueError):
        module.field_difference(a, b)


def optimizer_archive(betas, weight_decay):
    import io
    import zipfile
    import torch

    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w') as archive:
        for name in ('actor.optimizer.pth', 'critic.optimizer.pth', 'ent_coef_optimizer.pth'):
            value = io.BytesIO()
            torch.save(dict(param_groups=[dict(betas=betas, weight_decay=weight_decay, lr=0.0003, params=[0])]), value)
            archive.writestr(name, value.getvalue())
    return content.getvalue()


def test_effective_optimizer_values_separate_from_container_representation():
    original = optimizer_archive((0.9, 0.999), 0)
    current = optimizer_archive([0.9, 0.999], 0.0)
    proof = module.optimizer_group_comparison(original, current)
    assert all(p['effective_json_values_equal'] for p in proof.values())
    assert not any(p['serialized_python_container_equality'] for p in proof.values())
    changed = module.optimizer_group_comparison(original, optimizer_archive([0.8, 0.999], 0.0))
    assert not any(p['effective_json_values_equal'] for p in changed.values())
    with pytest.raises(ValueError):
        module.optimizer_group_comparison(original, optimizer_archive([0.9, 0.999], False))
