"""Closed-form trace decomposition with explicit endings; no optimization."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from rl_stocks.reporting.gae_variance import aggregate, components, protocol, statistics, trace


def arrays():
    values = np.array([0.2, -0.1, 0.3, 0.4], dtype=np.float32).reshape(-1, 1)
    raw = np.array([0.01, 0.02, -0.03, 0.04], dtype=np.float32).reshape(-1, 1)
    timeout_values = np.array([0, 0.17, 0, 0], dtype=np.float32).reshape(-1, 1)
    result = dict(
        values=values, raw_rewards=raw, rewards=raw + np.float32(0.99) * timeout_values,
        timeout_values=timeout_values, episode_starts=np.array([1, 0, 1, 0], dtype=np.float32).reshape(-1, 1),
        step_dones=np.array([0, 1, 0, 0], dtype=bool).reshape(-1, 1),
        timeouts=np.array([0, 1, 0, 0], dtype=bool).reshape(-1, 1),
        last_values=np.array([0.07], dtype=np.float32), dones=np.array([False]),
    )
    return result


def test_segmented_timeout_and_rollout_endpoint_closed_form():
    a = arrays()
    before = deepcopy(a)
    raw, tail, rounding = components(a, 0.99)
    expected_raw = [float(a['raw_rewards'][0, 0]) + 0.99 * float(a['raw_rewards'][1, 0]), float(a['raw_rewards'][1, 0]), float(a['raw_rewards'][2, 0]) + 0.99 * float(a['raw_rewards'][3, 0]), float(a['raw_rewards'][3, 0])]
    expected_tail = [0.99**2 * float(a['timeout_values'][1, 0]), 0.99 * float(a['timeout_values'][1, 0]), 0.99**2 * float(a['last_values'][0]), 0.99 * float(a['last_values'][0])]
    np.testing.assert_allclose(raw, expected_raw, atol=1e-15, rtol=0)
    np.testing.assert_allclose(tail, expected_tail, atol=1e-15, rtol=0)
    advantage, returns = trace(a, 0.99, 1.0)
    np.testing.assert_allclose(raw + tail + rounding, returns, atol=1e-15, rtol=0)
    np.testing.assert_allclose(advantage, returns - a['values'][:, 0], atol=1e-15, rtol=0)
    assert np.max(np.abs(rounding)) > 0
    for k in a:
        np.testing.assert_array_equal(a[k], before[k])


def test_true_terminals_zero_tail_and_no_cross_boundary_reward():
    a = arrays()
    a['timeouts'][:] = False
    a['timeout_values'][:] = 0
    a['rewards'] = a['raw_rewards'].copy()
    a['dones'][0] = True
    a['step_dones'][-1] = True
    _, tail, rounding = components(a, 0.99)
    assert not np.any(tail) and not np.any(rounding)
    before = trace(a, 0.99, 1.0)[1]
    a['raw_rewards'][2:] += 10
    a['rewards'] = a['raw_rewards'].copy()
    np.testing.assert_array_equal(before[:2], trace(a, 0.99, 1.0)[1][:2])


def test_lambda_zero_is_one_step_td_and_one_is_mc():
    a = arrays()
    advantage, _ = trace(a, 0.99, 0)
    expected = []
    for i in range(4):
        next_value = float(a['last_values'][0]) if i == 3 else float(a['values'][i+1, 0])
        continuation = not a['dones'][0] if i == 3 else not a['episode_starts'][i+1, 0]
        expected.append(float(a['rewards'][i, 0]) + .99 * next_value * continuation - float(a['values'][i, 0]))
    np.testing.assert_allclose(advantage, expected, atol=1e-15, rtol=0)


def test_zero_variance_not_silently_replaced_or_dropped():
    a = arrays()
    for k in ['values','raw_rewards','rewards','timeout_values','last_values']:
        a[k][:] = 0
    row = statistics(a)
    assert row['advantage_variance_ratio'] is None
    assert row['return_variance_ratio'] is None
    out = aggregate([row, row])
    assert out['advantage_variance_ratio'] == dict(defined=0, zero_denominators=2, median=None, fraction_above_one=None)


def test_variance_and_block_summary_match_numpy():
    a = arrays()
    row = statistics(a)
    a95, r95 = trace(a, .99, .95)
    a1, r1 = trace(a, .99, 1)
    assert row['advantage_variance_ratio'] == np.var(a1)/np.var(a95)
    assert row['return_variance_ratio'] == np.var(r1)/np.var(r95)
    assert aggregate([row])['advantage_variance_ratio']['median'] == row['advantage_variance_ratio']
    assert row['maximum_endpoint_identity_error'] < 1e-15


def test_protocol_rejects_tampering_before_any_source_access(tmp_path):
    import json
    root = Path(__file__).parents[1]
    spec = protocol(root)
    spec['gamma'] = 0.9
    p = tmp_path/'configs/experiments/gae_variance_audit.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='fixed'):
        protocol(tmp_path)


def test_running_source_rejected_before_ledger_or_array_access(tmp_path):
    import json
    import shutil
    from rl_stocks.reporting.gae_variance import audit

    root = Path(__file__).parents[1]
    for name in ['configs/experiments/gae_variance_audit.json', 'results/gae-traces-2026-10-25/summary.json']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    state = tmp_path / '.runtime/gae-screen-state.json'
    state.parent.mkdir()
    state.write_text(json.dumps(dict(status='running', cleanup_verified=False)))
    with pytest.raises(ValueError, match='owned-cleaned'):
        audit(tmp_path)
