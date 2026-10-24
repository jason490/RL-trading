"""Frozen SAC multi-step soft targets; no policy optimization or market claims."""
import hashlib
import math

import numpy as np


def soft_target(rewards, log_probabilities, *, gamma, alpha, terminated,
                bootstrap_q=0.0, bootstrap_log_probability=0.0):
    """Q(s0,a0) excludes entropy for a0; future actions and soft tail include it."""
    rewards = np.asarray(rewards, dtype=np.float64)
    logp = np.asarray(log_probabilities, dtype=np.float64)
    scalars = (gamma, alpha, bootstrap_q, bootstrap_log_probability)
    if (rewards.ndim != 1 or not len(rewards) or rewards.shape != logp.shape
            or not np.isfinite(rewards).all() or not np.isfinite(logp).all()
            or not all(math.isfinite(v) for v in scalars) or not 0 <= gamma < 1 or alpha <= 0
            or type(terminated) is not bool):
        raise ValueError('finite matched soft-target inputs and boolean termination required')
    powers = gamma ** np.arange(len(rewards), dtype=np.float64)
    reward = float(np.dot(powers, rewards))
    entropy = float(np.dot(powers[1:], -alpha * logp[1:]))
    continuation = 0.0 if terminated else float(gamma ** len(rewards))
    tail_q = continuation * bootstrap_q
    tail_entropy = -continuation * alpha * bootstrap_log_probability
    return dict(discounted_rewards=reward, future_action_entropy=entropy,
                discounted_bootstrap_q=tail_q, bootstrap_entropy=tail_entropy,
                bootstrap_discount=continuation, target=reward + entropy + tail_q + tail_entropy)


def fingerprint(value):
    """Hash nested tensor/array optimizer state deterministically without mutation."""
    h = hashlib.sha256()
    def visit(item):
        if hasattr(item, 'detach'):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            h.update(b'array' + str(item.dtype).encode() + repr(item.shape).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            h.update(b'dict')
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode() + str(len(item)).encode())
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float, bool, type(None))):
            h.update(type(item).__name__.encode() + repr(item).encode() + b'\0')
        else:
            raise ValueError('unsupported immutable-state fingerprint value')
    visit(value)
    return h.hexdigest()


def summarize_windows(records):
    """Conditional sampling summary; not a confidence interval for future markets."""
    gaps = np.asarray([r['mean_min_q_minus_soft_target'] for r in records], dtype=np.float64)
    if len(gaps) < 2 or not np.isfinite(gaps).all():
        raise ValueError('at least two finite window means required')
    mean = float(gaps.mean())
    se = float(gaps.std(ddof=1) / np.sqrt(len(gaps)))
    return dict(windows=len(gaps), mean_min_q_minus_soft_target=mean,
                conditional_window_sampling_se=se, descriptive_lower_two_se=mean - 2 * se,
                median_window_gap=float(np.median(gaps)), positive_window_fraction=float((gaps > 0).mean()))
