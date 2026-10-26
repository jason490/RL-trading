"""Constant-space observed window/date evidence; no sampling or RNG draws."""

import hashlib

from rl_stocks.core.contracts import canonical


class SamplingPaths:
    def __init__(self, env):
        self.starts = self.transitions = 0
        self.windows = hashlib.sha256()
        self.dates = hashlib.sha256()
        reset, step = env.reset, env.step

        def observed_reset(*args, **kwargs):
            result = reset(*args, **kwargs)
            self.windows.update(canonical(dict(
                symbol=env.episode['symbol'], episode_index=env.episode_index,
                start_index=env.i, eligible_start_upper_exclusive=env.start_sampler.bounds[env.episode_index],
                first_session=env.bars[0].session, last_session=env.bars[-1].session,
                decision_session=env.bars[env.i].session,
                window_transitions=env.episode_steps,
            )) + b'\n')
            self.starts += 1
            return result

        def observed_step(action):
            before = dict(symbol=env.episode['symbol'], decision_index=env.i,
                          decision_session=env.bars[env.i].session)
            result = step(action)
            before['execution_session'] = env.bars[env.i].session
            self.dates.update(canonical(before) + b'\n')
            self.transitions += 1
            return result

        env.reset, env.step = observed_reset, observed_step

    def snapshot(self):
        return dict(version=1, measured=True, starts=self.starts, transitions=self.transitions,
                    window_sha256=self.windows.hexdigest(), date_sha256=self.dates.hexdigest())


def validate_paths(value, counts, steps):
    if (not isinstance(value, dict) or type(value.get('version')) is not int
        or value['version'] != 1 or value.get('measured') is not True
        or type(value.get('starts')) is not int or value['starts'] != counts['starts']
        or type(value.get('transitions')) is not int
        or value['transitions'] != steps or value['transitions'] != counts['transitions']):
        raise ValueError('actual Treasury window/date counts required')
    for field in ('window_sha256', 'date_sha256'):
        sha = value.get(field)
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('observed Treasury window/date hashes required')
    return value
