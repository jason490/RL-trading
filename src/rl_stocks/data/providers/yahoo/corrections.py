"""Explicit user-approved exceptions; never a general OHLC repair tolerance."""
from dataclasses import replace
import math
from rl_stocks.core.contracts import digest


SPY_HIGH_POLICY = {
    'id': 'yahoo-spy-2026-10-16-high-v1',
    'provider': 'yahoo', 'feed': 'yahoo',
    'adjustments': 'split_adjusted_with_dividends',
    'symbol': 'SPY', 'session': '2026-10-16',
    'operation': 'raise_high_to_open', 'maximum_raise': 0.011,
    'original_ohlc_sha256': '9f41490a4be83f93aca85048d7160943aa7c3b3528324cdf80f7e59c67e57c77',
    'authorization': 'User explicitly authorized correcting this insignificant bar error.',
}


def correct_spy_high(bar):
    policy = SPY_HIGH_POLICY
    if any(getattr(bar, key) != policy[key] for key in
           ('provider', 'feed', 'adjustments', 'symbol', 'session')) or bar.high >= bar.open:
        return bar, None
    original = {key: getattr(bar, key) for key in ('open', 'high', 'low', 'close')}
    if not all(math.isfinite(value) for value in original.values()):
        return bar, None  # The normal validator rejects missing required prices.
    if digest(original) != policy['original_ohlc_sha256'] or not 0 < bar.open-bar.high <= policy['maximum_raise']:
        raise ValueError('approved SPY correction does not match received OHLC')
    corrected = replace(bar, high=bar.open, revision=bar.revision+'|'+policy['id'])
    audit = {'policy': policy['id'], 'symbol': bar.symbol, 'session': bar.session,
             'original_high': bar.high, 'corrected_high': corrected.high,
             'delta': corrected.high-bar.high, 'original_ohlc_sha256': digest(original)}
    return corrected, audit
