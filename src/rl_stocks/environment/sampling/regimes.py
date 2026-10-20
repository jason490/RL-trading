"""Training-only start pools based on prices already observed at each start."""
import math

MODES = ('legacy_uniform', 'ready_uniform', 'downtrend_balanced', 'full_window_uniform', 'stock_etf_cycle_windows')


class CausalStarts:
    def __init__(self, episodes, mode, max_holding=20):
        if mode not in ('ready_uniform','downtrend_balanced'):
            raise ValueError('causal start-pool mode required')
        self.mode = mode
        self.labels, self.all, self.down, self.other, records = [], [], [], [], []
        for index, episode in enumerate(episodes):
            bars = episode['bars']
            if episode['split'] != 'train' or any(b['session'] > '2020-12-31' for b in bars):
                raise ValueError('causal sampling requires frozen training dates only')
            if any(b['adjustments'] != 'split_adjusted_with_dividends' or b.get('split', 1.) != 1 for b in bars):
                raise ValueError('causal sampling requires verified split-adjusted price inputs')
            prices = [b['close'] for b in bars]
            if any(not math.isfinite(p) or p <= 0 for p in prices):
                raise ValueError('positive finite sampling prices required')
            labels = tuple(i >= 252 and prices[i]/prices[i-252]-1 < -.2
                           and prices[i]/prices[i-63]-1 < 0 for i in range(len(prices)))
            self.labels.append(labels)
            eligible = [(index, i) for i in range(252, len(bars)-max_holding-1)]
            if not eligible:
                raise ValueError('every training episode needs a 252-session-ready start')
            self.all.extend(eligible)
            self.down.extend(pair for pair in eligible if labels[pair[1]])
            self.other.extend(pair for pair in eligible if not labels[pair[1]])
            records.append({'symbol': episode['symbol'], 'eligible': len(eligible),
                            'downtrend': sum(labels[i] for _, i in eligible)})
        if mode == 'downtrend_balanced' and (not self.down or not self.other):
            raise ValueError('balanced sampling requires both training classes')
        self.summary = {'mode': mode, 'long_window': 252, 'short_window': 63,
                        'long_price_return_below': -.2, 'short_price_return_below': 0.,
                        'eligible': len(self.all), 'downtrend': len(self.down),
                        'other': len(self.other), 'episodes': records,
                        'selection': 'global uniform eligible starts' if mode == 'ready_uniform'
                            else 'equal-probability class, then global uniform within class',
                        'labels_in_observation_or_reward': False}

    def choose(self, rng):
        pool = self.all if self.mode == 'ready_uniform' else self.down if rng.integers(2) else self.other
        return pool[int(rng.integers(len(pool)))]

    def label(self, episode, index):
        return self.labels[episode][index]
