"""Predeclared relative gate for the action-initialization screen."""
from statistics import mean


def screen_gate(cells, protocol):
    """Cells are one matched seed with selected/final group metrics, not tests."""
    if set(cells) != set(protocol['arms']):
        raise ValueError('complete four-arm screen required')
    control = cells['standard']
    gate = protocol['screen_gate']
    comparisons = {}
    eligible = []
    for arm in protocol['arms'][1:]:
        cell = cells[arm]
        selected_gain = 100 * (cell['selected']['score'] - control['selected']['score'])
        final_gain = 100 * (cell['final']['score'] - control['final']['score'])
        group_gains = {group: 100 * (cell['final']['groups'][group]['net_return']
                                    - control['final']['groups'][group]['net_return'])
                       for group in ('stock', 'etf')}
        worsening = 100 * (control['final']['mean_max_drawdown'] - cell['final']['mean_max_drawdown'])
        passed = (selected_gain >= gate['minimum_selected_gain_pp']
                  and final_gain >= gate['minimum_final_gain_pp']
                  and min(group_gains.values()) >= gate['minimum_group_final_gain_pp']
                  and worsening <= gate['maximum_final_drawdown_worsening_pp'])
        comparisons[arm] = {'selected_gain_pp': selected_gain, 'final_gain_pp': final_gain,
                            'group_final_gain_pp': group_gains,
                            'final_drawdown_worsening_pp': worsening, 'passed': passed}
        if passed:
            eligible.append(arm)
    # Python's stable max preserves the protocol's declared arm order on ties.
    promoted = max(eligible, key=lambda a: cells[a]['final']['score']) if eligible else None
    return {'passed': promoted is not None, 'promoted_candidate': promoted, 'comparisons': comparisons}


def group_metrics(results, stocks):
    groups = {}
    for kind in ('stock', 'etf'):
        rows = [r for r in results.values() if (r['symbol'] in stocks) == (kind == 'stock')]
        if not rows:
            raise ValueError('both stocks and ETFs required')
        excess = sorted(r['metrics']['net_return'] - r['baselines']['buy_hold']['metrics']['net_return'] for r in rows)
        n = len(excess)
        groups[kind] = {'count': n, 'net_return': mean(r['metrics']['net_return'] for r in rows),
                        'buy_hold_return': mean(r['baselines']['buy_hold']['metrics']['net_return'] for r in rows),
                        'excess_return': mean(excess),
                        'median_excess_return': (excess[(n-1)//2] + excess[n//2]) / 2,
                        'max_drawdown': mean(r['metrics']['max_drawdown'] for r in rows)}
    return {'score': mean(r['metrics']['net_return'] for r in results.values()),
            'mean_max_drawdown': mean(r['metrics']['max_drawdown'] for r in results.values()),
            'groups': groups}
