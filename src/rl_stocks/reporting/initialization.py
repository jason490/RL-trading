"""Predeclared relative gate for the action-initialization screen."""
from statistics import mean, median


def screen_gate(cells, protocol):
    """Cells are one matched seed with selected/final group metrics, not tests."""
    if set(cells) != set(protocol['arms']):
        raise ValueError('complete four-arm screen required')
    reference=protocol.get('reference_arm','standard')
    control = cells[reference]
    gate = protocol['screen_gate']
    comparisons = {}
    eligible = []
    for arm in (a for a in protocol['arms'] if a!=reference):
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
    if len({r['symbol'] for r in results.values()})!=len(results):
        raise ValueError('one episode per symbol required for this matched study')
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
                        'buy_hold_max_drawdown': mean(r['baselines']['buy_hold']['metrics']['max_drawdown'] for r in rows),
                        'max_drawdown': mean(r['metrics']['max_drawdown'] for r in rows)}
    return {'score': mean(r['metrics']['net_return'] for r in results.values()),
            'mean_max_drawdown': mean(r['metrics']['max_drawdown'] for r in results.values()),
            'groups': groups,
            'stock_excess_by_symbol': {r['symbol']:r['metrics']['net_return']-
                r['baselines']['buy_hold']['metrics']['net_return'] for r in results.values() if r['symbol'] in stocks}}


def confirmation_gate(cells, candidate, protocol):
    seeds=tuple(str(s) for s in protocol['confirmation_seeds'])
    reference=protocol.get('reference_arm','standard')
    if candidate not in protocol['arms'] or candidate==reference or set(cells)!={reference,candidate} or any(
            set(cells[arm])!=set(seeds) for arm in cells):
        raise ValueError('matched standard/candidate cells for all declared seeds required')
    control=cells[reference];chosen=cells[candidate];gate=protocol['confirmation_gate']
    gains={phase:100*mean(chosen[s][phase]['score']-control[s][phase]['score'] for s in seeds)
           for phase in ('selected','final')}
    paired=sum(chosen[s]['final']['score']>control[s]['final']['score'] for s in seeds)
    worsening=100*mean(control[s]['final']['mean_max_drawdown']-chosen[s]['final']['mean_max_drawdown'] for s in seeds)
    group_gains={g:100*mean(chosen[s]['final']['groups'][g]['net_return']-
                            control[s]['final']['groups'][g]['net_return'] for s in seeds) for g in ('stock','etf')}
    relative=(gains['selected']>=gate['minimum_mean_selected_gain_pp']
              and gains['final']>=gate['minimum_mean_final_gain_pp']
              and paired>=gate['minimum_paired_positive_final_seeds']
              and worsening<=gate['maximum_mean_final_drawdown_worsening_pp']
              and min(group_gains.values())>=protocol['screen_gate']['minimum_group_final_gain_pp'])
    targets={}
    for name,threshold_key in [('comparable','target_comparable_excess_pp_each_stock_and_etf_group'),
                              ('outperforming','target_outperforming_excess_pp_each_stock_and_etf_group')]:
        threshold=gate[threshold_key]/100
        phases={}
        for phase in ('selected','final'):
            symbols=set(chosen[seeds[0]][phase]['stock_excess_by_symbol'])
            if any(set(chosen[s][phase]['stock_excess_by_symbol'])!=symbols for s in seeds):
                raise ValueError('matched stock panels required')
            stock_median=median(mean(chosen[s][phase]['stock_excess_by_symbol'][symbol] for s in seeds) for symbol in symbols)
            excess={g:mean(chosen[s][phase]['groups'][g]['excess_return'] for s in seeds) for g in ('stock','etf')}
            drawdown={g:100*mean(chosen[s][phase]['groups'][g]['buy_hold_max_drawdown']-
                                chosen[s][phase]['groups'][g]['max_drawdown'] for s in seeds) for g in ('stock','etf')}
            phases[phase]={'group_excess_pp':{g:100*v for g,v in excess.items()},
                           'median_stock_excess_pp':100*stock_median,'group_drawdown_worsening_vs_hold_pp':drawdown,
                           'passed':min(excess.values())>=threshold and 100*stock_median>=gate['minimum_median_stock_excess_pp']
                                    and max(drawdown.values())<=gate['maximum_mean_final_drawdown_worsening_pp']}
        passing_seeds=[]
        for seed in seeds:
            if all(min(chosen[seed][p]['groups'][g]['excess_return'] for g in ('stock','etf'))>=threshold
                   and chosen[seed][p]['groups']['stock']['median_excess_return']*100>=gate['minimum_median_stock_excess_pp']
                   and max(100*(chosen[seed][p]['groups'][g]['buy_hold_max_drawdown']-
                                chosen[seed][p]['groups'][g]['max_drawdown']) for g in ('stock','etf'))
                       <=gate['maximum_mean_final_drawdown_worsening_pp'] for p in ('selected','final')):
                passing_seeds.append(int(seed))
        targets[name]={'phases':phases,'passing_seeds':passing_seeds,
                       'passed':all(p['passed'] for p in phases.values()) and len(passing_seeds)>=gate['minimum_target_passing_seeds']}
    return {'relative_improvement_passed':relative,'mean_selected_gain_pp':gains['selected'],
            'mean_final_gain_pp':gains['final'],'paired_positive_final_seeds':paired,
            'mean_final_drawdown_worsening_pp':worsening,'mean_group_final_gain_pp':group_gains,
            'targets':targets,'objective_achieved':False,
            'interpretation':'Development confirmation only; independent prospective evidence remains required'}
