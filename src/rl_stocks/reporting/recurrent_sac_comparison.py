"""Pure preregistered reductions over already verified compact development cells."""
from copy import deepcopy
from statistics import mean

from rl_stocks.reporting.history import screen_gate as base_screen_gate
from rl_stocks.training.recurrent_comparison import protocol


def screen_gate(cells):
    """Caller must verify physical model/data/source/phase identities first."""
    return base_screen_gate(cells,protocol())


def confirmation_gate(seed_cells):
    spec=protocol()
    if set(seed_cells)!=set(spec['confirmation_seeds']) or any(type(s) is not int for s in seed_cells):
        raise ValueError('all and only registered three paired seeds required')
    gates={seed:screen_gate(cells) for seed,cells in seed_cells.items()}
    averaged={}
    for arm in spec['arms']:
        averaged[arm]={}
        for phase in ('selected','final'):
            rows=[seed_cells[seed][arm][phase] for seed in spec['confirmation_seeds']]
            cell=deepcopy(rows[0])
            for key in ('score','mean_max_drawdown'):
                cell[key]=mean(row[key] for row in rows)
            for group in ('stock','etf'):
                for key in ('net_return','excess_return'):
                    cell['groups'][group][key]=mean(row['groups'][group][key] for row in rows)
            cell['groups']['stock']['median_excess_return']=mean(row['groups']['stock']['median_excess_return'] for row in rows)
            averaged[arm][phase]=cell
    aggregate=screen_gate(averaged)
    candidate=spec['candidate']
    paired={control:sum(seed_cells[seed][candidate]['final']['score']>
                        seed_cells[seed][control]['final']['score'] for seed in seed_cells)
            for control in spec['arms'] if control!=candidate}
    passing=sum(g['passed'] for g in gates.values())
    passed=(aggregate['passed'] and passing>=2 and min(paired.values())>=2)
    outperformance_seeds=sum(all(seed_cells[seed][candidate][phase]['groups'][group]['excess_return']>=0
                            for phase in ('selected','final') for group in ('stock','etf'))
                            for seed in seed_cells)
    positive_mean=all(averaged[candidate][phase]['groups'][group]['excess_return']>=0
                      for phase in ('selected','final') for group in ('stock','etf'))
    return dict(passed=passed,seed_gates=gates,aggregate_gate=aggregate,
                paired_positive_final_seeds=paired,target_passing_seeds=passing,
                development_outperformance=passed and positive_mean and outperformance_seeds>=2,
                eligible_for_prospective=passed,objective_achieved=False,
                prospective_confirmation_required=True)


def phase_summary(results):
    """Compact exact-panel metrics; future decliners remain reporting-only."""
    from rl_stocks.reporting.initialization import group_metrics
    from rl_stocks.evaluation.direction import declining_stock_summary
    from rl_stocks.evaluation.runner import selection_score
    spec=protocol()
    if (len(results)!=len(spec['validation_symbols'])
            or sorted(row['symbol'] for row in results.values())!=sorted(spec['validation_symbols'])):
        raise ValueError('exact one-episode-per-symbol development panel required')
    stocks=set(spec['validation_symbols'])-set(spec['etf_symbols'])
    summary=group_metrics(results,stocks)
    summary['score']=selection_score(results)
    summary['declining_stocks_reporting_only']=declining_stock_summary(results,stocks)
    return summary
