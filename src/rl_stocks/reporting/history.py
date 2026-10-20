"""Fixed historical-augmentation screen; no checkpoint reselection."""
import math


def screen_gate(cells,protocol):
    if set(cells)!=set(protocol['arms']):raise ValueError('all three matched arms required')
    gate=protocol['screen_gate'];candidate=cells[protocol['candidate']]
    for cell in cells.values():
        for phase in ('selected','final'):
            row=cell[phase]
            values=[row['score'],row['mean_max_drawdown'],row['groups']['stock']['median_excess_return']]
            values += [row['groups'][g][k] for g in ('stock','etf') for k in ('net_return','excess_return')]
            if not all(math.isfinite(v) for v in values):raise ValueError('finite matched metrics required')
    controls={}
    for arm in (a for a in protocol['arms'] if a!=protocol['candidate']):
        ref=cells[arm]
        gains={phase:100*(candidate[phase]['score']-ref[phase]['score']) for phase in ('selected','final')}
        groups={g:100*(candidate['final']['groups'][g]['net_return']-ref['final']['groups'][g]['net_return']) for g in ('stock','etf')}
        dd=100*(ref['final']['mean_max_drawdown']-candidate['final']['mean_max_drawdown'])
        passed=(min(gains.values())>=gate['minimum_selected_and_final_gain_pp_vs_each_control']
            and min(groups.values())>=gate['minimum_final_stock_and_etf_gain_pp_vs_each_control']
            and dd<=gate['maximum_final_mean_drawdown_worsening_pp_vs_each_control'])
        controls[arm]=dict(gain_pp=gains,final_group_gain_pp=groups,final_drawdown_worsening_pp=dd,passed=passed)
    targets={}
    for phase in ('selected','final'):
        row=candidate[phase];groups={g:100*row['groups'][g]['excess_return'] for g in ('stock','etf')};median=100*row['groups']['stock']['median_excess_return']
        passed=min(groups.values())>=gate['minimum_selected_and_final_stock_and_etf_excess_pp'] and median>=gate['minimum_selected_and_final_median_stock_excess_pp']
        targets[phase]=dict(group_excess_pp=groups,median_stock_excess_pp=median,passed=passed)
    passed=all(c['passed'] for c in controls.values()) and all(t['passed'] for t in targets.values())
    return dict(passed=passed,promoted_candidate=protocol['candidate'] if passed else None,controls=controls,targets=targets,
        objective_achieved=False,prospective_confirmation_required=True)
