"""Fixed three-arm development screen and charts; no checkpoint reselection."""
import math


def screen_gate(cells,protocol):
    if set(cells)!=set(protocol['arms']):raise ValueError('all declared matched arms required')
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


def render_report(result,path,title='Historical augmentation'):
    """Write a standalone interactive development chart from compact records."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Checkpoint development returns','Final stock and ETF returns'))
    for arm,cell in result['cells'].items():
        curve=cell['learning_curve']
        fig.add_trace(go.Scatter(x=[r['steps'] for r in curve],y=[r['return_pct'] for r in curve],mode='lines+markers',name=arm),row=1,col=1)
        fig.add_trace(go.Bar(x=['Stocks','ETFs'],y=[100*cell['final']['groups'][g]['net_return'] for g in ('stock','etf')],name=arm+' final'),row=1,col=2)
    row=next(iter(result['cells'].values()))['final']['groups']
    holding=sum(row[g]['count']*row[g]['buy_hold_return'] for g in ('stock','etf'))/sum(row[g]['count'] for g in ('stock','etf'))
    fig.add_hline(y=100*holding,line_dash='dash',annotation_text='Matched holding',row=1,col=1)
    fig.add_trace(go.Bar(x=['Stocks','ETFs'],y=[100*row[g]['buy_hold_return'] for g in ('stock','etf')],name='Matched holding'),row=1,col=2)
    fig.update_xaxes(title_text='Training transitions',row=1,col=1)
    fig.update_yaxes(title_text='Net return (%)')
    fig.update_layout(title=title+': repeated development validation; gate '+('passed' if result['screen']['passed'] else 'failed'),barmode='group')
    fig.write_html(str(path),include_plotlyjs=True)
