"""Standalone compact development comparison; no training or policy reselection."""
from pathlib import Path


def render_comparison(receipt,destination):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from rl_stocks.training.recurrent_comparison import protocol
    spec=protocol()
    seeds=sorted(receipt['cells'],key=int)
    figure=make_subplots(rows=4,cols=2,specs=[[{},{}],[{},{}],[{'colspan':2},None],[{'colspan':2},None]],
        subplot_titles=('Selected: stocks','Selected: ETFs','Final: stocks','Final: ETFs',
                        'Checkpoint mean development returns','Final: individual stock excess vs holding (mean across seeds)'))
    labels=dict(native_sac='SAC',recurrent_ppo_reference='Recurrent PPO',gru_sac='GRU-SAC')
    colors=dict(native_sac='#3366cc',recurrent_ppo_reference='#ff9900',gru_sac='#109618')
    for row,phase in enumerate(('selected','final'),1):
        for col,group in enumerate(('stock','etf'),1):
            for arm in spec['arms']:
                values=[100*receipt['cells'][seed][arm][phase]['groups'][group]['net_return'] for seed in seeds]
                figure.add_trace(go.Bar(x=[f'Seed {s}' for s in seeds],y=values,name=labels[arm],
                    legendgroup=arm,showlegend=row==1 and col==1,marker_color=colors[arm]),row=row,col=col)
            holding=[100*receipt['cells'][seed][spec['arms'][0]][phase]['groups'][group]['buy_hold_return'] for seed in seeds]
            figure.add_trace(go.Scatter(x=[f'Seed {s}' for s in seeds],y=holding,mode='lines+markers',
                name='Matched holding',legendgroup='holding',showlegend=row==1 and col==1,
                line=dict(color='#222222',dash='dash')),row=row,col=col)
    for seed in seeds:
        for arm in spec['arms']:
            curve=receipt['cells'][seed][arm]['curve']
            figure.add_trace(go.Scatter(x=[p['steps'] for p in curve],y=[p['return_pct'] for p in curve],
                mode='lines+markers',name=f'{labels[arm]}, seed {seed}',legendgroup=arm,
                line=dict(color=colors[arm],dash={'11':'solid','23':'dash','37':'dot'}[seed]),hovertemplate='Step %{x}: %{y:.2f}%<extra>%{fullData.name}</extra>'),row=3,col=1)
    stocks=sorted(set(spec['validation_symbols'])-set(spec['etf_symbols']))
    for arm in spec['arms']:
        excess=[100*sum(receipt['cells'][seed][arm]['final']['stock_excess_by_symbol'][symbol]
                        for seed in seeds)/len(seeds) for symbol in stocks]
        figure.add_trace(go.Bar(x=stocks,y=excess,name=labels[arm],legendgroup=arm,
                               showlegend=False,marker_color=colors[arm]),row=4,col=1)
    figure.add_hline(y=0,line_dash='dash',line_color='#222222',row=4,col=1)
    figure.update_yaxes(title_text='Return (%)')
    figure.update_yaxes(title_text='Excess (percentage points)',row=4,col=1)
    figure.update_xaxes(title_text='Collected transitions',row=3,col=1)
    figure.update_layout(barmode='group',height=1400,
        title=f"Reused 2021–2023 development: {receipt['stage']} gate {'passed' if receipt['gate']['passed'] else 'failed'}; prospective evidence required")
    figure.write_html(str(Path(destination)),include_plotlyjs=True)
