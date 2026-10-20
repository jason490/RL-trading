"""Two-arm temperature gate and exact default-control reproduction checks."""
import torch
from rl_stocks.reporting.history import screen_gate


def gate(cells,protocol):
    spec={**protocol,'screen_gate':dict(protocol['screen_gate'])}
    for key in ('minimum_selected_and_final_gain_pp','minimum_final_stock_and_etf_gain_pp','maximum_final_mean_drawdown_worsening_pp'):
        spec['screen_gate'][key+'_vs_each_control']=spec['screen_gate'][key+'_vs_control']
    return screen_gate(cells,spec)


def model_equal(first,second):
    a=first.policy.state_dict();b=second.policy.state_dict()
    return (set(a)==set(b) and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)
        and torch.equal(first.log_ent_coef.detach().cpu(),second.log_ent_coef.detach().cpu())
        and first._n_updates==second._n_updates)
