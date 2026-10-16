import unittest
from copy import deepcopy
from rl_stocks.reporting.deterioration import diagnose_pair


def result(equity,cost,actions):
    rows=[];previous=100000.
    for date,value,action in zip(('2021-12-31','2022-01-03'),equity,actions):
        rows.append({'session':date,'equity':value,'previous_equity':previous,'raw_action':action,
                     'quantity':1 if action[0]>0 else -1,'exposure':action[0]})
        previous=value
    return {'rows':rows,'events':[],'metrics':{'net_return':equity[-1]/100000-1,
        'costs':cost,'max_drawdown':-.2,'turnover':2.}}


class DeteriorationTests(unittest.TestCase):
    def test_cost_accounting_does_not_blame_all_loss_on_churn(self):
        selected={'SPY-validation-0':result([110000,120000],1000,[[.5,0,0,0],[.5,0,0,0]])}
        final={'SPY-validation-0':result([90000,70000],2000,[[1,0,0,0],[-1,0,0,0]])}
        meta={'status':'complete','algorithm':'sac','seed':11,'dataset':'sample',
              'selected_step':10000,'actual_steps':20000,'selected_score':.2}
        value=diagnose_pair(meta,selected,final)
        self.assertAlmostEqual(value['final_minus_selected_return_pp'],-50)
        self.assertAlmostEqual(value['additional_costs_as_initial_equity_pp'],1)
        self.assertAlmostEqual(value['change_in_non_cost_pnl_as_initial_equity_pp'],-49)
        self.assertEqual(value['symbols_deteriorating'],1)
        phases=value['phases']
        self.assertEqual(phases['final']['actions']['saturation_fraction'],[1,0,0,0])
        self.assertEqual(phases['final']['actions']['target_sign_switch_fraction'],1)
        self.assertAlmostEqual(phases['selected']['yearly_return_pct']['2022'],100*(120000/110000-1))
        self.assertAlmostEqual(phases['final']['yearly_return_pct']['2022'],100*(70000/90000-1))

    def test_rejects_mismatched_splits_equity_and_invalid_actions(self):
        r={'SPY-validation-0':result([110000,120000],1000,[[.5,0,0,0],[.5,0,0,0]])}
        meta={'status':'complete','algorithm':'sac','seed':11,'dataset':'sample',
              'selected_step':10000,'actual_steps':20000,'selected_score':.2}
        with self.assertRaises(ValueError):diagnose_pair(meta,r,{'QQQ-validation-0':next(iter(r.values()))})
        with self.assertRaises(ValueError):diagnose_pair(meta,{'SPY-test-0':next(iter(r.values()))},{'SPY-test-0':next(iter(r.values()))})
        bad=deepcopy(r);bad['SPY-validation-0']['rows'][0]['previous_equity']=90000
        with self.assertRaises(ValueError):diagnose_pair(meta,r,bad)
        bad=deepcopy(r);bad['SPY-validation-0']['rows'][0]['raw_action'][0]=float('nan')
        with self.assertRaises(ValueError):diagnose_pair(meta,r,bad)
