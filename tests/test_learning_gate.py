import unittest
from rl_stocks.reporting.learning import assess_learning


class GateTests(unittest.TestCase):
    def history(self,start,end,drawdown=-10.):
        return [{'steps':step,'phase':'post_update','score':score,'mean_max_drawdown_pct':dd}
                for step,score,dd in [(50000,start,-10.),(100000,end,drawdown)]]

    def setUp(self):
        self.gate={'mean_current_gain_pp':2.,'mean_selected_gain_pp':2.,'paired_positive_seeds':2,
            'maximum_drawdown_worsening_pp':2.,'historical_best_selected_mean_return':.2,
            'minimum_selected_gain_over_historical_pp':2.}

    def test_paired_gain_requires_broad_improvement_and_drawdown_control(self):
        histories=[self.history(.2,.25),self.history(.21,.27),self.history(.2,.24)]
        self.assertTrue(assess_learning(histories,self.gate)['passed'])
        histories[-1]=self.history(.2,.24,-20.)
        self.assertFalse(assess_learning(histories,self.gate)['passed'])
        histories=[self.history(.2,.5),self.history(.2,.19),self.history(.2,.19)]
        self.assertFalse(assess_learning(histories,self.gate)['passed'])

    def test_incomplete_milestones_cannot_pass(self):
        with self.assertRaises(ValueError):assess_learning([self.history(.2,.25)],self.gate)
        bad=[self.history(.2,.25) for _ in range(3)];bad[0][0]['phase']='on_step'
        with self.assertRaises(ValueError):assess_learning(bad,self.gate)

    def test_complete_cpu_summary_and_csv_keep_all_three_seeds(self):
        import tempfile,json,csv
        from pathlib import Path
        from rl_stocks.training.curves import curve_record
        from rl_stocks.reporting.learning import report_learning
        protocol={'datasets':{'baseline':'id'},'seeds':[11,23,37],'gate':self.gate}
        runs=[]
        for seed in protocol['seeds']:
            history=[]
            for step,score in [(50000,.2),(100000,.25)]:
                result={'A':{'metrics':{'net_return':score,'max_drawdown':-.1,'costs':10.,'turnover':1.,'sharpe':.2},
                    'baselines':{'buy_hold':{'metrics':{'net_return':.3}}}}}
                history.append(curve_record(result,history,step,'post_update',1.))
            runs.append({'device':'cpu','algorithm':'sac','status':'complete','actual_steps':100000,
                'requested_steps':100000,'dataset':'id','seed':seed,'git':{'dirty':''},
                'reward_config':{'scale':1.},'validation_checkpoints':history})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.html';value=report_learning(runs,protocol,path)
            self.assertTrue(value['complete']);self.assertTrue(value['improvement_gate_passed'])
            self.assertFalse(value['test_evaluated'])
            self.assertEqual(len(json.loads(path.with_suffix('.json').read_text())['runs']),3)
            with path.with_suffix('.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertAlmostEqual(float(rows[-1]['gain_since_50k_pp']),5.)
