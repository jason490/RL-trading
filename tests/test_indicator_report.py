import copy,json,tempfile,unittest
from pathlib import Path
from rl_stocks.reporting.indicators import summarize_indicators,report_indicators

class IndicatorReportTests(unittest.TestCase):
    def fixture(self):
        protocol=json.loads(Path('configs/experiments/indicator_followup.json').read_text());runs=[]
        for label,identity in protocol['datasets'].items():
            extra={'baseline':0,'supply_demand':.03,'rotation':.01,'combined':.05}[label]
            for i,seed in enumerate(protocol['seeds']):
                score=.2+i*.01+extra
                metrics={'net_return':score,'max_drawdown':-.1,'costs':100.,'turnover':2.}
                runs.append({'status':'complete','smoke':False,'algorithm':'sac','device':'cpu',
                    'requested_steps':250000,'actual_steps':250000,'reward_config':{'kind':'net_log_equity','scale':1.},
                    'git':{'dirty':''},'source_hash':'source','dataset':identity,'seed':seed,'bundle_id':label+str(seed),
                    'simulation':{},'versions':{},'resolved_parameters':{},'selection_metric':'validation',
                    'selected_score':score,'selected_metrics':{'A':{'symbol':'A','policy':metrics,
                        'baselines':{'buy_hold':{'net_return':.3,'max_drawdown':-.2}}}},
                    'validation_checkpoints':[{'steps':250000,'phase':'post_update','score':score-.01,
                        'return_pct':100*(score-.01),'mean_max_drawdown_pct':-10.}]})
        return runs,protocol

    def test_factorial_attribution_and_real_report(self):
        runs,p=self.fixture();v=summarize_indicators(runs,p,{'source':'same-learning'})
        self.assertTrue(v['complete']);self.assertEqual(v['received_cells'],12)
        effects={r['contrast']:r['mean_selected_gain_pp'] for r in v['contrasts']}
        self.assertAlmostEqual(effects['supply_demand_without_sector'],3.)
        self.assertAlmostEqual(effects['supply_demand_with_sector'],4.)
        self.assertAlmostEqual(effects['interaction'],1.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.html';report_indicators(runs,p,{'source':'same-learning'},path)
            self.assertTrue(path.is_file());self.assertTrue(path.with_suffix('.csv').is_file())
            self.assertFalse(json.loads(path.with_suffix('.json').read_text())['test_evaluated'])

    def test_outlier_and_drawdown_cannot_pass_feature_gate(self):
        runs,p=self.fixture()
        for r,delta in zip([r for r in runs if r['dataset']==p['datasets']['supply_demand']],[.2,-.01,-.01]):
            baseline=.2+(p['seeds'].index(r['seed']))*.01
            r['selected_score']=baseline+delta;r['selected_metrics']['A']['policy']['net_return']=baseline+delta
        v=summarize_indicators(runs,p,{'source':'same-learning'})
        self.assertFalse(next(f for f in v['families'] if f['family']=='supply_demand')['feature_gate']['passed'])
        runs,p=self.fixture()
        for r in runs:
            if r['dataset']==p['datasets']['combined']:r['selected_metrics']['A']['policy']['max_drawdown']=-.2
        v=summarize_indicators(runs,p,{'source':'same-learning'})
        self.assertFalse(next(f for f in v['families'] if f['family']=='combined')['feature_gate']['passed'])

    def test_partial_or_mismatched_controls_do_not_support_attribution(self):
        runs,p=self.fixture();v=summarize_indicators(runs[:-1],p,{'source':'same-learning'})
        self.assertFalse(v['complete']);self.assertEqual(v['contrasts'],[])
        for field,value in [('versions',{'torch':'different'}),('git',{'dirty':' M file'}),('actual_steps',100000)]:
            bad=copy.deepcopy(runs);bad[-1][field]=value
            with self.assertRaises(ValueError):summarize_indicators(bad,p,{'source':'same-learning'})
        bad=copy.deepcopy(runs);bad[-1]['selected_metrics']['A']['baselines']['buy_hold']['net_return']=.4
        with self.assertRaises(ValueError):summarize_indicators(bad,p,{'source':'same-learning'})
        with self.assertRaises(ValueError):summarize_indicators(runs,p,{'source':'a','other':'b'})
