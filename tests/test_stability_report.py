import copy
import json
from pathlib import Path
import tempfile
import unittest
from rl_stocks.reporting.stability import report_stability,summarize_stability
from rl_stocks.rl.registry.factory import resolved_parameters


class StabilityReportTests(unittest.TestCase):
    def fixture(self):
        protocol=json.loads(Path('configs/experiments/algorithm_stability.json').read_text());runs=[]
        for algorithm,selected,gap in [('sac',.3,.1),('ppo',.35,.05),('recurrent_ppo',.4,.02)]:
            for i,seed in enumerate(protocol['seeds']):
                score=selected+i*.001
                history=[{'steps':s,'phase':'post_update' if s==100000 else 'on_step',
                    'score':score if s==30000 else score-gap,'mean_max_drawdown_pct':-10.}
                    for s in range(10000,100001,10000)]
                runs.append({'algorithm':algorithm,'seed':seed,'status':'complete','device':'cpu',
                    'dataset':protocol['dataset'],'actual_steps':100000,'requested_steps':100000,
                    'reward_config':{'kind':'net_log_equity','scale':1.},'milestones':[100000],
                    'evaluation_every':10000,'compact_checkpoints':True,'git':{'dirty':''},'source_hash':'source',
                    'resolved_parameters':resolved_parameters(algorithm,False,{'n_steps':1000,'batch_size':50} if algorithm!='sac' else None),
                    'selected_metrics':{'A':{'policy':{'net_return':score,'max_drawdown':-.1},
                        'baselines':{'buy_hold':{'net_return':.3}}}},'simulation':{},'versions':{},
                    'selection_metric':'validation','selected_step':30000,'selected_score':score,
                    'bundle_id':algorithm+str(seed),'validation_checkpoints':history})
        return runs,protocol

    def test_gate_checks_performance_and_retention_and_exports(self):
        runs,p=self.fixture();v=summarize_stability(runs,p,{'source':'learning'})
        self.assertTrue(v['complete']);self.assertEqual(v['completed_cells'],9)
        for pair in v['comparisons'].values():self.assertTrue(pair['passed'])
        pair=v['comparisons']['ppo_versus_sac']
        self.assertAlmostEqual(pair['mean_selected_gain_pp'],5.)
        self.assertAlmostEqual(pair['mean_final_gain_pp'],10.)
        self.assertAlmostEqual(pair['reduction_in_mean_selected_minus_final_gap_pp'],5.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'comparison.html';report_stability(runs,p,{'source':'learning'},path)
            self.assertTrue(path.exists());self.assertTrue(path.with_suffix('.csv').exists())
            self.assertFalse(json.loads(path.with_suffix('.json').read_text())['test_evaluated'])

    def test_incomplete_and_mismatched_runs_rejected_or_not_aggregated(self):
        runs,p=self.fixture();v=summarize_stability(runs[:-1],p,{'source':'learning'})
        self.assertFalse(v['complete']);self.assertNotIn('recurrent_ppo_versus_sac',v['comparisons'])
        for field,value in [('versions',{'different':True}),('device','cuda'),('actual_steps',99999)]:
            bad=copy.deepcopy(runs);bad[-1][field]=value
            with self.assertRaises(ValueError):summarize_stability(bad,p,{'source':'learning'})
        bad=copy.deepcopy(runs);bad[-1]['validation_checkpoints'][-1]['phase']='on_step'
        with self.assertRaises(ValueError):summarize_stability(bad,p,{'source':'learning'})
        with self.assertRaises(ValueError):summarize_stability(runs+[runs[-1]],p,{'source':'learning'})

    def test_stability_without_return_gain_and_larger_drawdown_cannot_pass(self):
        from rl_stocks.reporting.stability import paired_gate
        runs,p=self.fixture();v=summarize_stability(runs,p,{'source':'learning'})
        control=[r for r in v['runs'] if r['algorithm']=='sac']
        flat=[{**r,'selected_return_pct':0.,'final_return_pct':0.} for r in control]
        self.assertFalse(paired_gate(control,flat,p['gate'])['passed'])
        candidate=[{**r,'selected_return_pct':r['selected_return_pct']+5,
            'final_return_pct':r['final_return_pct']+10,'final_drawdown_pct':-20.} for r in control]
        self.assertFalse(paired_gate(control,candidate,p['gate'])['passed'])
