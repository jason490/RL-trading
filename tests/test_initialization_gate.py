import copy
import json
import unittest
from pathlib import Path
from rl_stocks.reporting.initialization import screen_gate, confirmation_gate


class GateTests(unittest.TestCase):
    def test_selected_improvement_cannot_hide_final_or_etf_deterioration(self):
        protocol=json.loads(Path('configs/experiments/action_initialization.json').read_text())
        def cell(selected, final, stock, etf):
            return {'selected':{'score':selected},'final':{'score':final,'mean_max_drawdown':-.2,
                    'groups':{'stock':{'net_return':stock},'etf':{'net_return':etf}}}}
        cells={'standard':cell(.1,.1,.1,.1),'low_noise':cell(.5,-.1,-.1,-.1),
               'wide':cell(.4,.2,.3,0.),'wide_low_noise':cell(.13,.13,.13,.13)}
        result=screen_gate(cells,protocol)
        self.assertEqual(result['promoted_candidate'],'wide_low_noise')
        self.assertFalse(result['comparisons']['low_noise']['passed'])
        self.assertFalse(result['comparisons']['wide']['passed'])
        bad=copy.deepcopy(cells);bad['wide_low_noise']['final']['mean_max_drawdown']=-.3
        self.assertFalse(screen_gate(bad,protocol)['passed'])
        del bad['wide']
        with self.assertRaises(ValueError):screen_gate(bad,protocol)

    @staticmethod
    def cell(score,hold=.1):
        phase={'score':score,'mean_max_drawdown':-.2,
               'stock_excess_by_symbol':{'A':score-hold,'B':score-hold},
               'groups':{g:{'net_return':score,'buy_hold_return':hold,'excess_return':score-hold,
                            'median_excess_return':score-hold,'max_drawdown':-.2,'buy_hold_max_drawdown':-.2}
                         for g in ('stock','etf')}}
        return {'selected':copy.deepcopy(phase),'final':copy.deepcopy(phase)}

    def test_confirmation_requires_paired_seeds_not_one_lucky_result(self):
        protocol=json.loads(Path('configs/experiments/action_initialization.json').read_text())
        cells={'standard':{str(s):self.cell(.1) for s in (11,23,37)},
               'wide':{str(s):self.cell(.2) for s in (11,23,37)}}
        result=confirmation_gate(cells,'wide',protocol)
        self.assertTrue(result['relative_improvement_passed'])
        self.assertTrue(result['targets']['outperforming']['passed'])
        self.assertFalse(result['objective_achieved'])
        for seed in ('23','37'):cells['wide'][seed]=self.cell(.09)
        result=confirmation_gate(cells,'wide',protocol)
        self.assertGreater(result['mean_final_gain_pp'],2.)
        self.assertFalse(result['relative_improvement_passed'])
        del cells['standard']['37']
        with self.assertRaises(ValueError):confirmation_gate(cells,'wide',protocol)

    def test_relative_improvement_does_not_imply_comparability_or_excuse_final_collapse(self):
        protocol=json.loads(Path('configs/experiments/action_initialization.json').read_text())
        cells={'standard':{str(s):self.cell(.1,.5) for s in (11,23,37)},
               'wide':{str(s):self.cell(.15,.5) for s in (11,23,37)}}
        result=confirmation_gate(cells,'wide',protocol)
        self.assertTrue(result['relative_improvement_passed'])
        self.assertFalse(result['targets']['comparable']['passed'])
        for seed in ('11','23','37'):cells['wide'][seed]['final']=self.cell(.05,.5)['final']
        result=confirmation_gate(cells,'wide',protocol)
        self.assertFalse(result['relative_improvement_passed'])
