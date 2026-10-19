import copy
import json
import unittest
from pathlib import Path
from rl_stocks.reporting.initialization import screen_gate


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
