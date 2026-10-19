import unittest

from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.evaluation.rules import ExecutionRule


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = Normalizer(('ema_20_ratio', 'ema_50_ratio'), (.01, -.01), (.1, .2))

    def action(self, fast, slow):
        observation = self.normalizer.transform([fast, slow], self.normalizer.names) + [0] * 5
        return ExecutionRule('ema_trend', self.normalizer).predict(observation)[0]

    def test_causal_trend_uses_known_ema_order_and_retains_four_actions(self):
        self.assertEqual(self.action(.01, .03), [.95, 1., 1., 1.])
        self.assertEqual(self.action(-.01, -.03), [-.95, 1., 1., 1.])
        self.assertEqual(self.action(.03, .01)[0], 0.)
        self.assertEqual(self.action(-.03, -.01)[0], 0.)

    def test_rule_contract_rejects_incompatible_or_nonfinite_observations(self):
        policy = ExecutionRule('short', self.normalizer)
        with self.assertRaises(ValueError): policy.predict([0] * 2)
        with self.assertRaises(ValueError): policy.predict([float('nan')] * 7)
        with self.assertRaises(ValueError): ExecutionRule('future_direction', self.normalizer)
        self.assertEqual(policy.predict([0] * 7)[0][0], -.95)
