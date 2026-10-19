from dataclasses import asdict
import unittest

from test_core import bar
from rl_stocks.environment.execution.simulator import Simulator, SimulationConfig
from rl_stocks.evaluation.direction import directional_attribution, declining_stock_summary


class DirectionTests(unittest.TestCase):
    def run_path(self, bars, actions, config=SimulationConfig(), atr=2.):
        sim = Simulator(config)
        rows = []
        for previous, current, action in zip(bars, bars[1:], actions):
            row = sim.advance(action, previous, current, atr)
            row['session'] = current.session
            rows.append(row)
        episode = {'bars': [asdict(b) for b in bars]}
        attribution = directional_attribution(episode, rows, sim.portfolio.events, config.initial_equity)
        self.assertAlmostEqual(attribution['long_net_pnl'] + attribution['short_net_pnl'],
                               rows[-1]['equity'] - config.initial_equity)
        self.assertAlmostEqual(attribution['long_costs'] + attribution['short_costs'], rows[-1]['costs'])
        return attribution, rows, sim, episode

    def test_falling_stock_short_profitable_after_all_costs(self):
        bars = [bar('2020-01-01'), bar(close=97, low=96),
                bar('2020-01-03', op=97, high=98, low=94, close=95, dividend=.5)]
        result, rows, sim, _ = self.run_path(bars, [[-.5, 1, 1, 1]] * 2)
        self.assertGreater(result['short_net_pnl'], 0)
        self.assertEqual(result['long_net_pnl'], 0)
        self.assertGreater(result['short_costs'], 0)
        self.assertEqual(result['short_active_sessions'], 2)
        self.assertLess(rows[-1]['equity'], 102500)  # costs and dividend liability paid

    def test_rising_stock_short_loses_and_disabled_short_stays_flat(self):
        bars = [bar('2020-01-01'), bar(close=103, high=104)]
        result, *_ = self.run_path(bars, [[-.5, 1, 1, 1]])
        self.assertLess(result['short_net_pnl'], 0)
        disabled, *_ = self.run_path(bars, [[-.5, 1, 1, 1]], SimulationConfig(allow_short=False))
        self.assertEqual(disabled['short_net_pnl'], 0)

    def test_reversal_partial_exit_split_and_dividend_reconcile(self):
        bars = [bar('2020-01-01'), bar(),
                bar('2020-01-03', op=50, high=50.25, low=49.75, close=50, split=2, dividend=1),
                bar('2020-01-04', op=50, high=50.2, low=48, close=49)]
        result, *_ = self.run_path(bars, [[.5, 1, 1, 1], [-.5, 1, 1, 1], [-.5, 1, -1, -1]])
        self.assertGreater(result['long_net_pnl'], 0)
        self.assertGreater(result['short_net_pnl'], 0)
        self.assertLess(result['max_equity_reconciliation_error'], 1e-6)

    def test_fractional_reverse_split_short_cash_in_lieu(self):
        bars = [bar('2020-01-01'), bar(),
                bar('2020-01-03', op=200, high=201, low=199, close=200, split=.5, dividend=1)]
        self.run_path(bars, [[-.101, 1, 1, 1], [0, 1, 1, 1]])

    def test_flat_and_gap_stop_and_early_episode_end(self):
        bars = [bar('2020-01-01'), bar(), bar('2020-01-03', op=120, high=121, low=119, close=120)]
        result, rows, sim, episode = self.run_path(bars, [[-.5, 1, 1, 1]] * 2)
        self.assertLess(result['short_net_pnl'], 0)
        self.assertEqual(sim.portfolio.quantity, 0)
        self.run_path(bars, [[0, 1, 1, 1]] * 2)
        directional_attribution(episode, rows[:1], sim.portfolio.events[:1])
        rows[-1]['equity'] += 1
        with self.assertRaises(ValueError):
            directional_attribution(episode, rows, sim.portfolio.events)

    def test_cohort_distinguishes_loss_reduction_from_short_profit(self):
        def result(symbol, policy, hold, short):
            return {'symbol': symbol, 'metrics': {'net_return': policy},
                    'baselines': {'buy_hold': {'metrics': {'net_return': hold}}},
                    'directional': {'short_net_pnl': short}}
        results = {'loses': result('DIS', -.1, -.4, 100),
                   'wins': result('AMT', .1, -.1, 100),
                   'etf': result('SPY', .1, -.1, 100),
                   'rises': result('AAPL', .1, .2, 100)}
        summary = declining_stock_summary(results, {'DIS', 'AMT', 'AAPL'})
        self.assertEqual(summary['count'], 2)
        self.assertEqual(summary['positive_policy_and_short_pnl_count'], 1)
