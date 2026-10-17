"""Only the explicitly approved bar may be repaired; raw evidence stays original."""
from dataclasses import replace
from datetime import date
import unittest
from unittest.mock import patch
import pandas as pd
from rl_stocks.core.contracts import Request
from rl_stocks.data.providers.yahoo.adapter import YahooProvider
from rl_stocks.data.providers.yahoo.corrections import SPY_HIGH_POLICY, correct_spy_high
from rl_stocks.data.validation.bars import validate


class YahooCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([{'Open':773.239990234375, 'High':773.22998046875,
                                   'Low':760.6400146484375, 'Close':763.25,
                                   'Volume':100000., 'Dividends':0., 'Stock Splits':0.}],
                                  index=[pd.Timestamp('2026-10-16', tz='America/New_York')])
        self.request = Request(('SPY',), date(2026,10,16), date(2026,10,17))

    def fetch(self, policy=None, frame=None, request=None):
        provider = YahooProvider(correction_policy=policy)
        with patch('yfinance.Ticker') as ticker:
            ticker.return_value.history.return_value = self.frame if frame is None else frame
            bars = provider.bars(request or self.request)
        return provider, bars

    def test_default_provider_stays_strict_and_unknown_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, 'OHLC geometry'):
            self.fetch()
        with self.assertRaisesRegex(ValueError, 'unknown Yahoo correction'):
            YahooProvider(correction_policy='allow-all')

    def test_exact_opt_in_repairs_high_and_keeps_raw_response(self):
        provider, bars = self.fetch(SPY_HIGH_POLICY['id'])
        validate(bars)
        self.assertEqual(bars[0].high, bars[0].open)
        self.assertEqual(provider.raw_source[0]['rows'][0]['High'], float(self.frame.iloc[0].High))
        self.assertLess(provider.raw_source[0]['rows'][0]['High'], bars[0].high)
        self.assertEqual(len(provider.corrections), 1)
        audit = provider.corrections[0]
        self.assertEqual(audit['policy'], SPY_HIGH_POLICY['id'])
        self.assertAlmostEqual(audit['delta'], .010009765625)
        self.assertIn(SPY_HIGH_POLICY['id'], bars[0].revision)

    def test_exception_does_not_cover_other_dates_symbols_or_larger_errors(self):
        for day, symbol, high in [('2026-10-15','SPY',773.22998046875),
                                  ('2026-10-16','QQQ',773.22998046875),
                                  ('2026-10-16','SPY',772.)]:
            with self.subTest(day=day, symbol=symbol, high=high):
                frame = self.frame.copy()
                frame.index = [pd.Timestamp(day, tz='America/New_York')]
                frame.loc[frame.index[0], 'High'] = high
                request = Request((symbol,), date(2026,10,15), date(2026,10,17))
                with self.assertRaises(ValueError):
                    self.fetch(SPY_HIGH_POLICY['id'], frame, request)

    def test_upstream_fixed_bar_unchanged_and_other_bad_fields_not_hidden(self):
        frame = self.frame.copy();frame.loc[frame.index[0], 'High'] = 774.
        provider, bars = self.fetch(SPY_HIGH_POLICY['id'], frame)
        self.assertEqual(bars[0].high, 774.)
        self.assertEqual(provider.corrections, [])
        fixed = bars[0]
        # The approved exception cannot suppress the normal validator.
        same, audit = correct_spy_high(replace(fixed, low=800.))
        self.assertIsNone(audit)
        with self.assertRaisesRegex(ValueError, 'OHLC geometry'):validate([same])
        frame = self.frame.copy();frame.loc[frame.index[0], 'Close'] = 763.26
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.fetch(SPY_HIGH_POLICY['id'], frame)

    def test_optional_missing_raw_field_is_losslessly_tagged_but_required_nan_rejected(self):
        from rl_stocks.core.contracts import canonical
        frame = self.frame.copy()
        frame['Adj Close'] = float('nan')
        provider, bars = self.fetch(SPY_HIGH_POLICY['id'], frame)
        source = provider.raw_source[0]['rows'][0]
        self.assertEqual(source['Adj Close'], {'nonfinite_float':'nan'})
        self.assertEqual(source['High'], float(self.frame.iloc[0].High))
        canonical({'raw_source':provider.raw_source, 'corrections':provider.corrections})
        self.assertEqual(bars[0].high, bars[0].open)
        for name in ('Volume', 'Close', 'High'):
            invalid = frame.copy();invalid.loc[invalid.index[0], name] = float('nan')
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, 'nonfinite bar'):
                    self.fetch(SPY_HIGH_POLICY['id'], invalid)
