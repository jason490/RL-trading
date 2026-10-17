"""Forward receipt timing, durable restart accounting and fail-closed recovery."""
from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from rl_stocks.core.contracts import Bar, canonical, digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.data.storage.immutable import put
from rl_stocks.environment.execution.simulator import SimulationConfig, Simulator
from rl_stocks.features.registry.assemble import NAMES, assemble
from rl_stocks.inference.shadow.journal import transaction, records, get
from rl_stocks.inference.shadow.protocol import register, read_protocol
from rl_stocks.inference.shadow.runner import tick
from rl_stocks.reporting.shadow import report
from rl_stocks.runtime.shadow import ready


class FakePolicy:
    def __init__(self, path, schema, feed):
        self.manifest = json.loads((Path(path)/'manifest.json').read_text())
        self.normalizer = Normalizer(tuple(NAMES), (0.,)*7, (1.,)*7)

    def predict(self, obs):
        return np.array([.8, 1., 1., 1.], dtype=np.float32)

    def predict_recurrent(self, obs, *, state, episode_start):
        value = 0. if state is None else float(state[0][0, 0, 0])
        state = (np.ones((1, 1, 2), dtype=np.float32)*(value+1),)*2
        return self.predict(obs), state


class ShadowTests(unittest.TestCase):
    def setUp(self):
        import exchange_calendars as xc
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        calendar = xc.get_calendar('XNYS', start='2026-05-01', end='2026-10-23')
        self.bars = []
        for symbol in ('SPY', 'QQQ'):
            for i, session in enumerate(calendar.sessions):
                op, end = calendar.session_open(session), calendar.session_close(session)
                price = 100.+i*.1
                self.bars.append(Bar(symbol, str(session.date()), op.isoformat(), end.isoformat(),
                                     (end+timedelta(minutes=15)).isoformat(),
                                     (end+timedelta(minutes=16)).isoformat(),
                                     'synthetic', 'synthetic', 'v1', price, price+.15,
                                     price-.15, price+.05, 100000.))
        self.cap = {'provider':'synthetic', 'feed':'synthetic', 'adjustments':'raw_with_actions'}
        bootstrap = [b for b in self.bars if b.session <= '2026-10-14']
        self.snapshot = put(self.root/'raw', bootstrap, {'capabilities':self.cap})
        self.config = {'name':'test-forward', 'snapshot':str(self.snapshot),
                       'review_sessions':2, 'max_age_seconds':86400, 'arms':[]}
        for seed in (11, 23):
            manifest = {'names':list(NAMES), 'feature_schema':'daily-indicators-v1',
                        'metadata':{'algorithm':'sac', 'seed':seed,
                                    'simulation':asdict(SimulationConfig())},
                        'data_manifest':{'universe':['SPY','QQQ'], 'snapshot':self.snapshot.name,
                                         'weekly':False, 'provenance':{'capabilities':self.cap}}}
            folder = self.root/'bundles'/digest(manifest)
            folder.mkdir(parents=True)
            (folder/'manifest.json').write_bytes(canonical(manifest))
            self.config['arms'].append({'name':'sac-'+str(seed),'family':'sac','seed':seed,'bundle':str(folder)})
        self.campaign = register(self.config, self.root/'campaigns',
                                 now='2026-10-15T12:00:00+00:00', policy_loader=FakePolicy)

    def snapshot_at(self, day, revision=False):
        bars = [b for b in self.bars if '2026-10-12' <= b.session <= day]
        if revision:
            bars[0] = replace(bars[0], volume=bars[0].volume+1)
        return put(self.root/'receipts', bars, {'receipt':day})

    def step(self, day):
        return tick(self.campaign, self.snapshot_at(day), now=day+'T21:30:00+00:00', policy_loader=FakePolicy)

    def journal(self):
        with transaction(self.campaign/'journal.sqlite3') as db:
            return records(db)

    def test_restart_exact_simulation_and_duplicate_idempotency(self):
        first = self.step('2026-10-15')
        self.assertEqual(first['settlements'], 0)
        self.assertEqual(first['decisions'], 4)
        second = self.step('2026-10-16')
        self.assertEqual(second['settlements'], 4)
        previous = next(b for b in self.bars if b.symbol == 'SPY' and b.session == '2026-10-15')
        bar = next(b for b in self.bars if b.symbol == 'SPY' and b.session == '2026-10-16')
        atr = assemble([b for b in self.bars if b.symbol == 'SPY' and b.session <= previous.session])[0][-1]['atr']
        sim = Simulator()
        expected = sim.advance(FakePolicy.predict(None, None), previous, bar, atr)
        row = self.journal()[-1]['settlements'][0]
        for key in ('equity', 'quantity', 'age', 'costs', 'protection'):
            if key == 'protection': self.assertEqual(row[key], list(expected[key]) if expected[key] else None)
            else: self.assertEqual(row[key], expected[key])
        before = self.journal()
        with patch('rl_stocks.inference.shadow.runner.latest_features', side_effect=AssertionError):
            self.assertEqual(self.step('2026-10-16')['status'], 'already_recorded')
        self.assertEqual(before, self.journal())
        self.step('2026-10-19')
        monday = next(b for b in self.bars if b.symbol == 'SPY' and b.session == '2026-10-19')
        atr = assemble([b for b in self.bars if b.symbol == 'SPY' and b.session <= bar.session])[0][-1]['atr']
        expected = sim.advance(FakePolicy.predict(None, None), bar, monday, atr)
        row = self.journal()[-1]['settlements'][0]
        for key in ('equity', 'cash', 'quantity', 'age', 'costs', 'turnover'):
            self.assertEqual(row[key], expected[key])
        self.assertEqual(self.step('2026-10-20')['status'], 'review_due')
        result = report(self.campaign, self.root/'report.html')
        self.assertTrue(result['review_due'])
        self.assertFalse(result['paper_eligible'])
        self.assertTrue((self.root/'report.csv').exists())
        self.assertIn('Buy and hold', (self.root/'report.html').read_text())
        # Mark-to-market positions persist over the weekend; no reset on restart.
        self.assertEqual(self.journal()[-1]['settlements'][0]['age'], 2)

    def test_mid_batch_failure_rolls_back_all_positions_decisions_and_receipts(self):
        calls = 0
        original_predict = FakePolicy.predict
        def fail(obs):
            nonlocal calls
            calls += 1
            if calls == 2:raise RuntimeError('injected process failure')
            return original_predict(None, obs)
        with patch.object(FakePolicy, 'predict', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                self.step('2026-10-15')
        self.assertEqual(self.journal(), [])
        with transaction(self.campaign/'journal.sqlite3') as db:
            self.assertIsNone(get(db, 'stream:sac-11:SPY'))
            self.assertIsNone(get(db, 'snapshots'))
        self.assertEqual(self.step('2026-10-15')['decisions'], 4)

    def test_missed_session_and_late_decision_cannot_be_backfilled(self):
        self.step('2026-10-15')
        with self.assertRaisesRegex(ValueError, 'missed observed session'):
            self.step('2026-10-19')
        self.assertEqual(len(self.journal()), 1)
        with self.assertRaisesRegex(ValueError, 'missed next-session'):
            tick(self.campaign, self.snapshot_at('2026-10-16'),
                 now='2026-10-19T14:00:00+00:00', policy_loader=FakePolicy)

    def test_revisions_future_stale_and_missing_context_rejected(self):
        for day, now, revision, message in (
            ('2026-10-14', '2026-10-15T21:30:00+00:00', False, 'pre-registration'),
            ('2026-10-16', '2026-10-15T21:30:00+00:00', False, 'future'),
            ('2026-10-15', '2026-10-15T21:30:00+00:00', True, 'revised'),
            ('2026-10-16', '2026-10-18T21:30:00+00:00', False, 'stale')):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    tick(self.campaign, self.snapshot_at(day, revision), now=now, policy_loader=FakePolicy)
        bars = [b for b in self.bars if b.symbol == 'SPY' and b.session == '2026-10-15']
        path = put(self.root/'receipts', bars, {})
        with self.assertRaisesRegex(ValueError, 'incomplete current'):
            tick(self.campaign, path, now='2026-10-15T21:30:00+00:00', policy_loader=FakePolicy)
        self.assertEqual(self.journal(), [])

    def test_protocol_mutation_and_wrong_feed_rejected(self):
        p = self.campaign/'protocol.json'
        value = json.loads(p.read_text());value['review_sessions'] = 3
        original = p.read_bytes();p.write_bytes(canonical(value))
        with self.assertRaisesRegex(ValueError, 'protocol changed'):read_protocol(self.campaign)
        p.write_bytes(original)
        incoming = [replace(b, feed='other') for b in self.bars if b.session == '2026-10-15']
        path = put(self.root/'receipts', incoming, {})
        with self.assertRaisesRegex(ValueError, 'provider/feed'):
            tick(self.campaign, path, now='2026-10-15T21:30:00+00:00', policy_loader=FakePolicy)

    def test_recurrent_memory_is_per_symbol_and_durable(self):
        config = dict(self.config)
        config['arms'] = [dict(self.config['arms'][0])]
        p = Path(config['arms'][0]['bundle'])/'manifest.json'
        manifest = json.loads(p.read_text());manifest['metadata']['algorithm'] = 'recurrent_ppo'
        p.write_bytes(canonical(manifest))
        self.campaign = register(config, self.root/'memory', now='2026-10-15T12:00:00+00:00', policy_loader=FakePolicy)
        self.step('2026-10-15');self.step('2026-10-16')
        with transaction(self.campaign/'journal.sqlite3') as db:
            for symbol in ('SPY', 'QQQ'):
                state = get(db, 'stream:sac-11:'+symbol)
                self.assertEqual(state['memory'], [[[[2., 2.]]], [[[2., 2.]]]])

    def test_scheduler_waits_for_post_registration_close_and_fixed_review(self):
        self.assertEqual(ready(self.campaign, '2026-10-15T18:00:00+00:00')['status'], 'waiting_for_forward_close')
        self.assertEqual(ready(self.campaign, '2026-10-15T21:30:00+00:00')['status'], 'ready')
        self.step('2026-10-15')
        self.assertEqual(ready(self.campaign, '2026-10-16T18:00:00+00:00')['status'], 'already_recorded')
        self.step('2026-10-16');self.step('2026-10-19')
        self.assertEqual(ready(self.campaign, '2026-10-19T21:30:00+00:00')['status'], 'review_due')

    def test_missing_forward_bar_calendar_mismatch_and_manifest_change_rejected(self):
        incoming = [b for b in self.bars if '2026-10-16' == b.session]
        path = put(self.root/'receipts', incoming, {})
        with self.assertRaisesRegex(ValueError, 'missing forward context'):
            tick(self.campaign, path, now='2026-10-16T21:30:00+00:00', policy_loader=FakePolicy)
        incoming = [replace(b, open_time=b.open_time.replace('13:30', '13:31'))
                    for b in self.bars if '2026-10-15' == b.session]
        path = put(self.root/'receipts', incoming, {})
        with self.assertRaisesRegex(ValueError, 'exchange open/close'):
            tick(self.campaign, path, now='2026-10-15T21:30:00+00:00', policy_loader=FakePolicy)
        folder = Path(self.config['arms'][0]['bundle'])
        m = json.loads((folder/'manifest.json').read_text());m['metadata']['seed'] = 100
        (folder/'manifest.json').write_bytes(canonical(m))
        with self.assertRaisesRegex(ValueError, 'registered manifest changed'):
            self.step('2026-10-15')
        self.assertEqual(self.journal(), [])

    def test_feature_prefix_causality_with_extreme_future_prices(self):
        from rl_stocks.inference.shadow.features import latest_features
        manifest = json.loads((Path(self.config['arms'][0]['bundle'])/'manifest.json').read_text())
        prefix = [b for b in self.bars if b.session <= '2026-10-15']
        expected = latest_features(prefix, manifest, '2026-10-15')
        future = [replace(b, open=b.open*100, high=b.high*100, low=b.low*100, close=b.close*100)
                  for b in self.bars if b.session > '2026-10-15']
        actual = latest_features(prefix+future, manifest, '2026-10-15')
        self.assertEqual(expected, actual)

    def test_collector_filters_incomplete_bars_and_halts_on_splits(self):
        from rl_stocks.inference.shadow.collect import collect
        from rl_stocks.core.contracts import Capabilities
        provider = type('Provider', (), {})()
        provider.capabilities = Capabilities('yahoo', feed='synthetic')
        provider.raw_source = []
        provider.bars = lambda request: [b for b in self.bars if b.session in ('2026-10-15','2026-10-16')]
        protocol = read_protocol(self.campaign)
        protocol['feed_contract']['provider'] = 'yahoo'
        with patch('rl_stocks.inference.shadow.collect.read_protocol', return_value=protocol):
            path = collect(self.campaign, self.root/'collected', now='2026-10-15T21:30:00+00:00', provider=provider)
            from rl_stocks.data.storage.immutable import load
            bars, _ = load(path)
            self.assertEqual({b.session for b in bars}, {'2026-10-15'})
            from rl_stocks.data.providers.yahoo.corrections import SPY_HIGH_POLICY
            protocol['receipt_correction_policy'] = SPY_HIGH_POLICY
            with self.assertRaisesRegex(ValueError, 'correction policy differs'):
                collect(self.campaign, self.root/'collected', now='2026-10-15T21:30:00+00:00', provider=provider)
            provider.correction_policy = SPY_HIGH_POLICY['id']
            provider.corrections = [{'policy':SPY_HIGH_POLICY['id'], 'delta':.010009765625}]
            path = collect(self.campaign, self.root/'collected', now='2026-10-15T21:30:00+00:00', provider=provider)
            _, provenance = load(path)
            self.assertEqual(provenance['correction_policy'], SPY_HIGH_POLICY)
            self.assertEqual(provenance['corrections'], provider.corrections)
            provider.raw_source = [{'rows':[{'date':'2026-10-15','Stock Splits':2.}]}]
            with self.assertRaisesRegex(ValueError, 'split requires'):
                collect(self.campaign, self.root/'collected', now='2026-10-15T21:30:00+00:00', provider=provider)
