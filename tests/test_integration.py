"""Real dependency-backed checks, skipped explicitly if dependencies are unavailable."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in
                ('numpy', 'gymnasium', 'torch', 'stable_baselines3', 'plotly'))

@unittest.skipUnless(AVAILABLE, 'RL dependencies unavailable: install the pinned environment')
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rl_stocks.core.contracts import Request
        from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
        from rl_stocks.data.validation.bars import coverage
        from rl_stocks.data.storage.immutable import put
        from rl_stocks.data.generation.datasets import generate, read_dataset
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        request = Request()
        bars = SyntheticProvider().bars(request)
        provenance = {'coverage': coverage(bars, request), 'dividends_verified': True,
                      'capabilities': {'feed': 'synthetic'}}
        snapshot = put(cls.root / 'raw', bars, provenance)
        cls.dataset_path = generate(snapshot, cls.root / 'datasets', allow_unverified=True)
        cls.dataset = read_dataset(cls.dataset_path)
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
    def test_algorithm_configuration_rejection(self):
        from rl_stocks.rl.registry.factory import resolved_parameters
        for algorithm,overrides in [('ppo',{'n_steps':130}),('sac',{'learning_rate':-1}),('td3',{'buffer_size':1})]:
            with self.assertRaises(ValueError):resolved_parameters(algorithm,True,overrides)
        with self.assertRaises(ValueError):resolved_parameters('unknown')

    def test_reward_scaling_preserves_equity_and_observations(self):
        import numpy as np
        from rl_stocks.environment.trading.env import TradingEnv
        from rl_stocks.data.generation.datasets import normalizer_from
        episode=next(e for e in self.dataset['episodes'].values() if e['split']=='train')
        environments=[TradingEnv([episode],normalizer_from(self.dataset['normalizer']),self.dataset['names'],reward_scale=s) for s in (1.,10.)]
        for env in environments:env.reset(seed=11)
        for _ in range(5):
            left,right=[env.step(np.array([.5,1.,1.,1.],dtype=np.float32)) for env in environments]
            np.testing.assert_array_equal(left[0],right[0])
            self.assertEqual(left[4]['equity'],right[4]['equity'])
            self.assertAlmostEqual(right[1],10*left[1])
            self.assertAlmostEqual(right[4]['unscaled_reward'],left[1])
        for env in environments:env.close()

    def test_segmented_sac_curves_retain_post_update_milestones(self):
        from rl_stocks.training.runner import train
        from rl_stocks.training.curves import latest_points
        run,_=train(self.dataset_path,'sac',self.root/'curves',steps=256,milestones=[128,256],
                    evaluation_every=128,compact_checkpoints=True,reward_scale=10.)
        records=[json.loads(line) for line in (run/'learning-curve.jsonl').read_text().splitlines()]
        self.assertEqual([r['steps'] for r in latest_points(records)],[128,256])
        self.assertTrue(all(r['phase']=='post_update' for r in latest_points(records)))
        metadata=json.loads((run/'metadata.json').read_text())
        self.assertEqual(metadata['actual_steps'],256)
        self.assertEqual(metadata['reward_config']['scale'],10.)
        self.assertTrue((run/'learning-curve.csv').exists())

    def test_colab_backend_is_safe_for_real_sb3_import(self):
        import os
        import subprocess
        import sys
        from rl_stocks.runtime.colab.jobs import bootstrap_environment
        parent = dict(os.environ, MPLBACKEND='module://missing_notebook_backend',
                      PYTHONHOME='/invalid/notebook/python')
        environment = bootstrap_environment(parent, self.root/'research')
        subprocess.run([sys.executable, '-c',
                        'import stable_baselines3, matplotlib; assert matplotlib.get_backend().lower() == "agg"'],
                       env=environment, check=True, capture_output=True, text=True, timeout=30)

    def test_env_checkers_reset_and_boundaries(self):
        import numpy as np
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check
        from rl_stocks.environment.trading.env import TradingEnv
        from rl_stocks.data.generation.datasets import normalizer_from
        episode = next(e for e in self.dataset['episodes'].values() if e['split'] == 'train')
        env = TradingEnv([episode], normalizer_from(self.dataset['normalizer']), self.dataset['names'],
                         random_start=True, episode_steps=2)
        check_env(env, skip_render_check=True)
        sb3_check(env)
        a,_ = env.reset(seed=11)
        b,_ = env.reset(seed=11)
        np.testing.assert_array_equal(a,b)
        _,_,terminated,truncated,_ = env.step(np.zeros(4, dtype=np.float32))
        self.assertFalse(terminated or truncated)
        _,_,terminated,truncated,_ = env.step(np.zeros(4, dtype=np.float32))
        self.assertFalse(terminated); self.assertTrue(truncated)
    def test_all_algorithms_save_load_reports_and_schema(self):
        import numpy as np
        from rl_stocks.training.runner import train
        from rl_stocks.inference.bundle import FrozenPolicy
        from rl_stocks.data.generation.datasets import normalizer_from
        from rl_stocks.environment.trading.env import TradingEnv
        for algorithm,steps in [('ppo',512),('sac',1024),('td3',1024)]:
            with self.subTest(algorithm=algorithm):
                run,bundle = train(self.dataset_path,algorithm,self.root/'artifacts',steps=steps)
                metadata = json.loads((run/'metadata.json').read_text())
                self.assertEqual(metadata['actual_steps'],steps)
                self.assertTrue(metadata['python'].startswith('3.13.'))
                self.assertTrue((run/'validation.html').exists())
                selected=json.loads((run/'selected-validation.json').read_text())
                score=sum(r['metrics']['net_return'] for r in selected.values())/len(selected)
                self.assertAlmostEqual(score,metadata['selected_score'])
                self.assertIn(metadata['selected_step'],[h['steps'] for h in metadata['validation_checkpoints']])
                self.assertEqual(set(metadata['selected_metrics']),set(selected))
                self.assertFalse((run/'final-test.html').exists())
                if algorithm!='ppo':
                    from rl_stocks.rl.registry.factory import CLASSES
                    from rl_stocks.environment.trading.env import TradingEnv
                    from rl_stocks.data.generation.datasets import normalizer_from
                    training_episode=next(e for e in self.dataset['episodes'].values() if e['split']=='train')
                    resume_env=TradingEnv([training_episode],normalizer_from(self.dataset['normalizer']),self.dataset['names'],episode_steps=32)
                    resumed=CLASSES[algorithm].load(run/'latest.zip',env=resume_env,device='cpu')
                    resumed.load_replay_buffer(run/'replay.pkl')
                    self.assertGreater(resumed.replay_buffer.size(),64)
                    resumed.learn(total_timesteps=32,reset_num_timesteps=False)
                    self.assertEqual(resumed.num_timesteps,steps+32)

                policy = FrozenPolicy(bundle,self.dataset['schema'],'synthetic')
                episode = next(e for e in self.dataset['episodes'].values() if e['split']=='validation')
                env = TradingEnv([episode],normalizer_from(self.dataset['normalizer']),self.dataset['names'])
                obs,_=env.reset(seed=11)
                original = asdict(policy.normalizer)
                np.testing.assert_array_equal(policy.predict(obs),policy.predict(obs))
                self.assertEqual(original,asdict(policy.normalizer))
                with self.assertRaises(ValueError):FrozenPolicy(bundle,'wrong-schema','synthetic')
                with self.assertRaises(ValueError):FrozenPolicy(bundle,self.dataset['schema'],'sip')
                with self.assertRaises(ValueError):policy.predict([float('nan')]*len(obs))
                (bundle/'model.zip').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):FrozenPolicy(bundle,self.dataset['schema'],'synthetic')

    @unittest.skipUnless(importlib.util.find_spec('alpaca') is not None, 'optional alpaca-py unavailable')
    def test_mocked_alpaca_request_and_bars(self):
        from datetime import date,datetime,timezone
        from types import SimpleNamespace
        from rl_stocks.core.contracts import Request
        from rl_stocks.data.providers.alpaca.adapter import AlpacaProvider
        row=SimpleNamespace(timestamp=datetime(2020,1,2,5,tzinfo=timezone.utc),
            open=100.,high=101.,low=99.,close=100.,volume=1000.)
        class Response:
            data={'SPY':[row]}
            def model_dump(self,mode):return {'data':'mocked source'}
        class Client:
            def get_stock_bars(self,request):self.request=request;return Response()
        client=Client()
        adapter=AlpacaProvider('iex',client)
        bars=adapter.bars(Request(('SPY',),date(2020,1,1),date(2020,1,4)))
        self.assertEqual(client.request.feed.value,'iex')
        self.assertEqual(client.request.adjustment.value,'split')
        self.assertEqual(bars[0].session,'2020-01-02')
        self.assertEqual(bars[0].feed,'iex')
        self.assertEqual(bars[0].split,1.)
    @unittest.skipUnless(importlib.util.find_spec('yfinance') is not None, 'optional yfinance unavailable')
    def test_mocked_yahoo_split_and_dividend_semantics(self):
        from datetime import date
        from types import SimpleNamespace
        from unittest.mock import patch
        import pandas as pd
        from rl_stocks.core.contracts import Request
        from rl_stocks.data.providers.yahoo.adapter import YahooProvider
        frame=pd.DataFrame({'Open':[100.],'High':[101.],'Low':[99.],'Close':[100.],
            'Volume':[1000.],'Dividends':[.25],'Stock Splits':[2.]},
            index=pd.DatetimeIndex(['2020-01-02'],tz='America/New_York'))
        calls=[]
        def history(**kwargs):calls.append(kwargs);return frame
        adapter=YahooProvider()
        with patch('yfinance.Ticker',return_value=SimpleNamespace(history=history)):
            bars=adapter.bars(Request(('SPY',),date(2020,1,1),date(2020,1,4)))
        self.assertFalse(calls[0]['auto_adjust'])
        self.assertEqual(bars[0].split,1.)
        self.assertEqual(bars[0].dividend,.25)
        self.assertEqual(adapter.raw_source[0]['rows'][0]['Stock Splits'],2.)

if __name__ == '__main__': unittest.main()
