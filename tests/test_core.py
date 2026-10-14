import csv
from dataclasses import asdict, replace
from datetime import date
import json
import math
from pathlib import Path
import tempfile
import unittest

from rl_stocks.core.contracts import Bar, RecordType, Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.providers.files.adapter import FileProvider
from rl_stocks.data.validation.bars import validate, coverage
from rl_stocks.data.storage.immutable import put, load
from rl_stocks.data.generation.datasets import generate, read_dataset, Normalizer
from rl_stocks.features.registry.assemble import assemble
from rl_stocks.features.volume.relative import relative_volume
from rl_stocks.features.timeframes.weekly import completed_weekly
from rl_stocks.environment.accounting.portfolio import Portfolio
from rl_stocks.environment.execution.simulator import Simulator, SimulationConfig
from rl_stocks.risk.decisions import decode, valid_entry
from rl_stocks.runtime.colab.budget import BudgetStop, Ledger, Usage, parse_usage
from rl_stocks.runtime.colab.runner import ColabJob


def bar(day='2020-01-02',op=100,high=101,low=99,close=100,**kw):
    return Bar('SPY',day,day+'T14:30:00+00:00',day+'T21:00:00+00:00',
        day+'T21:15:00+00:00','2026-10-14T00:00:00+00:00','synthetic','synthetic','fixture',
        op,high,low,close,1000,**kw)

class DataTests(unittest.TestCase):
    def setUp(self):
        self.request=Request(('SPY',),date(2020,1,1),date(2020,6,1))
        self.bars=SyntheticProvider().bars(self.request)
    def test_seed_and_capability(self):
        self.assertEqual(self.bars,SyntheticProvider().bars(self.request))
        with self.assertRaises(ValueError):SyntheticProvider.capabilities.require(RecordType.OPTIONS)
    def test_validation(self):
        validate(self.bars)
        for bad in ([self.bars[0],self.bars[0]], [replace(self.bars[0],close=float('nan'))],
            [replace(self.bars[0],available_time=self.bars[0].open_time)],
            [replace(self.bars[0],open_time='2020-01-01T00:00:00')]):
            with self.assertRaises(ValueError):validate(bad)
    def test_coverage_missing(self):
        r=coverage(self.bars,self.request,[b.session for b in self.bars]+['2020-06-02'])
        self.assertEqual(r['SPY']['missing'],['2020-06-02'])
    def test_provider_interchangeability(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bars.csv'
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=asdict(self.bars[0]).keys());writer.writeheader()
                writer.writerows(map(asdict,self.bars))
            self.assertEqual(self.bars,FileProvider(path).bars(self.request))
    def test_storage_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p=put(d,self.bars,{'source':'fixture'})
            self.assertEqual(p,put(d,self.bars,{'source':'fixture'}))
            self.assertEqual(load(p)[0],self.bars)
            (p/'snapshot.json').write_text('{}')
            with self.assertRaises(ValueError):load(p)
    def test_prefix_feature_causality(self):
        full,names=assemble(self.bars)
        prefix,_=assemble(self.bars[:80])
        self.assertEqual(prefix,[r for r in full if r['bar']<80])
        altered=self.bars[:80]+[replace(b,close=b.close*1.001,high=b.high*1.01) for b in self.bars[80:]]
        self.assertEqual(prefix,[r for r in assemble(altered)[0] if r['bar']<80])
    def test_volume_excludes_current(self):
        values=relative_volume([10]*20+[1000])
        self.assertEqual(values[-1],100)
    def test_weekly_causality(self):
        full=completed_weekly(self.bars)
        self.assertEqual(full[:80],completed_weekly(self.bars[:80]))
    def test_normalizer_immutable(self):
        n=Normalizer.fit(['a','b'],[[1,2],[3,4]])
        original=asdict(n)
        n.transform([100,200],['a','b'])
        self.assertEqual(original,asdict(n))
        with self.assertRaises(ValueError):n.transform([1,2],['b','a'])
        with self.assertRaises(ValueError):n.transform([float('inf'),2],['a','b'])
    def test_dataset_splits_and_purge(self):
        request=Request()
        bars=SyntheticProvider().bars(request)
        report=coverage(bars,request)
        with tempfile.TemporaryDirectory() as d:
            p=put(Path(d)/'raw',bars,{'coverage':report,'dividends_verified':True,'capabilities':{'feed':'synthetic'}})
            with self.assertRaises(ValueError):generate(p,Path(d)/'datasets')
            target=generate(p,Path(d)/'datasets',allow_unverified=True)
            data=read_dataset(target)
            train=[]
            for episode in data['episodes'].values():
                dates=[b['session'] for b in episode['bars']]
                if episode['split']=='train':
                    self.assertLess(max(dates),'2020-12-31');train.extend(episode['features'])
                elif episode['split']=='validation':
                    self.assertGreaterEqual(min(dates),'2021-01-01');self.assertLess(max(dates),'2023-12-31')
                else:self.assertGreaterEqual(min(dates),'2024-01-01')
            expected=Normalizer.fit(data['names'],train)
            self.assertEqual(list(expected.means),data['normalizer']['means'])

    def test_forward_split_continuity(self):
        bars=[bar(f'2020-03-{i:02d}') for i in range(1,29)]
        bars+= [bar('2020-04-'+f'{i:02d}') for i in range(1,29)]
        split=replace(bars[-1],open=50,high=50.5,low=49.5,close=50,split=2,volume=2000)
        rows,_=assemble(bars[:-1]+[split])
        self.assertAlmostEqual(rows[-1]['values'][0],0.)
        self.assertAlmostEqual(rows[-1]['atr'],1.)
    def test_accumulator_resume(self):
        from rl_stocks.data.accumulation.backfill import accumulate
        provider=SyntheticProvider()
        with tempfile.TemporaryDirectory() as d:
            first=accumulate(provider,self.request,d,False)
            def fail(request):raise AssertionError('cached chunks must not refetch')
            provider.bars=fail
            second=accumulate(provider,self.request,d,False)
            self.assertEqual(first,second)
    def test_mixed_feed_rejected(self):
        with self.assertRaises(ValueError):validate([self.bars[0],replace(self.bars[1],feed='different')])
    def test_dependency_audit_rejects_version_drift(self):
        from unittest.mock import patch
        from rl_stocks.training.dependencies import EXPECTED,verify_environment
        with patch('importlib.metadata.version',side_effect=lambda p:EXPECTED[p]):
            self.assertEqual(verify_environment(),EXPECTED)
        with patch('importlib.metadata.version',side_effect=lambda p:'0.0.0' if p=='numpy' else EXPECTED[p]):
            with self.assertRaises(RuntimeError):verify_environment()

    def test_normalizer_bad_fit(self):
        with self.assertRaises(ValueError):Normalizer.fit(['a','b'],[[1],[2]])
        with self.assertRaises(ValueError):Normalizer.fit(['a'],[[float('nan')]])

class AccountingTests(unittest.TestCase):
    def test_long_short_pnl_and_reversal(self):
        p=Portfolio();p.trade(100,100,'t','entry',0,0)
        self.assertEqual(p.equity(110),101000)
        p.trade(0,110,'t','exit',0,0)
        p.trade(-100,110,'t','short',0,0)
        self.assertEqual(p.equity(100),102000)
        p.trade(100,100,'t','reverse',0,0)
        self.assertEqual(p.quantity,100)
        self.assertEqual(p.basis,100)
    def test_costs(self):
        p=Portfolio();p.trade(100,100,'t','entry',1,2)
        self.assertLess(p.equity(100),100000)
        self.assertAlmostEqual(p.costs,3.0002)
    def test_split_and_dividend(self):
        p=Portfolio();p.trade(101,100,'t','entry',0,0)
        p.corporate_action(bar(op=50,high=51,low=49,close=50,split=2,dividend=1))
        self.assertEqual(p.quantity,202);self.assertEqual(p.basis,50)
        self.assertEqual(p.equity(50),100202)
    def test_short_dividend_and_borrow(self):
        p=Portfolio();p.trade(-100,100,'t','entry',0,0)
        p.corporate_action(bar(dividend=1))
        self.assertEqual(p.equity(100),99900)
        self.assertGreater(p.borrow(100),0)
    def test_repeat_target_no_add(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,1,1,1],bar('2020-01-01'),bar(),2)
        first=sim.portfolio.quantity
        sim.advance([.5,1,1,1],bar(),bar('2020-01-03'),2)
        self.assertEqual(sim.portfolio.quantity,first)
        self.assertEqual(len(sim.portfolio.events),1)
    def test_geometry_both_directions(self):
        for sign in (-1,1):
            d=decode([sign,0,0,0],'SPY','t','u',100,2,100000)
            self.assertTrue(valid_entry(d,100))
            self.assertFalse(valid_entry(d,120 if sign==1 else 80))
        with self.assertRaises(ValueError):decode([float('nan'),0,0,0],'SPY','t','u',100,2,100000)
    def test_exposure_cap_and_whole_shares(self):
        sim=Simulator()
        result=sim.advance([1,1,1,1],bar('2020-01-01'),bar(),2)
        self.assertLessEqual(abs(result['exposure']),1)
        self.assertIsInstance(result['quantity'],int)
        self.assertGreaterEqual(result['cash'],0)
    def test_gap_reject(self):
        sim=Simulator()
        r=sim.advance([.5,-1,-1,-1],bar('2020-01-01'),bar(op=110,high=112,low=109,close=111),2)
        self.assertEqual(r['rejected'],'gap_invalid_geometry');self.assertEqual(r['quantity'],0)
    def test_gap_stop_existing(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,1,1,1],bar('2020-01-01'),bar(),2)
        r=sim.advance([.5,1,1,1],bar(),bar('2020-01-03',op=80,high=85,low=79,close=82),2)
        self.assertEqual(r['quantity'],0)
        self.assertEqual(sim.portfolio.events[-1]['price'],80)
    def test_invalid_signal_geometry_preserves_gap_protection(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,1,1,1],bar('2020-01-01'),bar(),2)
        result=sim.advance([.5,1,1,1],bar(),bar('2020-01-03',op=80,high=85,low=79,close=82),50)
        self.assertEqual(result['quantity'],0)
        self.assertEqual(result['rejected'],'nonpositive_protective_price')
        self.assertEqual(sim.portfolio.events[-1]['price'],80)

    def test_stop_first_race(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,0,0,0],bar('2020-01-01'),bar(high=120,low=80),2)
        self.assertEqual(sim.portfolio.events[-1]['reason'],'stop')
    def test_partial_both_boundaries(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,1,-1,-1],bar('2020-01-01'),bar(high=101.2,low=99.5),2)
        self.assertEqual(sim.portfolio.quantity,250)
        self.assertEqual(sim.portfolio.events[-1]['reason'],'target_near')
        sim.protect(bar('2020-01-03',op=101.2,high=102,low=100,close=101.5))
        self.assertEqual(sim.portfolio.quantity,0)
        self.assertEqual(sim.portfolio.events[-1]['reason'],'target_far')
    def test_short_partial_zone(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0,borrow_rate=0))
        sim.advance([-.5,1,-1,-1],bar('2020-01-01'),bar(high=100.5,low=98.8),2)
        self.assertEqual(sim.portfolio.quantity,-250)
        sim.protect(bar('2020-01-03',op=98.8,high=99,low=98,close=98.5))
        self.assertEqual(sim.portfolio.quantity,0)
    def test_holding_limit(self):
        sim=Simulator(SimulationConfig(max_holding=1))
        r=sim.advance([.5,1,1,1],bar('2020-01-01'),bar(),2)
        self.assertEqual(r['quantity'],0);self.assertEqual(sim.portfolio.events[-1]['reason'],'holding_limit')
    def test_availability_guard(self):
        sim=Simulator()
        with self.assertRaises(ValueError):sim.advance([0,0,0,0],replace(bar('2020-01-01'),available_time='2020-01-03T21:00:00+00:00'),bar(),2)

    def test_tiny_exposure_rounds_flat_and_can_close_through_gap(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,1,1,1],bar('2020-01-01'),bar(),2)
        result=sim.advance([.00001,-1,-1,-1],bar(),bar('2020-01-03',op=110,high=111,low=109,close=110),2)
        self.assertEqual(result['quantity'],0)
        self.assertEqual(result['decision']['side'],'flat')
        self.assertIsNone(result['rejected'])

    def test_simulator_reversal_closes_first(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0,borrow_rate=0))
        sim.advance([.5,1,1,1],bar('2020-01-01'),bar(),2)
        sim.advance([-.5,1,1,1],bar(),bar('2020-01-03'),2)
        self.assertEqual(sim.portfolio.quantity,-500)
        self.assertEqual(sim.portfolio.events[-2]['reason'],'reverse_close')
    def test_split_preserves_target(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,1,1,1],bar('2020-01-01'),bar(),2)
        r=sim.advance([.5,1,1,1],bar(),bar('2020-01-03',op=50,high=50.5,low=49.5,close=50,split=2),2)
        self.assertEqual(r['quantity'],1000)
        self.assertEqual(len(sim.portfolio.events),1)
        self.assertEqual(r['equity'],100000)
    def test_gap_partial_does_not_refill_same_bar(self):
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        sim.advance([.5,1,-1,1],bar('2020-01-01'),bar(high=100.5,low=99.5),2)
        sim.advance([.5,1,-1,1],bar(),bar('2020-01-03',op=101.2,high=101.3,low=100.5,close=101),2)
        self.assertEqual(sim.portfolio.quantity,250)

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.now=0.;self.path=Path(self.temp.name)/'ledger.json'
        self.ledger=Ledger(self.path,lambda:self.now)
    def test_parse(self):
        self.assertEqual(parse_usage('Current balance: 150.00 compute units\nUsage rate: 5.25/hr\nActive assignments: 1'),Usage(150,5.25,1))
        with self.assertRaises(BudgetStop):parse_usage('unknown')
    def test_idle_estimate_and_measured_delay(self):
        self.ledger.begin('s',Usage(150,0,0))
        self.now=60;self.ledger.observe(Usage(150,5,1))
        self.assertGreater(self.ledger.value['consumed'],0)
        self.now=120;self.ledger.observe(Usage(149,5,1))
        self.assertEqual(self.ledger.value['consumed'],1)
    def test_stop_no_restart(self):
        self.ledger.begin('s',Usage(150,0,0))
        self.now=60;self.ledger.observe(Usage(21,100,1))
        with self.assertRaises(BudgetStop):self.ledger.guard(60)
        self.ledger.finish(True)
        persisted=Ledger(self.path,lambda:self.now)
        with self.assertRaises(BudgetStop):persisted.begin('s2',Usage(150,0,0))
    def test_two_hour_deadline(self):
        self.ledger.begin('s',Usage(150,0,0));self.now=7190
        with self.assertRaises(BudgetStop):self.ledger.guard(20)
    def test_accumulation_retry(self):
        self.ledger.begin('s',Usage(150,0,0));self.now=60
        self.ledger.observe(Usage(148,5,1));self.ledger.finish(True)
        prior=self.ledger.value['consumed']
        other=Ledger(self.path,lambda:self.now);other.begin('s2',Usage(148,0,0))
        self.now=120;other.observe(Usage(147,5,1))
        self.assertEqual(other.value['consumed'],prior+1)
    def test_existing_runtime_refused(self):
        with self.assertRaises(BudgetStop):self.ledger.begin('s',Usage(150,5,1))
    def test_cleanup_verifies_backend(self):
        job=ColabJob.__new__(ColabJob);job.session='s';job.command=['python','colab_safe.py'];job.ledger=self.ledger
        self.ledger.begin('s',Usage(150,0,0));calls=[]
        def short(args,timeout=25):
            calls.append(args[0]);return 'No active sessions found' if args[0]=='sessions' else 'stopped'
        job.short=short;job.meter=lambda:Usage(150,0,0)
        self.assertTrue(job.cleanup());self.assertEqual(calls,['stop','sessions'])
        self.assertEqual(self.ledger.active['status'],'stopped')
    def test_cleanup_failure_retains_recovery(self):
        job=ColabJob.__new__(ColabJob);job.session='s';job.command=['python','colab_safe.py'];job.ledger=self.ledger
        self.ledger.begin('s',Usage(150,0,0))
        job.short=lambda args,timeout=25:'[s] endpoint | Hardware: A100' if args[0]=='sessions' else 'ok'
        job.meter=lambda:Usage(150,5,1)
        self.assertFalse(job.cleanup());self.assertEqual(self.ledger.active['status'],'cleanup_failed')
        with self.assertRaises(BudgetStop):self.ledger.begin('new',Usage(150,0,0))

    def test_meter_failure_stops_operation(self):
        job=ColabJob.__new__(ColabJob);job.ledger=self.ledger
        self.ledger.begin('s',Usage(150,0,0));job.last_meter=0.;job.allocated=True
        def bad_meter():raise OSError('network unavailable')
        job.meter=bad_meter
        with self.assertRaises(BudgetStop):job.tick()

    def test_reported_zero_rate_is_allowed_only_for_cpu(self):
        job=ColabJob(self.temp.name,'python','CPU');job.ledger=self.ledger
        self.ledger.begin('cpu',Usage(150,0,0));job.last_meter=0.;job.allocated=True
        job.meter=lambda:Usage(150,0,1)
        job.tick()
        self.assertFalse(self.ledger.value['budget_stopped'])
        job.accelerator='T4';job.last_meter=0.
        with self.assertRaises(BudgetStop):job.tick()
    def test_cleanup_on_provision_failure(self):
        import contextlib
        import io
        (Path(self.temp.name)/'requirements.lock').write_text('test preflight fixture')
        class FailedJob(ColabJob):
            def short(self,args,timeout=25):
                return 'No active sessions found' if args[0]=='sessions' else 'help'
            def meter(self):return Usage(150,0,0)
            def operation(self,args,timeout):raise TimeoutError('provision failure')
        job=FailedJob(self.temp.name,'python')
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):job.run(Path(self.temp.name),'ppo')
        self.assertEqual(job.ledger.active['status'],'stopped')

class ReportingTests(unittest.TestCase):
    def test_offline_report_generation(self):
        from rl_stocks.reporting.html import report
        from rl_stocks.evaluation.runner import baselines
        from rl_stocks.evaluation.metrics import metrics
        bars=[bar('2020-01-01'),bar(),bar('2020-01-03',high=110,close=105)]
        sim=Simulator(SimulationConfig(fee_bps=0,impact_bps=0))
        rows=[]
        for previous,current in zip(bars,bars[1:]):
            r=sim.advance([.5,1,1,1],previous,current,2)
            r.update(session=current.session,price=current.close);rows.append(r)
        result={'SPY-validation':{'rows':rows,'events':sim.portfolio.events,
            'metrics':metrics(rows,sim.portfolio.events),'baselines':baselines({'bars':[asdict(b) for b in bars]},sim.config),
            'split':'validation','symbol':'SPY'}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.html'
            report(result,path,'Offline wiring audit')
            self.assertGreater(path.stat().st_size,1000)
            self.assertTrue(path.with_suffix('.json').exists())
            self.assertIn('Equity',path.read_text())

    def test_comparison_includes_failures_and_matching_groups(self):
        from rl_stocks.reporting.comparison import compare
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i,(status,seed,score) in enumerate([('complete',11,.1),('complete',23,.2),('failed',37,None)]):
                path=root/'runs'/str(i);path.mkdir(parents=True)
                data={'algorithm':'ppo','status':status,'seed':seed,'dataset':'same',
                    'simulation':{'fee_bps':1},'requested_steps':512,'smoke':True,
                    'selection_metric':'mean_validation_net_return'}
                if score is not None:data['selected_score']=score
                (path/'metadata.json').write_text(json.dumps(data))
            target=compare(root/'runs',root/'comparison.html')
            data=json.loads(target.with_suffix('.json').read_text())
            self.assertEqual(len(data['runs']),3)
            self.assertEqual(data['summaries'][0]['failed'],1)
            self.assertAlmostEqual(data['summaries'][0]['mean_validation_score'],.15)
            self.assertGreater(data['summaries'][0]['seed_standard_deviation'],0)

if __name__=='__main__':unittest.main()
