import json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from rl_stocks.runtime.colab.transport import connection_lost,worker_not_started,export_ready,TransportLost
from rl_stocks.runtime.colab.runner import ColabJob
from rl_stocks.runtime.colab.budget import Usage,BudgetStop


class TransportTests(unittest.TestCase):
    def test_hung_cli_is_terminated_when_terminal_error_is_printed(self):
        with tempfile.TemporaryDirectory() as directory:
            job=ColabJob(directory,'unused','CPU');job.ledger.begin(job.session,Usage(150,0,0))
            job.command=[sys.executable,'-c',"import time; print('RuntimeError: Connection was lost.',flush=True); time.sleep(120)"]
            job.tick=lambda:None
            started=time.monotonic()
            with self.assertRaises(TransportLost):job.operation(['install'],20)
            self.assertLess(time.monotonic()-started,5)
            self.assertEqual(job.ledger.active['transport_failure'],'install')

    def test_probe_allows_only_confirmed_not_started_and_sanitizes_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'.runtime';p.mkdir()
            (p/'sessions.json').write_text(json.dumps({'owned':{'url':'https://example.invalid','token':'secret'}}))
            for status,expected in [(404,True),(200,False),(401,False),(500,False)]:
                class Response:status_code=status
                self.assertEqual(worker_not_started(directory,'owned',lambda *a,**k:Response()),expected)
            def error(*a,**k):raise RuntimeError('https://secret.invalid?token=secret')
            self.assertFalse(worker_not_started(directory,'owned',error))

    def test_prestart_retry_never_reexecutes_started_or_unknown_workers(self):
        for safe,count in [(True,2),(False,1)]:
            with tempfile.TemporaryDirectory() as directory:
                job=ColabJob(directory,'unused','CPU');job.ledger.begin(job.session,Usage(150,0,0));calls=[]
                def operation(*args):calls.append(args);raise TransportLost('lost')
                job.operation=operation
                with patch('rl_stocks.runtime.colab.runner.worker_not_started',return_value=safe),patch.object(job,'await_export') as waiting:
                    if safe:
                        with self.assertRaises(TransportLost):job.execute_worker()
                    else:job.execute_worker()
                    self.assertEqual(waiting.call_count,0 if safe else 1)
                self.assertEqual(len(calls),count)
        with tempfile.TemporaryDirectory() as directory:
            job=ColabJob(directory,'unused','CPU');calls=[]
            def operation(*args):calls.append(args);raise BudgetStop('halt')
            job.operation=operation
            with self.assertRaises(BudgetStop):job.execute_worker()
            self.assertEqual(len(calls),1)

    def test_export_probe_rejects_outages_and_redacts_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'.runtime';p.mkdir()
            (p/'sessions.json').write_text(json.dumps({'owned':{'url':'https://example.invalid','token':'secret'}}))
            for status,expected in [(404,False),(200,True)]:
                class Response:status_code=status
                self.assertEqual(export_ready(directory,'owned',lambda *a,**k:Response()),expected)
            for status in (401,403,500):
                class Response:status_code=status
                with self.assertRaisesRegex(TransportLost,'^remote export status unavailable$'):
                    export_ready(directory,'owned',lambda *a,**k:Response())
            def error(*a,**k):raise RuntimeError('https://secret.invalid?token=secret')
            with self.assertRaisesRegex(TransportLost,'^remote export status unavailable$'):
                export_ready(directory,'owned',error)

    def test_lost_stream_waits_with_metering_and_checkpoint_download(self):
        with tempfile.TemporaryDirectory() as directory:
            job=ColabJob(directory,'unused','CPU');job.ledger.begin(job.session,Usage(150,0,0))
            with patch.object(job,'tick') as tick,patch.object(job,'short') as download,patch('rl_stocks.runtime.colab.runner.export_ready',side_effect=[False,True]),patch('rl_stocks.runtime.colab.runner.time.sleep'):
                job.await_export(time.monotonic()+100)
            self.assertEqual(tick.call_count,2)
            self.assertEqual(download.call_count,1)
            self.assertTrue(job.ledger.active['remote_export_verified'])
            self.assertEqual(len(job.ledger.active['retrieved']),1)

    def test_export_wait_preserves_deadline_and_budget_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            job=ColabJob(directory,'unused','CPU');job.ledger.begin(job.session,Usage(150,0,0))
            with self.assertRaises(TimeoutError):job.await_export(time.monotonic()+19)
            with patch.object(job,'tick',side_effect=BudgetStop('metering unavailable')),patch('rl_stocks.runtime.colab.runner.export_ready') as probe:
                with self.assertRaises(BudgetStop):job.await_export(time.monotonic()+100)
                probe.assert_not_called()

    def test_bounded_tail_does_not_match_ordinary_log_text(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'log';p.write_text('Expected RuntimeError: Connection was lost.\n')
            self.assertFalse(connection_lost(p))
            p.write_text('x'*200000+'\n\x1b[31mRuntimeError: Connection was lost.\x1b[0m\n')
            self.assertTrue(connection_lost(p))
