import json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from rl_stocks.runtime.colab.transport import connection_lost,worker_not_started,TransportLost
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
                with patch('rl_stocks.runtime.colab.runner.worker_not_started',return_value=safe):job.execute_worker()
                self.assertEqual(len(calls),count)
        with tempfile.TemporaryDirectory() as directory:
            job=ColabJob(directory,'unused','CPU');calls=[]
            def operation(*args):calls.append(args);raise BudgetStop('halt')
            job.operation=operation
            with self.assertRaises(BudgetStop):job.execute_worker()
            self.assertEqual(len(calls),1)

    def test_bounded_tail_does_not_match_ordinary_log_text(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'log';p.write_text('Expected RuntimeError: Connection was lost.\n')
            self.assertFalse(connection_lost(p))
            p.write_text('x'*200000+'\n\x1b[31mRuntimeError: Connection was lost.\x1b[0m\n')
            self.assertTrue(connection_lost(p))
