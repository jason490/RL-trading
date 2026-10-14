import json
from pathlib import Path
import signal
import tempfile
import unittest
from rl_stocks.runtime.colab.control import process_stop_request


class ControlTests(unittest.TestCase):
    def test_exact_owned_parent_receives_one_graceful_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'.runtime';state.mkdir()
            (state/'usage-ledger.json').write_text(json.dumps({'attempts':[{'session':'owned','status':'active'}]}))
            request=state/'stop-owned-runner.json';request.write_text(json.dumps({'session':'owned','reason':'planned batch handoff'}))
            signals=[]
            self.assertTrue(process_stop_request(root,(123,['python','.runtime/run-study-campaign.py'],root),lambda *args:signals.append(args)))
            self.assertEqual(signals,[(123,signal.SIGINT)])
            self.assertFalse(request.exists())
            self.assertFalse(process_stop_request(root,(123,[],root),lambda *args:signals.append(args)))

    def test_other_sessions_parents_and_directories_are_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'.runtime';state.mkdir()
            (state/'usage-ledger.json').write_text(json.dumps({'attempts':[{'session':'owned','status':'active'}]}))
            request=state/'stop-owned-runner.json';request.write_text(json.dumps({'session':'other'}))
            send=lambda *args:self.fail('must not signal another process')
            self.assertFalse(process_stop_request(root,(123,['python','.runtime/run-study-campaign.py'],root),send))
            request.write_text(json.dumps({'session':'owned'}))
            self.assertFalse(process_stop_request(root,(123,['bash'],root),send))
            self.assertFalse(process_stop_request(root,(123,['python','.runtime/run-study-campaign.py'],'/tmp/other'),send))
