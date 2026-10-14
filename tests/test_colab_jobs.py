import unittest
import json
from pathlib import Path
import tempfile

from rl_stocks.runtime.colab.jobs import job_plan


class JobPlanTests(unittest.TestCase):
    def test_smoke_gate_precedes_research(self):
        plan = job_plan("ppo")
        self.assertEqual([j["steps"] for j in plan["jobs"]], [512, 512, 1024, 1024, 50000])
        self.assertTrue(all(j["smoke"] for j in plan["jobs"][:-1]))
        self.assertFalse(plan["jobs"][-1]["smoke"])
        self.assertEqual(plan["jobs"][0]["device"], "cpu")
        self.assertEqual(plan["seed"], 11)

    def test_default_does_not_start_long_experiments(self):
        self.assertEqual(len(job_plan()["jobs"]), 4)
        with self.assertRaises(ValueError):
            job_plan("unknown")

    def test_uploaded_git_provenance_survives_without_checkout(self):
        from rl_stocks.core.provenance import repository_state
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source-git.json"
            expected = {"revision": "a" * 40, "dirty": ""}
            source.write_text(json.dumps(expected))
            self.assertEqual(repository_state(directory), expected)
            source.write_text('{"token":"unexpected"}')
            with self.assertRaises(ValueError):
                repository_state(directory)

    def test_bootstrap_uses_colab_interpreter_without_downgrade(self):
        from rl_stocks.runtime.colab.jobs import bootstrap_commands
        commands = bootstrap_commands('uv', '/content/research', '/usr/bin/python3.13')
        self.assertEqual(commands[0], ['uv', 'venv', '--python', '/usr/bin/python3.13',
                                       '/content/research/.venv'])
        self.assertFalse(any('install' in command[:2] or 'python' == command[1]
                             for command in commands[:-1]))
        self.assertEqual(commands[1][4], '/content/research/.venv/bin/python')
        self.assertEqual(commands[-1], ['/content/research/.venv/bin/python',
                                       '/content/research/scripts/remote_train.py'])
