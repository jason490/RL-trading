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

    def test_notebook_environment_does_not_override_research(self):
        from rl_stocks.runtime.colab.jobs import bootstrap_environment
        parent = {'MPLBACKEND': 'module://matplotlib_inline.backend_inline',
                  'PYTHONHOME': '/old-python', 'VIRTUAL_ENV': '/old-env',
                  'PYTHONPATH': '/notebook/packages', 'CUDA_VISIBLE_DEVICES': '0'}
        environment = bootstrap_environment(parent, '/content/research')
        self.assertEqual(environment['MPLBACKEND'], 'Agg')
        self.assertEqual(environment['PYTHONPATH'], '/content/research/src')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0')
        self.assertNotIn('PYTHONHOME', environment)
        self.assertNotIn('VIRTUAL_ENV', environment)
        self.assertEqual(parent['MPLBACKEND'], 'module://matplotlib_inline.backend_inline')

    def test_lower_credit_preflight_has_no_long_training(self):
        plan=job_plan(accelerator='CPU')
        self.assertEqual([j['steps'] for j in plan['jobs']], [512,1024,1024])
        self.assertTrue(all(j['device']=='cpu' and j['smoke'] for j in plan['jobs']))
        self.assertTrue(all(j['smoke'] for j in job_plan(accelerator='T4')['jobs']))
        with self.assertRaises(ValueError):job_plan('ppo','T4')

    def test_a100_smoke_only_refused_before_allocation(self):
        from rl_stocks.runtime.colab.runner import ColabJob
        with tempfile.TemporaryDirectory() as directory:
            job=ColabJob(directory,'python','A100')
            with self.assertRaisesRegex(ValueError,'full --research'):
                job.run(directory)
            self.assertEqual(job.ledger.value['attempts'],[])

    def test_matched_study_algorithms_seeds_and_validation(self):
        from rl_stocks.runtime.colab.jobs import study_plan,validate_plan
        from rl_stocks.runtime.colab.runner import ColabJob
        plan=study_plan()
        self.assertEqual(len(plan['jobs']),13)
        research=[j for j in plan['jobs'] if not j['smoke']]
        self.assertEqual({(j['algorithm'],j['seed']) for j in research},
                         {(a,s) for a in ('ppo','sac','td3') for s in (11,23,37)})
        self.assertTrue(all(j['steps']==50000 and j['device']=='cuda' for j in research))
        self.assertEqual(validate_plan(plan),plan)
        altered=json.loads(json.dumps(plan));altered['jobs'][-1]['seed']=42
        with self.assertRaises(ValueError):validate_plan(altered)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):ColabJob(directory,'python','CPU').run(directory,study=True)

    def test_bounded_algorithm_studies_keep_identical_seeds_and_budgets(self):
        from rl_stocks.runtime.colab.jobs import study_plan,validate_plan
        for algorithm in ('ppo','sac','td3'):
            plan=study_plan(algorithm)
            self.assertEqual(len(plan['jobs']),3)
            self.assertEqual({j['algorithm'] for j in plan['jobs']},{algorithm})
            self.assertEqual([j['seed'] for j in plan['jobs']],[11,23,37])
            self.assertTrue(all(j['steps']==50000 and not j['smoke'] for j in plan['jobs']))
            self.assertEqual(validate_plan(plan),plan)
