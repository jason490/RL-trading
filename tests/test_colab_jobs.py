import unittest

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
