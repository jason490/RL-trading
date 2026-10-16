import copy
import tempfile
import unittest
from rl_stocks.runtime.colab.jobs import stability_plan,validate_plan
from rl_stocks.runtime.colab.runner import ColabJob


class StabilityJobTests(unittest.TestCase):
    def test_single_smoke_is_cpu_and_has_exact_algorithm_budget(self):
        from rl_stocks.runtime.colab.jobs import single_smoke_plan
        for algorithm,steps in [('ppo',512),('recurrent_ppo',512),('sac',1024),('td3',1024)]:
            plan=single_smoke_plan(algorithm);self.assertEqual(validate_plan(plan),plan)
            self.assertEqual(plan['jobs'][0]['steps'],steps)
            self.assertEqual(plan['accelerator'],'CPU')
            plan['jobs'][0]['steps']=100000
            with self.assertRaises(ValueError):validate_plan(plan)
    def test_fixed_cpu_plan_and_tampering_rejected(self):
        for algorithm in ('ppo','sac','recurrent_ppo'):
            for seed in (11,23,37):
                p=stability_plan(algorithm,seed);self.assertEqual(validate_plan(p),p)
                self.assertEqual(p['jobs'][0]['milestones'],[100000])
                self.assertEqual(p['jobs'][0]['evaluation_every'],10000)
                for key,value in [('device','cuda'),('steps',250000),('reward_scale',10.)]:
                    bad=copy.deepcopy(p);bad['jobs'][0][key]=value
                    with self.assertRaises(ValueError):validate_plan(bad)
        for algorithm,seed in [('td3',11),('ppo',True),('ppo',19)]:
            with self.assertRaises(ValueError):stability_plan(algorithm,seed)

    def test_conflicting_or_gpu_probe_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            for accelerator,options in [('T4',{}),('CPU',{'learning':{'seed':11}}),('CPU',{'study':True})]:
                job=ColabJob(directory,'unused',accelerator)
                with self.assertRaises(ValueError):job.run('unused',stability={'algorithm':'ppo','seed':11},**options)
                self.assertEqual(job.ledger.value['attempts'],[])
