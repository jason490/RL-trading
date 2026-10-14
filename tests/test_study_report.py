from copy import deepcopy
import unittest
from rl_stocks.reporting.study import summarize


class StudyReportTests(unittest.TestCase):
    def setUp(self):
        self.protocol={"datasets":{"baseline":"a","combined":"b"},"algorithms":["ppo","sac"],
                       "seeds":[11,23,37],"transitions_per_run":50000}
        self.runs=[{"dataset":dataset,"algorithm":algorithm,"seed":seed,"status":"complete",
                    "actual_steps":50000,"simulation":{"fee_bps":1},"selection_metric":"net",
                    "source_hash":"fixed-source","versions":{"torch":"fixed"},
                    "selected_score":float(dataset=="b"),"bundle_id":f"{dataset}-{algorithm}-{seed}"}
                   for dataset in ("a","b") for algorithm in ("ppo","sac") for seed in (11,23,37)]

    def test_predeclared_family_and_seed_ties(self):
        result=summarize(self.runs,self.protocol)
        self.assertEqual(result['candidate']['dataset'],'combined')
        self.assertEqual(result['candidate']['algorithm'],'ppo')
        self.assertEqual(result['candidate']['seed'],11)
        self.assertFalse(result['test_evaluated'])

    def test_rejects_missing_duplicate_and_unmatched_runs(self):
        variants=[self.runs[:-1],self.runs+[self.runs[0]]]
        for field in ('simulation','source_hash','versions'):
            runs=deepcopy(self.runs);runs[-1][field]='different';variants.append(runs)
        for runs in variants:
            with self.assertRaises(ValueError):summarize(runs,self.protocol)
