import unittest
from rl_stocks.evaluation.behavior import behavior


class BehaviorTests(unittest.TestCase):
    def test_protective_reentry_distinguishes_reversal_and_costs(self):
        rows=[{'session':f'2020-01-0{i}','equity':100000.,'previous_equity':100000.,
               'quantity':10,'borrow_cost':1.} for i in (2,3,4)]
        def event(day,quantity,delta,reason):
            return {'time':f'2020-01-0{day}T15:00:00Z','quantity':quantity,'delta':delta,
                    'reason':reason,'fee':2.,'impact':3.}
        events=[event(2,10,10,'add'),event(2,0,-10,'stop'),event(3,10,10,'add'),
                event(3,0,-10,'reverse_close'),event(3,-10,-10,'reverse')]
        result=behavior({'rows':rows,'events':events,'metrics':{'costs':28.,'net_return':0.,
                         'max_drawdown':0.,'turnover':1.}})
        self.assertEqual(result['protective_reentries_within_one_session'],1)
        self.assertEqual(result['entries'],3)
        self.assertEqual(result['total_decomposed_costs'],28.)
        self.assertEqual(result['mean_abs_log_reward'],0.)
