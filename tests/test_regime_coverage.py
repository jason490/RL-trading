import copy,unittest
from rl_stocks.reporting.regimes import regime_coverage

class RegimeCoverageTests(unittest.TestCase):
    def test_future_validation_cannot_change_training_diagnostics(self):
        bars=[{'session':f'2020-01-{i+1:02d}','close':price,'split':1,'adjustments':'split_adjusted_with_dividends'} for i,price in enumerate([10,10,8,6,4,3])]
        dataset={'episodes':{'train':{'symbol':'A','split':'train','bars':bars},'validation':{'symbol':'A','split':'validation','bars':[]}}}
        universe={'survivorship':'surviving sample','selected_on':'2026-10-14'}
        first=regime_coverage(dataset,universe,3,1)
        self.assertEqual(first['records'][0]['longest_consecutive_declining_both_windows'],3)
        dataset['episodes']['validation']['bars']=[{'close':1000000}]
        self.assertEqual(first,regime_coverage(dataset,universe,3,1))
        self.assertFalse(first['historical_membership_verified'])
        bad=copy.deepcopy(dataset);bad['episodes']['train']['bars'][-1]['session']='2021-01-01'
        with self.assertRaises(ValueError):regime_coverage(bad,universe,3,1)
