import csv,json,tempfile,unittest
from pathlib import Path
from rl_stocks.training.curves import curve_record,append_record,latest_points,render_curve


class CurveTests(unittest.TestCase):
    def result(self,score):
        return {'A':{'metrics':{'net_return':score,'max_drawdown':-.1,'sharpe':.5,
                    'costs':100.,'turnover':2.},'baselines':{'buy_hold':{'metrics':{'net_return':.3}}}}}

    def test_percentage_points_anchor_and_post_update_replacement(self):
        history=[]
        for step,score,phase in [(40000,.1,'on_step'),(50000,.2,'on_step'),
                                 (50000,.19,'post_update'),(100000,.24,'post_update')]:
            history.append(curve_record(self.result(score),history,step,phase,1.))
        self.assertEqual(len(latest_points(history)),3)
        self.assertAlmostEqual(history[-1]['change_pp'],5.)
        self.assertAlmostEqual(history[-1]['gain_since_50k_pp'],5.)
        self.assertAlmostEqual(history[-1]['excess_return_pp'],-6.)
        self.assertAlmostEqual(history[-1]['equity_change_pct_from_previous'],100*(1.24/1.19-1))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'curve.jsonl'
            for point in history:append_record(path,point)
            self.assertEqual(len(path.read_text().splitlines()),4)
            render_curve(history,Path(directory)/'curve.html')
            with path.with_suffix('.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertIn('Validation learning curve',path.with_suffix('.html').read_text())

    def test_nonfinite_scores_are_rejected_before_serialization(self):
        with self.assertRaises(ValueError):curve_record(self.result(float('nan')),[],1,'on_step',0.)
