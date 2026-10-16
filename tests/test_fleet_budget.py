import tempfile
import unittest
from pathlib import Path
from rl_stocks.runtime.colab.budget import Usage,BudgetStop
from rl_stocks.runtime.colab.fleet_budget import FleetBudget


class FleetBudgetTests(unittest.TestCase):
    def test_overlapping_account_debits_are_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            now=[0.];b=FleetBudget(Path(directory)/'ledger.json',clock=lambda:now[0])
            b.start('fleet',Usage(150,0,0),5)
            for i,hardware in enumerate(('CPU','T4','A100','CPU','CPU')):b.admit(str(i),hardware)
            now[0]=3600;b.observe(Usage(148,2,5))
            self.assertAlmostEqual(b.ledger.value['consumed'],2.01)
            # A second reading of the same shared debit cannot multiply it by five.
            b.observe(Usage(148,2,5));self.assertAlmostEqual(b.ledger.value['consumed'],2.01)
            now[0]=5400;b.observe(Usage(147,1,4))
            self.assertAlmostEqual(b.ledger.value['consumed'],3.015)
            for i in range(5):b.finish_child(str(i),True)
            self.assertTrue(b.finish())
            b.start('next',Usage(147,0,0),1)
            self.assertAlmostEqual(b.active['base_consumed'],3.015)

    def test_child_deadline_and_global_working_budget_halt_everyone(self):
        with tempfile.TemporaryDirectory() as directory:
            now=[0.];b=FleetBudget(Path(directory)/'ledger.json',clock=lambda:now[0])
            b.start('fleet',Usage(150,0,0),3);b.admit('a','CPU');now[0]=100;b.admit('b','CPU')
            now[0]=7141
            with self.assertRaises(BudgetStop):b.guard('a',60)
            self.assertTrue(b.ledger.value['budget_stopped'])
            with self.assertRaises(BudgetStop):b.admit('c','CPU')
            with self.assertRaises(BudgetStop):b.guard('b',1)
        with tempfile.TemporaryDirectory() as directory:
            b=FleetBudget(Path(directory)/'ledger.json');b.start('fleet',Usage(150,0,0),2);b.admit('a','CPU')
            b.ledger.value['consumed']=129.99;b.active['rate']=5.
            with self.assertRaises(BudgetStop):b.guard('a',60)
        with tempfile.TemporaryDirectory() as directory:
            b=FleetBudget(Path(directory)/'ledger.json');b.start('fleet',Usage(150,0,0),20);b.admit('a','A100')
            b.ledger.value['consumed']=50.;b.active['rate']=1000.
            # Even a one-second next operation must leave room for fleet teardown.
            with self.assertRaises(BudgetStop):b.guard('a',1)

    def test_unmonitored_or_unresolved_resources_prevent_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            b=FleetBudget(Path(directory)/'ledger.json');b.start('fleet',Usage(150,0,0),4);b.admit('owned','CPU')
            with self.assertRaises(BudgetStop):b.observe(Usage(150,.08,2))
            b.finish_child('owned',True);self.assertTrue(b.finish())
            with self.assertRaises(BudgetStop):b.start('retry',Usage(150,0,0),1)
        with tempfile.TemporaryDirectory() as directory:
            b=FleetBudget(Path(directory)/'ledger.json');b.start('fleet',Usage(150,0,0),1);b.admit('owned','CPU')
            b.finish_child('owned',False);self.assertFalse(b.finish())
            with self.assertRaises(BudgetStop):b.start('retry',Usage(150,0,0),1)
