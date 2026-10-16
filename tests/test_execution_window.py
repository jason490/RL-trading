import unittest
from rl_stocks.runtime.colab.budget import BudgetStop,execution_window

class ExecutionWindowTests(unittest.TestCase):
    def test_setup_and_reconnect_keep_retrieval_shutdown_reserve(self):
        self.assertEqual(execution_window(7100,6600),6499)
        self.assertEqual(execution_window(6800,6600),6199)
        self.assertEqual(execution_window(7100,5400),5400)
        for remaining in (600,660,0,float('nan')):
            with self.assertRaises(BudgetStop):execution_window(remaining,6600)
