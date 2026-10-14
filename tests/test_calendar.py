from datetime import date
import importlib.util
import unittest

from rl_stocks.core.contracts import Request
from rl_stocks.data.validation.bars import exchange_sessions


@unittest.skipUnless(importlib.util.find_spec("exchange_calendars"), "exchange calendar unavailable")
class CalendarTests(unittest.TestCase):
    def test_holiday_start_and_exclusive_weekend_end(self):
        sessions = exchange_sessions(Request(("SPY",), date(2010, 1, 1), date(2010, 1, 9)))
        self.assertEqual(sessions, ["2010-01-04", "2010-01-05", "2010-01-06",
                                    "2010-01-07", "2010-01-08"])

    def test_session_end_is_excluded(self):
        sessions = exchange_sessions(Request(("SPY",), date(2020, 1, 2), date(2020, 1, 3)))
        self.assertEqual(sessions, ["2020-01-02"])
