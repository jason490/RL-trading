import os
from datetime import datetime, timedelta, timezone
from rl_stocks.core.contracts import Bar, Capabilities, Request, utcnow
from rl_stocks.data.validation.bars import validate

class AlpacaProvider:
    def __init__(self, feed="iex", client=None):
        if feed not in ("iex", "sip"):
            raise ValueError("explicit iex/sip feed required")
        self.feed, self.client = feed, client
        self.capabilities = Capabilities("alpaca", earliest="2016-01-01", feed=feed,
            adjustments="split_adjusted_with_dividends",
            limitations="Account entitlement and corporate-action cash-flow coverage must be audited")
    def bars(self, request: Request):
        # This adapter returns split adjusted bars. Alpaca bar endpoint omits dividends;
        # require an audited dividend archive before creating a training dataset.
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment, DataFeed
        import exchange_calendars as xc
        client = self.client or StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        if self.client is None:
            from rl_stocks.data.providers.alpaca.transport import BoundedSession
            # Isolated private extension verified against alpaca-py 0.42.0.
            client._session.close()
            client._session = BoundedSession()
            client._retry = 2
            client._retry_wait = 1
        calendar = xc.get_calendar("XNYS", start=str(request.start), end=str(request.end))
        retrieved = utcnow()
        response = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=list(request.symbols),
            start=datetime.combine(request.start, datetime.min.time(), timezone.utc),
            end=datetime.combine(request.end, datetime.min.time(), timezone.utc),
            timeframe=TimeFrame.Day, feed=DataFeed(self.feed), adjustment=Adjustment.SPLIT))
        self.raw_source=response.model_dump(mode="json")
        result = []
        for symbol in request.symbols:
            for row in response.data.get(symbol, []):
                day = str(row.timestamp.astimezone(__import__('zoneinfo').ZoneInfo("America/New_York")).date())
                op, end = calendar.session_open(day), calendar.session_close(day)
                result.append(Bar(symbol, day, op.isoformat(), end.isoformat(),
                    (end+timedelta(minutes=15)).isoformat(), retrieved, "alpaca", self.feed,
                    retrieved, row.open, row.high, row.low, row.close, row.volume, 0, 1,
                    "split_adjusted_with_dividends"))
        validate(result)
        return result
