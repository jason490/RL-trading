from datetime import timedelta
from rl_stocks.core.contracts import Bar, Capabilities, Request, utcnow
from rl_stocks.data.validation.bars import validate

class YahooProvider:
    capabilities = Capabilities("yahoo", feed="yahoo", adjustments="split_adjusted_with_dividends",
        limitations="Unofficial personal use; revised history; availability reconstructed with 15-minute lag; no vintage guarantee")
    def bars(self, request: Request):
        import yfinance as yf
        import exchange_calendars as xc
        result = []
        self.raw_source=[]
        retrieved = utcnow()
        calendar = xc.get_calendar("XNYS", start=str(request.start), end=str(request.end))
        for symbol in request.symbols:
            frame = yf.Ticker(symbol).history(start=str(request.start), end=str(request.end),
                interval="1d", auto_adjust=False, back_adjust=False, actions=True,
                repair=False, raise_errors=True, timeout=30)
            if frame.empty:
                raise ValueError(f"no Yahoo bars for {symbol}; no fallback")
            self.raw_source.append({"symbol":symbol,"retrieved_time":retrieved,
                "rows":[{"date":str(index),**{str(k):float(v) for k,v in row.items()}}
                        for index,row in frame.iterrows()]})
            for index, row in frame.iterrows():
                day = str(index.date())
                if not calendar.is_session(day):
                    raise ValueError(f"non-session Yahoo bar: {day}")
                op, end = calendar.session_open(day), calendar.session_close(day)
                # Yahoo OHLC are split adjusted even with auto_adjust=False.
                # Do not apply Stock Splits again. Dividends remain cash flows.
                result.append(Bar(symbol, day, op.isoformat(), end.isoformat(),
                    (end+timedelta(minutes=15)).isoformat(), retrieved, "yahoo", "yahoo",
                    retrieved, float(row.Open), float(row.High), float(row.Low), float(row.Close),
                    float(row.Volume), float(row.get("Dividends", 0)), 1.0,
                    "split_adjusted_with_dividends"))
        validate(result)
        return result
