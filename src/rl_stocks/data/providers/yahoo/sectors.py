"""Current metadata is a dated snapshot, never a historical classification."""
from rl_stocks.core.contracts import Capabilities, RecordType, utcnow

ALIASES = {
    "Communication Services": "communication_services",
    "Consumer Cyclical": "consumer_discretionary",
    "Consumer Defensive": "consumer_staples", "Energy": "energy",
    "Financial Services": "financials", "Healthcare": "health_care",
    "Industrials": "industrials", "Basic Materials": "materials",
    "Real Estate": "real_estate", "Technology": "technology", "Utilities": "utilities",
}


class YahooSectors:
    capabilities = Capabilities("yahoo", records=(RecordType.SECTORS,), feed="yahoo",
                                limitations="Current snapshot only; no historical sector vintages")

    def records(self, symbols, ticker_factory=None):
        if ticker_factory is None:
            import yfinance as yf
            ticker_factory = yf.Ticker
        result = []
        for symbol in symbols:
            info = ticker_factory(symbol).get_info()
            sector = info.get("sector")
            if sector not in ALIASES:
                raise ValueError(f"missing/unsupported Yahoo sector for {symbol}")
            retrieved = utcnow()
            result.append({"symbol": symbol, "sector": ALIASES[sector], "source_sector": sector,
                           "available_time": retrieved, "retrieved_time": retrieved,
                           "effective_from": retrieved[:10], "effective_to": None,
                           "source": "yahoo-current-info", "revision": retrieved})
        return result
