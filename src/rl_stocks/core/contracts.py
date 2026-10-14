from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Protocol, Sequence

UNIVERSE = ("SPY", "QQQ", "IWM", "DIA")
ACTION_SCHEMA = "target-zone-v1"

class RecordType(str, Enum):
    BARS = "bars"
    TRADES = "trades"
    QUOTES = "quotes"
    SECTORS = "sectors"
    FILINGS = "filings"
    OPTIONS = "options"
    NEWS = "news"

@dataclass(frozen=True)
class Capabilities:
    provider: str
    records: tuple[RecordType, ...] = (RecordType.BARS,)
    intervals: tuple[str, ...] = ("1d",)
    earliest: str | None = None
    feed: str = "research"
    adjustments: str = "raw_with_actions"
    limitations: str = ""
    def require(self, record: RecordType, interval: str = "1d") -> None:
        if record not in self.records or interval not in self.intervals:
            raise ValueError(f"unsupported {record}/{interval} for {self.provider}")

@dataclass(frozen=True)
class Request:
    symbols: tuple[str, ...] = UNIVERSE
    start: date = date(2010, 1, 1)
    end: date = date(2026, 10, 1)  # exclusive: includes September 2026
    def __post_init__(self):
        if self.start >= self.end or not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("invalid date range or symbols")

@dataclass(frozen=True)
class Bar:
    symbol: str
    session: str
    open_time: str
    event_time: str
    available_time: str
    retrieved_time: str
    provider: str
    feed: str
    revision: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    dividend: float = 0.0
    split: float = 1.0  # new shares / old shares, effective at open
    adjustments: str = "raw_with_actions"

class BarProvider(Protocol):
    capabilities: Capabilities
    def bars(self, request: Request) -> Sequence[Bar]: ...

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def bar_dicts(bars):
    return [asdict(b) for b in bars]
