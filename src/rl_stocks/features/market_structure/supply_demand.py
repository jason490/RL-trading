"""Incremental OHLCV base/impulse zones; no future lifecycle rewrites."""
from dataclasses import asdict, dataclass
from bisect import bisect_right
import math

from rl_stocks.core.contracts import digest
from rl_stocks.features.price.indicators import atr


@dataclass(frozen=True)
class ZoneConfig:
    atr_period: int = 14
    impulse_atr: float = 1.2
    max_base: int = 3
    base_body_ratio: float = 0.6
    departure_atr: float = 0.8
    max_zones: int = 10

    def __post_init__(self):
        for value in (self.atr_period, self.max_base, self.max_zones):
            if type(value) is not int or value < 1:
                raise ValueError("positive integer zone periods/limits required")
        if self.atr_period < 2 or not 0 < self.base_body_ratio <= 1:
            raise ValueError("invalid zone period/body ratio")
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.impulse_atr, self.departure_atr)):
            raise ValueError("positive finite zone thresholds required")


@dataclass
class Zone:
    identity: str
    kind: str
    bottom: float
    top: float
    formation_event_time: str
    available_time: str
    formation_session: str
    strength: float
    state: str = "fresh"

    @property
    def proximal(self):
        return self.top if self.kind == "demand" else self.bottom


class ZoneEngine:
    """Bars must already be in forward split-continuous units."""

    def __init__(self, timeframe="d1", config=ZoneConfig()):
        self.timeframe, self.config = timeframe, config
        self.bars, self.zones, self.events = [], [], []
        self.ready = False
        self._atr = None
        self._ranges = []
        self.last_release = None

    def update(self, bar, available_time=None):
        release = available_time or bar.available_time
        if release < bar.available_time:
            raise ValueError("zone cannot precede source availability")
        if self.last_release is not None and release < self.last_release:
            raise ValueError("zone releases must be chronological")
        self.last_release = release
        if self.bars and (bar.symbol != self.bars[-1].symbol or
                          bar.session <= self.bars[-1].session):
            raise ValueError("zone engine requires one chronological symbol")
        self.ready = self._atr is not None and self._atr > 0
        for zone in self.zones:
            if zone.state == "broken":
                continue
            broken = bar.close < zone.bottom if zone.kind == "demand" else bar.close > zone.top
            touched = bar.low <= zone.top if zone.kind == "demand" else bar.high >= zone.bottom
            state = "broken" if broken else "mitigated" if touched else zone.state
            if state != zone.state:
                zone.state = state
                self.events.append({"zone_id": zone.identity, "state": state,
                                    "event_time": bar.event_time, "available_time": release})
        if self.ready:
            body = abs(bar.close - bar.open)
            kind = "demand" if bar.close > bar.open else "supply"
            bases = []
            for base in reversed(self.bars[-self.config.max_base:]):
                width = base.high - base.low
                if width <= 0 or abs(base.close-base.open)/width > self.config.base_body_ratio:
                    break
                bases.append(base)
            if bases and body > self.config.impulse_atr * self._atr:
                bottom, top = min(b.low for b in bases), max(b.high for b in bases)
                departure = bar.close-top if kind == "demand" else bottom-bar.close
                if top > bottom and departure >= self.config.departure_atr*self._atr:
                    identity = digest({"symbol": bar.symbol, "timeframe": self.timeframe,
                                       "formation": bar.event_time, "kind": kind,
                                       "bottom": bottom, "top": top,
                                       "config": asdict(self.config)})
                    zone = Zone(identity, kind, bottom, top, bar.event_time, release,
                                bar.session, body/self._atr*(1+min(self._atr/(top-bottom), 3)))
                    self.zones.append(zone)
                    self.events.append({**asdict(zone), "zone_id": identity})
        tr = max(bar.high-bar.low, abs(bar.high-self.bars[-1].close),
                 abs(bar.low-self.bars[-1].close)) if self.bars else bar.high-bar.low
        self._ranges.append(tr)
        if len(self._ranges) == self.config.atr_period:
            self._atr = sum(self._ranges)/self.config.atr_period
        elif len(self._ranges) > self.config.atr_period:
            self._atr = (self._atr*(self.config.atr_period-1)+tr)/self.config.atr_period
        self.bars.append(bar)

    def snapshot(self, close, daily_atr, session, decision_time, daily_sessions):
        if self.last_release is not None and decision_time < self.last_release:
            raise ValueError("historical query requires replaying the historical prefix")
        if not math.isfinite(daily_atr) or daily_atr <= 0:
            raise ValueError("positive finite daily ATR required")
        values, selected = [float(self.ready)], {}
        for kind in ("demand", "supply"):
            live = [z for z in self.zones if z.kind == kind and z.state != "broken"
                    and z.available_time <= decision_time]
            strength = lambda z: z.strength*(1.5 if z.state == "fresh" else 1)
            live.sort(key=lambda z: (-strength(z), z.identity))
            live = live[:self.config.max_zones]
            live.sort(key=lambda z: (abs(close-z.proximal), -strength(z)))
            # Stable final ties prefer most recent availability, then identity.
            if live:
                distance = abs(close-live[0].proximal)
                score = strength(live[0])
                tied = [z for z in live if abs(close-z.proximal) == distance and strength(z) == score]
                zone = sorted(tied, key=lambda z: z.identity)[0]
                for candidate in tied:
                    if candidate.available_time > zone.available_time:
                        zone = candidate
                signed = close-zone.top if kind == "demand" else zone.bottom-close
                age = len(daily_sessions)-bisect_right(daily_sessions, zone.available_time,
                                                       key=lambda item: item[1])
                values.extend([1., max(-20., min(20., signed/daily_atr)),
                               min(20., (zone.top-zone.bottom)/daily_atr),
                               float(zone.state == "fresh"), min(10., math.log1p(strength(zone))),
                               min(10., age/252.)])
                selected[kind] = asdict(zone)
            else:
                values.extend([0.]*6)
        return values, selected


def feature_names(timeframes):
    return [name for tf in timeframes for name in [f"sd_{tf}_ready"]+
            [f"sd_{tf}_{kind}_{field}" for kind in ("demand", "supply") for field in
             ("present", "proximal_distance_atr", "width_atr", "fresh", "strength_log", "age_sessions")]]


def supply_demand(bars, timeframes=("d1", "w1"), config=ZoneConfig()):
    """Emit immutable daily snapshots; weekly bars release in the next week.

    The first week is discarded conservatively because its leading sessions
    may be absent after a history boundary or gap. No monthly/intraday variant.
    """
    from rl_stocks.features.timeframes.aggregation import completed_week_bars
    if not timeframes or len(set(timeframes)) != len(timeframes) or any(
            tf not in ("d1", "w1") for tf in timeframes):
        raise ValueError("supply/demand supports unique d1/w1 timeframes")
    engines = {tf: ZoneEngine(tf, config) for tf in timeframes}
    daily_atr = atr(bars)
    result = []
    daily_sessions = []
    for i, (bar, completed) in enumerate(zip(bars, completed_week_bars(bars))):
        if daily_sessions and daily_sessions[-1][1] > bar.available_time:
            raise ValueError("nonmonotonic daily availability")
        daily_sessions.append((bar.session, bar.available_time))
        if completed is not None and "w1" in engines:
            engines["w1"].update(completed, bar.available_time)
        if "d1" in engines:
            engines["d1"].update(bar)
        values, zones = [], {}
        for tf, engine in engines.items():
            if daily_atr[i] and daily_atr[i] > 0:
                vector, selected = engine.snapshot(bar.close, daily_atr[i], bar.session,
                                                   bar.available_time, daily_sessions)
            else:
                vector, selected = [0.]*13, {}
            values.extend(vector)
            zones[tf] = selected
        result.append({"values": values, "zones": zones})
    return result, {tf: engine.events for tf, engine in engines.items()}
