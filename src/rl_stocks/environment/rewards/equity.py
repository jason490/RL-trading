"""Cost-aware equity objectives; neither rewards short proceeds nor future labels."""
import math

KINDS = ("net_log_equity", "net_initial_equity_change")


def validate_reward_kind(kind):
    if kind not in KINDS:
        raise ValueError("unknown equity reward kind")
    return kind


def equity_reward(previous, current, kind="net_log_equity", initial_equity=100000.):
    validate_reward_kind(kind)
    if not all(math.isfinite(v) for v in (previous, current, initial_equity)) or initial_equity <= 0:
        raise ValueError("finite equity and positive initial capital required")
    if kind == "net_initial_equity_change":
        return (current - previous) / initial_equity
    return math.log(current / previous) if previous > 0 and current > 0 else -20.
