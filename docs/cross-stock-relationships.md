# Cross-stock relationships

## Current policy and new optional component

Current accepted policies trade one symbol per episode. Shared weights can learn
patterns common across stocks, but their21-field observations do not contain a
synchronized stock panel, peer returns or pairwise correlations. Adding stocks to
the training pool does not by itself expose contemporaneous relationships.
Earlier SPY-context experiments failed their fixed gates; they remain closed.

`features/market/relationships.py` now implements a separate, disabled component:

| Coordinate | Meaning |
| --- | --- |
| stock_market_correlation_20 | Pearson correlation of20 completed daily log returns with SPY |
| stock_sector_correlation_20 | Same calculation against an explicitly supplied sector proxy |
| sector_return_20 | Sum of the sector proxy's20 log returns |
| relative_sector_return_20 | Stock minus sector20-session log return |

Twenty returns require21 aligned closes. Split factors make returns continuous
when raw corporate-action bars are used. Exact session windows must agree across
stock, market and sector. Every contributing bar must be available at decision
time. Known gaps or constant-return series produce missing readiness, not a zero
correlation or stale forward fill. Instantiate separately for each chronological
split; never bridge a split boundary implicitly. Fit any later preprocessing on
training rows only and freeze it thereafter.

The caller must supply historically justified stock-to-sector mappings. The new
universe's present research categories are not such evidence. This component is
not wired into datasets, checkpoints, frozen policies or broker execution. Tests
cover alignment, availability, missing-session rejection and future isolation.

## Experiment and larger architectures

First audit historical sector membership, price continuity, common-session
coverage and feature distributions without using returns to choose stocks. Then
preregister a fixed same-dimension masked/full input comparison against the
strongest relevant SAC control and matched holding. Keep execution, costs,
reward, data and checkpoint rules fixed; require selected/final gates and seeds
11/23/37 before prospective confirmation. This is a distinct peer/sector
relationship proposal, not permission to rerun the failed SPY-only family.

A later pairwise panel can calculate rolling correlations against a fixed peer
list, or summarize peers using sector breadth and relative strength. A graph or
attention encoder can process simultaneous stock observations and masks, but
requires a new observation/frozen-inference contract and separate capacity/cost
profiling. Every asset must have an availability timestamp; missing/delisted
assets need explicit masks and historical identities. Correlation is descriptive
and does not establish causation or future predictability. Portfolio allocation
across stocks remains a separate environment milestone.
