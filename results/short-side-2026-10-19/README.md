# Short-side audit — 2026-10-19

Audit of the completed matched 100,000-transition PPO, SAC and recurrent PPO
study: seeds 11/23/37, 2021–2023 development validation, the original 22 stocks
and four broad ETFs, basic seven features, unchanged net-equity reward and costs.
No new training, checkpoint selection, Colab allocation or retired-test access.

Three stocks had negative same-interval buy-and-hold returns: AMZN (-7.73%),
DIS (-47.38%) and NEE (-14.63%). Their equal-weight holding return was -23.25%.
The cohort is determined after evaluation for reporting only.

| Model | Selected policy return | Final policy return | Final profitable policy **and** short P&L |
| --- | ---: | ---: | ---: |
| PPO | -15.75% | -20.62% | 1/9 stock-seed observations |
| SAC | -19.46% | -18.99% | 2/9 |
| Recurrent PPO | -3.69% | -21.64% | 1/9 |

These are mean returns on the three declining stocks across three seeds, not
returns for the entire evaluation panel and not nine independent stocks.
None of the families consistently turns falling stocks into positive returns.
Selected recurrent PPO's smaller loss does not establish a better overall model.

There are real positive short-side examples. Final SAC seed 37 returned **16.22%**
on NEE versus holding **-14.63%**, including **$7,823.34** short net P&L per
$100,000 account. But final PPO seed 23 on DIS returned **-41.47%** despite
**$7,994.00** positive short P&L: losses on long exposure more than offset shorts.
Final recurrent PPO seed 11 returned **37.39%** on NEE while its short contribution
was **-$4,848.73**; that positive total must not be described as profitable shorting.

`summary.json` contains each run/phase's declining-stock details and SHA-256 of
its saved source report. `per-symbol.csv` contains all **468 symbol/seed/phase**
records, including rising stocks and ETFs. Attribution reconstructs signed cash
flows, dividends, split cash in lieu, entry/exit fees, impact, daily borrow and
open-position marks. Long plus short P&L reconciles to equity every session.
The source dataset is
`dfd600f62d5f12f7aa1c62b64a3fe2a1d00a48b3264f2a7fff31354958a57059`;
source study: [algorithm stability](../algorithm-stability-2026-10-16/README.md).

Next research should distinguish entry-direction/timing errors from execution
constraints with causal trend baselines before adding longer training or new
reward penalties. The short-specific report is diagnostic; its future-negative
labels cannot select checkpoints or enter the model. See
[short-side methodology](../../docs/short-side.md) and
[bounded research workflow](../../docs/research-loop.md).
