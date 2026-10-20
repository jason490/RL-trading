# Supervised signal feasibility — data prepared; fitting pending

[The protocol](../../configs/experiments/supervised_signal.json) fixes a completed
20-session gross unit total-return target, masked/full ten-field MLPs with three
seeds, constant and ridge controls, and final-only predictive gates before any
model fitting. No model has been trained and no PPO weights have changed.

| Fold | Examples | Symbols | Decision range | Last label available |
| --- | ---: | ---: | --- | --- |
| Fit | 45,266 | 26 | 2011-01-03 through 2017-11-30 | 2017-12-29 |
| Internal probe | 18,616 | 26 | 2018-01-02 through 2020-11-03 | 2020-12-02 |

The 252-session source warmup explains the 2011 first decision. Source training
purging plus the target horizon explain the early November 2020 last decision.
Every boundary-crossing target is discarded. Fit preprocessing ignores the
source's pooled normalizer and all probe outcomes/features. Changing project
validation content cannot alter the prepared training examples.

[Preflight](preflight.json) binds the checksummed derived payload, fold hashes,
counts and preprocessing. The immutable payload stays under ignored
`data/signals/84b1e38843f8e623c3abbee3d1373c4e20556b39b85891ba8201c1ff54723235/`.
It contains no raw OHLC or project-validation episodes; no raw data are published.

Targets enter at the next open after entry-day corporate actions, include later
splits and cash dividends, and become available at the twentieth close. They
exclude fees, impact, borrow, protection and whole-share rounding. Predictive
accuracy therefore does not establish tradable returns, positive short profits,
or satisfaction of the stock/ETF benchmark goal. The internal probe has been
used in earlier RL training and is not fresh market evidence.

Eleven focused target/context/accounting tests pass, including fold completion,
future perturbation isolation, delayed availability, entry-day versus later
corporate actions, episode separation, unused project-validation data and
fit-only preprocessing. Immutable readback and protocol/data identity checks pass.
No Colab allocation or additional project units; usage remains 35.856639/150.

Next implement the bounded CPU model-fitting job, portable predictor restoration,
strict artifact/metric checks and centrally metered cleanup before allocation.
Actual Colab Python and tiny finite-loss/save/load checks must pass before fixed
full fits. A failed feasibility gate stops the family; a pass would require a
separate matched policy protocol and prospective evidence before any success claim.

[Definitions, leakage controls and commands](../../docs/supervised-signal.md).
