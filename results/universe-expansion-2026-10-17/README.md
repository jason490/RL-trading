# Training universe expansion — 2026-10-17

This separate development experiment compares PPO trained on 22 versus 55 stocks,
plus the same four broad ETFs, while scoring both arms on the original 26-symbol
validation panel. The expanded sample remains retrospective and does not resolve
survivorship bias. Current sector annotations are not model inputs.

- [Declared protocol](../../configs/experiments/universe_expansion.json)
- [Dataset identities and coverage](data.json)
- [Separate Alpaca audit](alpaca-audit.json)
- [Operations and interpretation](../../docs/universe-expansion.md)

The Yahoo snapshot passed calendar coverage for all 70 configured symbols.
The control contains 70,200 training rows; the expanded arm contains 158,546.
Both have exactly the same 19,058 validation rows, including executable bars,
feature values, dates and cash dividends. Each normalizer fits only its arm's
training data. Neither dataset contains test episodes, and no 2024 onward bars
were requested for this experiment.

Alpaca returned 139,508 SIP bars for 70 symbols over 2016–2023. LIN has 713 missing
sessions; no failed symbol was dropped or feed substituted. Corporate-action
coverage remains unverified, so this is an ingestion audit, not a training-ready
Alpaca dataset. The credentials remain local and untracked.

Execution acceptance, validation results, artifact publication, shared compute
usage and verified cleanup are recorded here after the fixed Colab cells finish.
No new profitability, held-out performance or broker-order claim is made by data
acceptance alone.
