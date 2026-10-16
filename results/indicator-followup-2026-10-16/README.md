# Matched 250k indicator follow-up

This separate study fills the missing longer-input comparison. Nine fresh CPU
SAC cells are planned: combined inputs, supply/demand-only and sector-only, each
at 250k transitions with seeds 11/23/37. Reward, architecture, costs and dates stay
fixed. The three completed baseline 250k runs are the matched control, not new
training jobs. The previous exhausted learning study remains unchanged.

- [Committed protocol](../../configs/experiments/indicator_followup.json) and [category guide](../../docs/indicator-experiments.md).
- [Matched dataset preflight](dataset-preflight.json): identical train/validation bar sequences across all four feature families (7/33/62/88 inputs).
- [Current comparison](comparison.json) and [summary CSV](comparison.csv): complete families only; all four are required for interaction attribution. A partial study cannot establish the complete effect.
- Per-seed compact CSVs sit beside the comparison; standalone HTML graphs are generated under `artifacts/public/indicator-followup-2026-10-16/`.
- [Training-only regime coverage](training-regimes.json) and [per-symbol CSV](training-regimes.csv): 16 of 26 symbols contain declining trailing windows. Only 1,851 of 63,648 eligible sessions (2.91%) have trailing-year price declines below −20%; overlapping observations are not independent events.

Historical membership and delisting returns remain unverified. Training-only
regime labels use trailing prices, never future recovery outcomes. No additional
stocks or rewards enter the indicator experiment. The inspected 2024–September
2026 interval stays retired. No new held-out audit or broker orders occur.

The cumulative project ledger starts at 34.032604 units; the same 150-unit ceiling,
130-unit teardown threshold and two-hour runtime deadline apply. Runtime state,
minute metering and recovery IDs stay in `.runtime/`; authentication logs are
private. Publish verified artifacts and then clean temporary duplicates.
