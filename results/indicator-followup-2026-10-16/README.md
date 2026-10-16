# Matched 250k indicator follow-up

This separate study fills the missing longer-input comparison. Nine fresh CPU
SAC cells are complete: combined inputs, supply/demand-only and sector-only, each
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

## Completed results

Three seeds per family; net validation returns over 2021–2023, equal means of
26 separate stock episodes. These are not annualized or independent test results.

| Inputs | Selected mean | Final mean | Selected seed std | Paired feature gate |
| --- | ---: | ---: | ---: | --- |
| Basic control | 23.96% | 14.46% | 4.14 pp | Control |
| Basic + supply/demand | 18.39% | 11.83% | 5.66 pp | Failed |
| Basic + sector context | 30.78% | −11.30% | 14.86 pp | Failed |
| Both additions | 46.82% | −1.73% | 22.84 pp | Failed |

Buy-and-hold mean: **30.60%**. Combined inputs improve selected return by
22.85 points and selected drawdown by 5.52 points versus basic inputs, but lose
16.19 points at the final endpoint and worsen final drawdown by 3.80 points.
Sector-only improves selected return by 6.82 points but loses 25.77 final points.
Supply/demand-only loses 5.57 selected points and 2.63 final points. No feature
family passes the declared return/stability/drawdown gate. No paper candidate is
approved and the previously inspected final-test interval remains retired.

Supply/demand adds −5.57 selected points without sector context and +16.04 with
it; the selected interaction is +21.61 points, with a 38.81-point seed standard
deviation. This is a noisy validation result, not proof of a generalizable
interaction. Strong checkpoint selection does not mean later training improves.

All nine new bundles and three reused controls passed exact CPU action parity,
immutable normalization and schema/shape rejection checks; see
[bundle verification](bundle-verification.json). Clean uploaded Git source,
dependency, simulator, benchmark and bar-sequence proofs match the original
control. The later RecurrentPPO implementation was not used in these runs.

The follow-up used **1.065622** additional conservative units, including its
interrupted attempt and retry. Cumulative project consumption is **35.098226**
units, leaving **114.901774** of the 150-unit ceiling. Summed measured account
debits are 31.43 units; rate estimates sum to 34.787288; the conservative ledger
sums each attempt's larger value. All 49 owned attempts are stopped and verified.

Models, milestone weights, standalone graphs and checksums are published at
[the indicator-follow-up release](https://github.com/jason490/RL-trading/releases/tag/indicator-followup-2026-10-16).
The selected-versus-final diagnosis and new bounded algorithm comparison are
[documented separately](../../docs/algorithm-stability.md).
