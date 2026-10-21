# Fixed recurrent unit sizing — improved returns, full gates failed

Four preregistered saved-policy replays: masked recurrent seed11 selected/final,
unrestricted versus sign-preserving unit exposure. Other learned protective
outputs, costs, borrow, whole shares and20-session holding cap remain unchanged.
Fresh causal observations and recurrent memory follow each altered trading path.
No fitting, provider calls, Colab allocation, test evaluation or checkpoint selection.

| Mode | Selected mean return | Final mean return |
| --- | ---: | ---: |
| Unchanged control | 24.99% | 24.04% |
| Unit magnitude | 28.53% | 32.80% |
| Matched holding | 30.60% | 30.60% |

Separate cumulative2021–2023 development episodes, not annualized returns or an
investable portfolio. Final unit stocks35.40% versus holding32.63% (+2.77pp),
ETFs18.55% versus19.47% (−0.92pp), median stock excess+0.41pp. Final absolute
comparability targets pass. Both selected/final gains over control exceed2pp.

**The complete diagnostic fails.** Selected ETFs trail holding3.36pp and median
stock excess is−7.56pp. Final mean maximum drawdown−33.20% versus control−29.99%
worsens3.21pp, exceeding2pp. Selected mean drawdown is−33.62%. Unit selected/final
joint positive total return AND positive short P&L outcomes are0/3 on reporting-
only declining stocks. Do not promote the final phase or ignore failed selected/
risk gates. No magnitude/threshold sweep, extra phases or training follows this
failed diagnostic. Overall multi-seed/prospective objective remains unmet.

## Integrity

Both unrestricted replay gates pass the original declared tolerances before any
unit candidate evaluation. Quantities, sessions, signs and holding baselines match
exactly. Full reports are not byte-identical: maximum selected/final daily equity
differences are$0.013815/$0.005020, action differences1.699e-6/7.153e-7, return
fraction differences1.230e-7/4.721e-8. These are below preregistered$0.10,2e-6,
1e-6 bounds; no tolerance was relaxed. Weights, update counts, steps, normalization,
original metadata and report hashes remain unchanged.12 filter/state/reference
checks pass without fitting.

[Exact gates/parity/groups](summary.json),
[fixed protocol](../../configs/experiments/recurrent_unit_sizing.json),
[guide](../../docs/recurrent-sizing.md),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/recurrent-unit-sizing-2026-10-21/comparison.html),
[compressed full paths](https://github.com/jason490/RL-trading/releases/tag/recurrent-unit-sizing-2026-10-21).
[Original model release](https://github.com/jason490/RL-trading/releases/tag/recurrent-state-2026-10-21)
retains the exact trained weights; they are not duplicated in this release.
Replay source is committed f74fa59; original training source dd3b256 remains explicit.

## Interpretation and next work

Sizing affects returns and risk enough to justify attention, but this one-seed
reused-development replay is not independent evidence. Its phase ordering reverses:
unrestricted selected beats final, whereas unit final beats selected. The policy
was trained and checkpoints selected under adaptive magnitude, not this deployed
transform. That is a possible training/execution mismatch, not proof of cause.
Any different training/execution-alignment hypothesis requires its own frozen
protocol and controls; it cannot be called confirmation of this failed replay.
Do not tune a magnitude to rescue the selected/risk gates.

Project usage remains38.014408/150, account last verified349.80 with zero owned
sessions. This replay uses local CPU inference only and adds no Colab units.
Source data, original weights, four compressed replay paths and failure evidence
are retained. No broker orders or prospective campaign started.

Publication verifies8 server assets. Cleanup removes19,493,536 duplicate
bytes with13 retained file hashes unchanged. All four saved
paths reproduce group and short-side metrics exactly after cleanup.
[Publication](release-verification.json), [cleanup](cleanup.json),
[metric reproduction](reproduction.json).
