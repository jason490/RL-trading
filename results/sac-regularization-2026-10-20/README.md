# SAC regularization — interrupted, no promotion

The fixed paired100k seed11 Colab CPU screen is incomplete. Aggregate metering
became unavailable on2026-10-20; the controller triggered a protective budget
stop. Control completed, but candidate final100k was not retrieved. The original
shutdown attempt could not verify the candidate; exact-fleet recovery on
2026-10-21 now verifies every owned session stopped. No training restarted.

| Arm | Selected/saved best mean | Final100k post-update | Latest retrieved curve |
| --- | ---: | ---: | ---: |
| Control | 21.97% | 21.63% | 21.63% final100k |
| Regularization1e-4 | 6.39% | Missing | 5.10% at80k, on-step |
| Matched holding | 30.60% | 30.60% | 30.60% |

Separate2021–2023 cumulative development episodes, not annual returns or an
investable portfolio. Candidate80k is not substituted for final100k; the fixed
screen is unassessable and cannot promote or authorize confirmation. All saved
phases achieve0/3 joint positive total/short P&L on falling stocks; no robust
short success is established. Future decline cohorts remain reporting-only.

A snapshot retains stale `validation.json` matching the50k selected report,
not the80k curve. The interruption summary labels it unaligned; do not present
its6.39% as80k performance. Actual latest-checkpoint model restores80000
transitions. Its weights and actual optimizer settings are finite/valid. Complete
control selected/final and partial candidate best/latest-checkpoint restore;
strict saved-path accounting passes. No full candidate frozen bundle/final
acceptance is claimed.78 control plus156 smoke frozen-prefix predictions pass.

## Operations and budget

The first two-session smoke attempt failed strict plan validation because SB3
mutated caller-owned nested optimizer settings. Fix6792d2b isolates overrides;
20 focused checks pass. Replacement two1024 smokes pass actual optimizer save/
load, identical initial networks/alpha and actual sampling,156 exact frozen
predictions, immutable normalization/schema rejection. All4 smoke sessions and
both screen sessions are now stopped with verified cleanup.

Cumulative **conservative estimate37.78721340/150 units**, remaining112.21278660.
Study increment1.52331813 includes failed attempt/retry and conservative gap
charging. Screen measured account debit0.32 units versus conservative estimate
1.48961591. The ledger carries last known rate across the unmonitored interval;
this is an upper accounting estimate, not measured GPU/CPU usage. It is retained
without downward reset. Post-recovery account349.97, rate0, assignments0.
The budget-stop flag stays set; **no automatic restart**. Further allocations
require explicit review/authorization; offline evidence work can continue.

[Interruption evidence and hashes](interruption-summary.json),
[checkpoint curves](learning-curves.csv), [control restoration](control-restoration.json),
[smoke acceptance](smoke-summary.json), [smoke restoration](smoke-restoration.json),
[first failure](smoke-attempt-1.json),
[protocol](../../configs/experiments/sac_regularization.json),
[guide](../../docs/sac-regularization.md).

Models, checkpoints, replay, reports and source exports remain retained locally;
public assets exclude raw bars, credentials and replay. Retired2024–September2026
stays unopened. No extension, weight sweep, confirmation or promotion follows
this incomplete comparison.

[Published complete and explicitly incomplete artifacts](https://github.com/jason490/RL-trading/releases/tag/sac-regularization-2026-10-20),
[13 server asset digests](release-verification.json),
[duplicate cleanup](cleanup.json), [byte-identical reproduction](reproduction.json).
Cleanup removed3 published archive copies/25,401,659 bytes with656 retained
hashes unchanged. Reproduction command is in the category guide; all original
checkpoints, failed archives, replay, data, sources and ledger remain retained.
