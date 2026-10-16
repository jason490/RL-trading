# Algorithm and recurrent-memory stability study

All nine declared research cells completed on Google Colab CPU: SAC, PPO and
RecurrentPPO at 100,000 transitions, seeds 11/23/37, basic inputs, unchanged
net-equity reward and costs. Peak concurrency was three; no further training
was added after the results. The [protocol](../../configs/experiments/algorithm_stability.json)
and [detailed guide](../../docs/algorithm-stability.md) describe the fixed comparison.

| Algorithm | Selected mean return | Final mean return | Selected-to-final gap | Final seed standard deviation |
| --- | ---: | ---: | ---: | ---: |
| SAC | 19.70% | 16.50% | 3.20 pp | 6.63 pp |
| PPO | 18.69% | 16.90% | 1.79 pp | 6.09 pp |
| RecurrentPPO | 13.27% | 6.13% | 7.14 pp | 1.33 pp |

Buy-and-hold averaged 30.60%. These are total net validation returns across
separate stock episodes, averaged equally over stocks and seeds, not a portfolio,
annualized returns or new held-out evidence. Validation covers 2021–2023 with
purged boundaries (the final episode ends November 30, 2023). The retrospective
26-stock/ETF universe limits generalization. The inspected test interval remains
retired; no orders are enabled and no candidate is paper-eligible.

## Interpretation

All three predeclared comparisons failed: PPO versus SAC, RecurrentPPO versus SAC,
and RecurrentPPO versus PPO. PPO's final mean beat SAC by only 0.40 percentage
points, its selected mean was 1.01 points lower, and its gap reduction was 1.41
points, below the required 2 points. PPO beat SAC's endpoint in two seeds but
lost by 14.14 points in seed 37. RecurrentPPO lost to both controls at the endpoint
in every seed. Memory did not improve this configuration; that does not establish
that recurrent models cannot work with other separately tested configurations.

The [saved-path behavior](behavior.json) distinguishes costs from direction/timing
and exposure changes. PPO seed 37 fell 4.57 points from selected to final; extra
costs account for 3.77 points on the recorded paths. Recurrent seed 23 fell 11.39
points; extra costs account for 2.41 and the remaining non-cost P&L change is
−8.98 points. Exposure-action saturation rose from 8% to 47% in that recurrent
run. A universal turnover penalty would not explain every failure. Cost add-back
is accounting, not a cost-free rerun; neither these observations nor optimizer
losses prove a causal optimization failure. Selecting the best of many validation
checkpoints also creates an optimistic gap to a single endpoint.

The [earlier diagnosis](diagnosis-before-study.json) covers all 12 completed 250k
indicator/control runs. Their best checkpoints remain retained separately.

## Reports and reproducibility

- [Interactive graphs and models](https://github.com/jason490/RL-trading/releases/tag/algorithm-stability-2026-10-16): download `comparison.html` for a standalone learning comparison. Extract `validation-reports.tar.gz` alongside `validation.html` for per-stock equity charts and metrics.
- [Complete comparison](comparison.json) and [summary CSV](comparison.csv): all seed scores, drawdowns, source proofs and failed gates.
- Per-seed `sac-11.csv`, `ppo-11.csv`, `recurrent_ppo-11.csv` and corresponding seeds 23/37: compact validation measurements every 10k steps, including incremental return changes and the post-update endpoint.
- [Frozen verification](bundle-verification.json): all nine bundles pass exact CPU sequence predictions, recurrent state/reset parity where applicable, immutable normalization and schema/feed/shape rejection.
- [Concurrent preflight](concurrent-preflight.json): actual three-runtime CPU acceptance, including the reviewed transport timeout and bounded retry fix.

All nine use clean source revision `e6ea781`, one learning-source identity and
locked dependencies, with the actual Colab Python 3.13.16 kernel. Full local
acceptance passed 146 tests, zero skips/failures. Models and derived reports are
published separately from raw provider bars and replay buffers.

All owned sessions stopped and teardown was verified. Conservative cumulative
project usage is 35.389379 units, leaving 114.610621 of the 150-unit ceiling.
The nine-run research fleet added 0.257628 units; concurrent preflight and its
reviewed retries are also included in the cumulative total. Across the entire
project, summed account debits are 31.65 units and rate estimates are 35.078441;
the conservative ledger accounts for metering precision and delay rather than
assuming either alone is exact.

## Publication and cleanup

All 13 release assets have matching server SHA-256 digests and sizes; see
[release verification](release-verification.json). After verification, cleanup
removed 44 duplicate/temporary files totaling 678,561,987 logical bytes (about
647 MiB); observed free space increased by 678,645,760 bytes. All nine frozen
bundles, best/latest weights, nine endpoint milestones, final SAC replay buffers,
curves and detailed validation outputs remain. All 129 retained files were
rehashed. Unique failed-preflight checkpoints, smoke outputs, datasets and
ledger/recovery records were preserved; earlier studies were untouched.
See [cleanup inventory](cleanup.json) and [sanitized usage](usage-summary.json).

## Next work

Stop this bounded protocol here. Establish historical membership and verified
delisting returns before a separately declared universe experiment. A matched
turnover intervention is a separate hypothesis, especially for PPO seed 37.
TQC, stacked observations, additional recurrent tuning, new rewards, and longer
runs were not executed in this study. Persistent shadow inference and broker
order lifecycle work remain later milestones. New out-of-sample claims require
a fresh locked audit interval.
