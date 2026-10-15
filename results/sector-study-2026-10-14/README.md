# Matched validation study

26 separate symbol episodes; net modeled costs. Validation selection was committed as `179b358` before the separate held-out audit.

| Dataset | Algorithm | Mean validation return | Seed standard deviation |
| --- | --- | ---: | ---: |
| baseline | ppo | 12.97% | 3.57 pp |
| baseline | sac | 15.60% | 5.95 pp |
| baseline | td3 | 4.18% | 8.90 pp |
| combined | ppo | 7.82% | 15.21 pp |
| combined | sac | 19.79% | 27.08 pp |
| combined | td3 | 5.25% | 19.07 pp |
| rotation | ppo | -1.92% | 3.79 pp |
| rotation | sac | 4.12% | 7.82 pp |
| rotation | td3 | 5.09% | 10.96 pp |
| supply_demand | ppo | 9.00% | 9.46 pp |
| supply_demand | sac | 5.93% | 8.35 pp |
| supply_demand | td3 | 3.52% | 3.66 pp |

Predeclared validation selection: `{"algorithm": "sac", "bundle_id": "e57e4f2bc5923f4bc78fca75243c32591848dc234a065eac6ba90eecba981b4b", "dataset": "combined", "seed": 23, "validation_score": 0.5105110373950441}`.

These are retrospective surviving-stock research results, not a live-performance claim.

The first 29 cells used CUDA and the final seven used CPU following a user-directed cost change. CPU/CUDA numerical differences confound a strict feature-only comparison. The winning family has a 27.08 percentage-point seed standard deviation; its mean remains below buy-and-hold (30.60%).

## Held-out audit: 2024 through September 2026

The three frozen combined-SAC seeds were evaluated once after the validation
selection was committed. Candidate seed 23 remains the preselected candidate.

| Seed | Mean net test return across 26 symbols |
| --- | ---: |
| 11 | 9.22% |
| 23 | 13.90% |
| 37 | 9.38% |
| Three-seed mean | 10.83% |
| Buy-and-hold | 48.46% |

Seed standard deviation: 2.66 percentage points. Average per-symbol maximum
drawdown was 26.82%, versus 25.76% for buy-and-hold. These separate symbol
episodes do not form an investable portfolio. The selected policy does not
demonstrate an advantage over the benchmark and remains paper-ineligible.
Normalization and all frozen bundle files remained unchanged.

The validation decision JSON intentionally remains marked `test_evaluated=false`: it
is the immutable historical decision. [The separate audit](test-audit.json) records
test evidence. Future changes need training/validation-only experiments and a fresh
locked audit interval; this test must not become an iterative tuning target.
