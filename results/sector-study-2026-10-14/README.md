# Matched validation study

26 separate symbol episodes; net modeled costs. Final-test audit has not yet run.

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
