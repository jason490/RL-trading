# Saved recurrent-state path diagnosis

Three completed100k seed11 runs, selected/final development reports only.
No fitting, dataset opening, provider calls, retired-test access or Colab allocation.
Metadata/report hashes remain unchanged.15 CLI/accounting/behavior/report checks
pass, including exact explicitly declared mixed PPO/recurrent algorithm panels.
[Exact per-symbol accounting](summary.json), [compact CSV](summary.csv).

| Final arm/group | Net excess vs holding | Incremental costs | Non-cost excess |
| --- | ---: | ---: | ---: |
| feedforward_full / stock | -31.28pp | 6.84pp | -24.44pp |
| feedforward_full / etf | -5.18pp | 6.18pp | 1.00pp |
| recurrent_masked / stock | -6.81pp | 6.11pp | -0.69pp |
| recurrent_masked / etf | -5.19pp | 4.73pp | -0.47pp |
| recurrent_full / stock | -34.88pp | 5.81pp | -29.07pp |
| recurrent_full / etf | -27.31pp | 5.20pp | -22.10pp |

Net excess equals non-cost excess minus incremental costs. Cost addback describes
saved paths, not a zero-cost counterfactual: actions and exposure would change
under another learning/execution contract. Full-state candidate performance is
primarily weak participation/timing on these paths; costs alone cannot close its
gaps. No optimizer or information-causality claim follows.

Masked recurrent final stock long/short contributions25.34%/+0.48%; ETF15.79%/
−1.52%. Mean end-session long exposure67.04%/67.44%, short6.05%/8.01%.
Its non-cost excess is near holding but actual returns still miss targets.
Small same-side resizes below5% equity cost only0.067/0.081pp; removing them alone
cannot close6.81/5.19pp stock/ETF gaps. Mandatory protective exits and20-session
holding cap remain required. Future declining cohorts do not select policies.

Full recurrent final stock long/short contributions+1.53%/−3.78%, ETFs−1.71%/
−6.13%. Mean long exposure51.58%/46.71%, short7.14%/6.90%. Selected-to-final
stock return falls8.19pp: long contribution−4.58pp, short−3.61pp, with incremental
cost drag+2.35pp. ETF net falls0.85pp: longs improve3.80pp, shorts worsen4.66pp;
cost drag increases2.26pp. Directional contributions already include costs, so
these two decompositions must not be added together. Both states remain failed.

Reproduce from retained reports:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_participation.py \
 --state .runtime/recurrent-state-screen-state.json \
 --algorithm ppo --algorithm recurrent_ppo --steps 100000 \
 --output /tmp/recurrent-state-paths.json
```

Repeated algorithm flags declare an exact panel; absent or undeclared algorithms
reject. Single-algorithm invocations keep their default PPO behavior. All source
studies must be complete with verified cleanup and identical holding panels;
output must be new. No probability of future profits is established.

No extension/confirmation/promotion follows the failed state protocol. Before a
distinct cost intervention, audit whether avoidable discretionary fills or required
protection/holding exits account for the stronger control's drag; future changes
must retain matched controls and absolute stock/ETF gates. Project remains
38.014408/150; no active owned sessions. Robust multi-seed/prospective success
remains unproven.
