# Execution-state observation study — completed, failed promotion

The full observation improves this seed's overall development return but fails
the fixed dual-control gate and the user's profitable-short requirement. No
confirmation seeds or longer training were launched. The optional state feature
remains available for research; the default legacy input contract is preserved.

## Matched 50k PPO results

Seed 11, original 26-symbol panel, wide initialization, legacy-uniform starts,
unchanged costs/actions/net log equity reward. All selected models were the final
post-update models; selected and final returns happen to coincide here.

| Profile | Selected/final mean return | Stocks | ETFs | Mean maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 7.96% | 8.12% | 7.09% | -29.81% |
| Masked, same dimension as full | 15.81% | 15.91% | 15.27% | -26.95% |
| Full execution state | 22.59% | 25.13% | 8.64% | -32.50% |
| Same-symbol buy-and-hold | 30.60% | 32.63% | 19.47% | — |

Full gains 6.78 percentage points overall against masked inputs, but ETF return
falls 6.63 points and drawdown worsens 5.55 points. Against legacy, drawdown
worsens 2.70 points, above the two-point allowance. Both comparisons therefore
fail. These are single-seed development results, not a replacement for the
previous multi-seed baseline or independent evidence. No retired test was used.

## Profitable shorting is still unresolved

Final full-state results for the three stocks with negative holding returns:

| Stock | Buy-and-hold | Policy total return | Net short P&L on $100k initial equity |
| --- | ---: | ---: | ---: |
| AMZN | -7.73% | -16.42% | +$1,185.17 |
| DIS | -47.38% | -45.95% | +$515.09 |
| NEE | -14.63% | -16.90% | +$180.69 |

Short contributions are positive, but long losses dominate. None has both
positive total return and positive short contribution. The legacy control also
has positive short P&L on all three but negative total returns. Masked inputs
produce +19.80% total on AMZN with negative short P&L, which also does not satisfy
the shorting requirement. Full-episode negative-holding labels are reporting-only.
All 156 profile/phase/symbol rows are retained in [per-symbol.csv](per-symbol.csv).

## Verification and operations

Three 512-step CPU smokes preceded three 50k CPU cells. Actual Colab Python was
3.13.16. Source `8b39004de0e0fc0da92b14914d5362bd3a3cf615` was frozen for every
cell and shared the original ledger/allocation lock. Legacy smokes and research
exactly reproduce previous wide controls: 13 selected and 13 final policy
tensors per run, plus every checkpoint score. Six bundles pass 468 exact frozen
predictions, immutable normalization/data and incompatible-input rejection.
All six owned sessions stopped with verified cleanup; account active assignments
are zero. Regression evidence: full suite 214 passed, followed by 32 focused
profile/runtime tests and ten profile/audit checks after final additions.

Conservative study usage is **0.049487 units**, cumulative **35.687823/150**,
with **114.312177** remaining and **94.312177** before working teardown at 130.
Meter-window measured debits total 0.02; rounded account balances changed from
350.77 to 350.73 (0.04). These readings and the conservative estimate are
reported separately, never added. Each allocation included setup/retrieval in
its two-hour deadline. No budget stop occurred and no broker orders were sent.

See [methodology](../../docs/execution-observations.md),
[protocol](../../configs/experiments/execution_observations.json),
[preflight](preflight.json), [smokes](smoke.json),
[screen and dual gates](screen.json), [short-side audit](short-side.json),
[frozen checks](frozen-checks.json), [compute](compute.json),
[legacy smoke parity](legacy-smoke-parity.json), and
[legacy research parity](legacy-screen-parity.json).
The interactive checkpoint chart is generated at
`artifacts/execution-observations/screen.html` and published with frozen models.

## Next diagnostic

Replay fixed selected/final policies with universal long-only and short-only
exposure filters on **all** validation symbols, retaining their risk actions and
recomputing portfolio observations along each new path. This can separate short
signal quality from long losses; accounting attribution alone is not a
counterfactual trading strategy. No future decline label may activate a filter,
no checkpoint is reselected, and no training/allocation is needed for this
bounded inference diagnostic. A profitable subset would not establish an
all-market strategy or resolve survivorship/prospective validation limits.

## Published artifacts and cleanup

[Download models and interactive reports](https://github.com/jason490/RL-trading/releases/tag/execution-observations-2026-10-19). All 16 server
SHA-256/size pairs are verified ([proof](release-verification.json)). Cleanup
removed 27 duplicate archives/markers, 305,562,368 logical bytes (291.41 MiB).
All 314 retained file hashes remained unchanged. Final archives matched every
extracted file; payloads matched retained source, dataset and plans before
removal. Extracted models/reports, source export, raw data and the cumulative
ledger remain intact ([cleanup evidence](cleanup.json)).
