# Longer-horizon context — completed; both gates failed

| Arm | Selected return | Final return | Final stocks | Final ETFs | Mean final maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched_baseline | 0.83% | 0.83% | 0.07% | 5.02% | -15.90% |
| masked | 13.93% | 13.93% | 14.84% | 8.91% | -26.62% |
| full | 14.26% | 13.89% | 15.86% | 3.07% | -24.26% |

Matched holding returned30.60% overall,32.63% stocks,19.47% ETFs. Full context gains only0.34pp selected and loses0.04pp final versus the masked control; final ETFs worsen5.84pp. Against the matched baseline, final return improves13.06pp but ETFs worsen1.95pp and drawdown worsens8.37pp. Both fixed gates fail. No confirmation seeds, promotion, or longer training for this family.

The large difference between the two controls means an apparent gain against the smaller baseline alone would be misleading. This one-seed screen does not prove the features can never help, but it does not justify adoption. Longer warmup removes5,278 early training rows and changes training normalization relative to the older dataset. Its22.59% native run is therefore not an equivalent control.

## Declining stocks: full-context final weights

| Symbol | Total return | Net short P&L | Both positive |
| --- | ---: | ---: | --- |
| AMZN | -11.43% | $6,448.58 | no |
| DIS | -31.84% | $1,935.44 | no |
| NEE | 2.15% | $-1,670.94 | no |

Full-context joint success is0/3. AMZN and DIS have positive short P&L outweighed by long losses; NEE has positive total return despite a negative short contribution. Masked control passes on NEE only. Future negative-holding labels are reporting-only and do not choose weights, rewards or trades.

## Data, verification and operations

The committed [protocol](../../configs/experiments/trend_context.json) uses the retained2010–2023 snapshot and original26-symbol development panel. All arms have64,922 training and19,058 validation rows, identical eligible bar/ATR panels and first7 training normalizer statistics. Original validation dates are unchanged. Price context excludes cash dividends; net-equity simulation still includes them and short costs. No retired-test bars or broker orders are used.

All3 CPU512-step smokes and3 CPU50k cells completed on actual Colab defaultPython3.13.16 with source780f382f6ea6cd862535b12c1723ee4dc06ade87. All6 bundles pass468 exact CPU prefix predictions, deterministic repeat, immutable normalization, and incompatible schema/feed/nonfinite rejection. All6 owned sessions stopped and shutdown was verified. No research training occurred locally.

[Screen and complete checkpoint curves](screen.json), [smoke acceptance](smoke-acceptance.json), [research frozen checks](frozen-checks.json), [data preflight](preflight.json), [environment checks](environment-preflight.json), [compute](compute.json), [feature guide](../../docs/trend-context.md).

Study conservative usage0.061261 units versus measured account debit0.04; project35.856639/150, remaining114.143361. Reported rate0.00 and0 active assignments. All14 published assets passed server hash/size verification; duplicate cleanup is complete. Original models, full reports/curves, data, source export and ledger remain.

## Next diagnosis

Retain the original default feature set. Inspect saved exposure, turnover, side contributions and protective-exit behavior to separate missed market participation, trading costs and signal errors before proposing another training change. Adding context did not solve ETF performance or joint short profitability. New settings require another bounded preregistration and independent prospective evidence.

## Published artifacts and cleanup

[Models and visual reports](https://github.com/jason490/RL-trading/releases/tag/trend-context-2026-10-20).
All14 server SHA-256/size pairs match ([verification](release-verification.json)).
Cleanup removed21 proven duplicate archives/markers,196,449,793 logical bytes
(187.35MiB), with316 retained hashes unchanged ([cleanup](cleanup.json)).
Six earlier checkpoint archives differ from final extraction and were kept
conservatively. No unresolved session or cleanup failure remains.
