# Recurrent sizing alignment — complete, candidate gates failed

Fresh fixed100k seed11 Google Colab CPU runs isolate training/evaluation sizing
with identical masked21 observations, expanded111 training symbols, original26
2021–2023 development episodes, normalization, sampling, model initialization,
PPO settings, reward, costs and20-session holding limit.

| Arm | Training / evaluation sizing | Selected return | Final post-update return |
| --- | --- | ---: | ---: |
| Adaptive | adaptive / adaptive | 24.99% | 24.04% |
| Posthoc unit control | adaptive / unit | 32.80% | 32.80% |
| Aligned unit candidate | unit / unit | 20.04% | 13.93% |
| Matched holding | — | 30.60% | 30.60% |

Separate cumulative development episodes, not annualized returns or an investable
portfolio. Adaptive and posthoc final policy tensors match exactly; their training
is identical, while validation execution changes selection. Posthoc selects its
final weights under unit evaluation, explaining why its selected score differs
from the earlier fixed replay's adaptively selected checkpoint. Do not relabel
that earlier failed replay as successful.

**Aligned training fails every required return gate.** Selected/final gains are
−4.95/−10.12pp versus adaptive and−12.76/−18.88pp versus posthoc. Candidate final
stocks14.86% versus holding32.63%, ETFs8.77% versus19.47%, median stock excess
−24.77pp. Final mean drawdown−37.41% worsens7.42pp versus adaptive and4.21pp versus
posthoc, beyond2pp. No extension, threshold/architecture sweep, confirmation or
promotion follows this fixed family. Posthoc control's32.80% result is still
one-seed repeatedly used development evidence; it does not establish robustness
or prospective success. Its drawdown is3.21pp worse than adaptive.

## Short side and interpretation

Aligned selected joint positive total return AND positive net short P&L succeeds
on2/3 reporting-only declining stocks; final falls to0/3. Adaptive1/3 selected,
0/3 final; posthoc0/3 both. A temporary selected short outcome cannot rescue final
or whole-panel failures. Future decline labels never enter learning or selection.

Alignment alone did not improve this configuration. Unit execution requests no
voluntary flat allocation except exact zero; all100,000 training requests were
nonzero. The candidate transformed62,293 sizes, processed0 non-unit exposures
and changed0 protective components. Both controls preserve55,315 non-unit raw
sizes with0 transforms. This proves execution, not a cause of poor returns.
A separate hypothesis would need evidence and fixed controls; this failed study
cannot be rescued by changing thresholds or dropping its stronger control.

## Integrity and operations

All3 initial tensor records and actual sampling counts match; all111 symbols
receive transitions. Actual initial/selected/final settings, steps, tensor hashes,
execution counters/action traces and mandatory checkpoint identities restore.
468 smoke/screen frozen predictions match exactly, including mapped action,
hidden/cell state, repeats, independent resets and caller-state immutability.
Normalization is frozen, incompatible inputs reject, and strict reports reconcile
source bars, equity, costs, directions and matched holding paths.

All6 owned sessions stopped, artifacts retrieved and teardown verified. Actual
Python3.13.16, clean workercaddfcb3bbccc855c769e194fe10a075ce596a96; existing CLI
reused. Project38.13665836/150, remaining111.86334164; study increment0.12225039
includes setup/idle/retrieval/all attempts. Official account349.71, zero rate and
assignments. Original ledger and account150/170/project150/130 safeguards retained.
No provider calls, retired2024–September2026 evaluation, local fitting or orders.

[Exact gates and per-symbol metrics](summary.json), [10k curves](learning-curves.csv),
[screen frozen restoration](restoration.json), [smokes](smoke-summary.json),
[compute](compute.json), [protocol](../../configs/experiments/recurrent_alignment.json),
[guide](../../docs/recurrent-sizing-alignment.md),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/recurrent-alignment-2026-10-21/comparison.html),
[models and reports](https://github.com/jason490/RL-trading/releases/tag/recurrent-alignment-2026-10-21).

Overall robust multi-seed/prospective objective remains unmet. Next inspect saved
risk/action behavior before any different preregistered learning hypothesis.

Ledger observed debits0.08 units; rounded account balance decrease0.09 includes
metering delay. Conservative rate estimate/charge0.12225039 stays retained.

Publication verifies13 server assets. Cleanup removes48,871,879 duplicate bytes,
456 retained hashes unchanged. Strict report reproduces byte for byte after
cleanup; all weights, frozen bundles, execution counters, curves, source, data
and ledger remain. [Publication](release-verification.json), [cleanup](cleanup.json),
[reproduction](reproduction.json).
