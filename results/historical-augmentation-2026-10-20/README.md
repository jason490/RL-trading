# Historical ETF augmentation — smoke passed, screen running

Three immutable datasets preserve every original episode and the training-only
normalizer, with identical validation. Additional recent and earlier ETF arms
have matched expected ETF sampling mass. Earlier training adds8,776 independently
warmed and end-purged rows. [Preflight](preflight.json),
[protocol](../../configs/experiments/historical_augmentation.json),
[guide](../../docs/historical-augmentation.md).

Full-window sampling prevents short remaining data windows from confounding
nominal mass. Realized group starts/transitions are logged. Legacy sampler
semantics remain unchanged; no future losing-stock labels are used.

Three512-transition Colab CPU smokes completed and shut down successfully.
The fixed50k seed11 screen is now launched from the same frozen source.
No performance gate result or promotion is available yet.

Broader verification passes75 preparation/sampling/core/observation/reward tests
without fitting. Repeated preparation reproduces all three exact data identities.
Downtrend labels are not measured by full-window sampling; any compatibility
counters are explicitly marked unmeasured, not evidence of zero downtrends.

Version15 now binds dataset identities, fixed protocol, CPU jobs and full-window
sampling on the launcher and worker. Returned archives check PPO/init/state/
reward settings, checkpoint curves, complete frozen metadata and reconciled
symbol/history sampling counts. Invalid checksummed contracts fail. Sampling
statistics are copied before parity resets to prevent nested-count mutation.
Actual metered Colab acceptance is next; no fitting result is implied.

## Actual Colab smoke acceptance

All three smokes ran Python3.13.16,512 PPO transitions, with matched full-window
sampling. Archive contracts and234 exact frozen-prefix predictions pass;
normalization is immutable and incompatible inputs rejected.
[Summary](smoke-summary.json), [restoration](smoke-restoration.json).
All three owned smoke sessions stopped with cleanup verified. The conservative
smoke increment is0.01713346 units, project35.88837393/150. Separate account
balance350.60 and zero reported assignments/rate were checked after shutdown;
billing can lag. Retain all models/checkpoints until final publication and
duplicate verification; no files removed prematurely.

The screen shares the original study start35.87124047 and2-unit limit, rather
than resetting the allowance after smoke. Three CPU cells,50k steps each,
seed11;10k curves and selected/final weights will be checked. This is development
research, not profitability or shorting evidence.
