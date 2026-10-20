# Historical ETF augmentation — prepared, not trained

Three immutable datasets preserve every original episode and the training-only
normalizer, with identical validation. Additional recent and earlier ETF arms
have matched expected ETF sampling mass. Earlier training adds8,776 independently
warmed and end-purged rows. [Preflight](preflight.json),
[protocol](../../configs/experiments/historical_augmentation.json),
[guide](../../docs/historical-augmentation.md).

Full-window sampling prevents short remaining data windows from confounding
nominal mass. Realized group starts/transitions are logged. Legacy sampler
semantics remain unchanged; no future losing-stock labels are used.

No model trained, Colab runtime allocated, performance result or promotion.
Project usage remains35.871240/150. Complete metered worker/job/artifact
integration and freeze committed source before the fixed three-arm smoke/screen.

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
