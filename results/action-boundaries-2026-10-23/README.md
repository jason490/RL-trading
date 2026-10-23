# Gaussian action-boundary diagnostic

Completed fixed read-only protocol on111 training symbols ×32 flat-account
observations per model (3,552 each;10,656 total). Pinned untrained full64 and
existing control selected/final weights, independent actor memory reset per
symbol, original frozen normalization and execution-masked observations.
No policy actions executed, fitting, optimizer steps, provider calls, Colab
allocation, validation performance scoring or retired-test evaluation.

| Policy | Any component outside[-1,1] | Exposure | Stop | Target near | Zone width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned untrained |78.28%|31.73%|31.73%|31.73%|31.73%|
| Existing selected |79.65%|34.28%|31.91%|34.50%|31.06%|
| Existing final |79.38%|36.23%|32.44%|37.67%|24.06%|

These are analytic Gaussian probabilities conditional on fixed training-prefix
states, not measured historical rollout clipping frequency. Joint probability
uses the conditional diagonal-Gaussian independence; probabilities averaged over
states must not be reconstructed from averaged component probabilities. No return
improvement or sole causal explanation is established. Flat states and early
prefixes are a deliberate controlled diagnostic, not representative training
visitation or the entire market history.

Installed RecurrentPPO2.7.0 clips raw Gaussian samples before environment steps.
The score-function estimator over raw actions remains a valid design; this audit
does not establish an implementation bug. The many-to-one action mapping motivates
a separate support-aware policy hypothesis. Prefer an explicitly squashed diagonal
Gaussian with the same means/log_std dimensions and unchanged network before
adding state-dependent exploration, which changes additional mechanisms. Verify
likelihood/Jacobian/entropy, actual rollout/predict/save-load/frozen contracts and
identical initial tensors before any paired Colab training. No gSDE or distribution
change was applied by this diagnostic; separate numeric preregistration required.

54 focused checks pass. Actual saved models/full Adam/Torch RNG/normalizer and
all protected source/data/model hashes unchanged. Project38.81646887; last official
account349.17, zero rate/assignments; prior four sessions stopped. No extra units.
Clipping/capacity/L2/guard families remain closed. Robust-return goal unachieved.

[Protocol](../../configs/experiments/action_boundary_diagnostic.json),
[summary/invariants](summary.json), [compact components](components.csv),
[method](../../docs/action-boundaries.md).
