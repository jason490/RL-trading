# Gaussian action boundaries

## Question and fixed diagnostic

Native RecurrentPPO uses an unbounded diagonal Gaussian and clips sampled actions
to the environment's[-1,1] Box before execution. This is distinct from market
input clipping, which failed its fixed experiment. The four components still
represent signed target exposure, stop distance, target-near distance and zone
width. No broker orders or action semantics change in this diagnostic.

The [fixed protocol](../configs/experiments/action_boundary_diagnostic.json)
compares the pinned untrained64-unit control with its existing selected/final
weights. Inspect32 observations from the beginning of each of111 training
symbol episodes, with immutable normalization and execution-masked state. Use a
fixed zero-action account path; never execute policy outputs, pick outcome-ranked
stocks/dates or score validation returns. Reset each model's recurrent memory per
symbol and carry it only within that prefix.

Compute exact Gaussian marginal mass beyond each bound and conditional joint
mass beyond any bound. These are probabilities conditional on the observed
prefix/state, not measured clipping frequencies from the original training
rollouts. Summarize means/standard deviations and prove weights, complete Adam,
normalizer and source checksums unchanged. No fitting, optimizer updates,
provider requests, Colab allocation, retired-test access or checkpoint selection.

## Library compatibility before future training

The installed SB3/contrib2.7.0 source is authoritative for current code. The
[RecurrentPPO documentation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/ppo_recurrent.html)
describes state-dependent exploration and output squashing. Any bounded-policy
experiment requires separate preregistration, correct rollout/log-probability
handling, actual constructor/save-load/frozen checks and Colab smoke acceptance;
simply changing an inference action transform is not a matched training study.
No distribution, exploration coefficient or model is selected by this diagnostic.
Existing failed clipping, capacity, L2 and guard families remain closed.
