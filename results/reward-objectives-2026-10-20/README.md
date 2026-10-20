# Reward objective study — smoke accepted; screen running

Both reward variants completed 512-transition Google Colab CPU smokes using
Python 3.13.16. Artifacts were retrieved and both owned sessions stopped and
verified. Two bundles pass 156 exact CPU frozen predictions with deterministic
repeat, immutable normalization and incompatible schema/feed/input rejection.
These are wiring checks, not profitability evidence.

[Smoke receipt](smoke-acceptance.json) records exact source, models, checks and
usage. The committed [protocol](../../configs/experiments/reward_objectives.json)
requires a matched seed11 50k screen next, using the same frozen source and
shared original ledger. Seeds23/37 require the predefined screen gate.
No retired-test data or broker orders are used.

## Completed control

The unchanged log-reward seed11 control completed 50k transitions at 22.59448%
selected/final development return. All checkpoint scores and all 13 tensors
in both selected and final policies exactly reproduce the prior native PPO
control ([parity evidence](native-control-parity.json)). Its artifacts were
retrieved and its owned session stopped. The equity-change candidate is still
running; no matched gate or profitability conclusion is available yet.
