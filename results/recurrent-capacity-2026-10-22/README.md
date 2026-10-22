# Fixed recurrent-capacity study

Status: protocol, source integration and actual local construction/checker
acceptance pass. No Colab allocation or training/performance results yet.

[Method and gates](../../docs/recurrent-capacity.md),
[fixed protocol](../../configs/experiments/recurrent_capacity.json),
[actual construction checks](construction.json).

Control64 versus larger128 recurrent/dense widths have61,513/221,321 trainable
parameters. Same21 observations, original seven market features, expanded111
training/original26 development dataset, original normalization, net-log-equity
reward, actions, short capability,20-session limit, costs, sampling and100k budget.
No failed session-feature family extension or larger-width grid.

Both actual full-architecture environments pass Gymnasium/SB3 checkers, exact
untrained model-record/optimizer checks and recurrent forward prediction. Per-arm
pinned tensors restore; initial CPU sampling RNG is identical across widths.
Shapes differ, so cross-width tensor equality is not claimed. No local learning
or optimizer step occurred; construction is not profitability evidence.

96 focused tests pass, including full-width smoke/screen artifact fixtures,
wrong width/optimizer/snapshot/schema contracts, required checkpoint evidence,
actual acceptance-byte binding, stale/over-deadline profile rejection, report
routing, confirmation gating and legacy session/guard/account-floor regressions.
Static archive fixtures do not claim actual training. Source compilation and two
actual immutable smoke payload validations pass without allocation.

Next commit/freeze source, check official usage/default Python and run two512
Colab CPU smokes using the full64/128 architectures. Require156 exact frozen
predictions, actual optimizer/initialization/sampling/checkpoint restoration and
verified owned shutdown. Long plans cannot be created without accepted smoke
metadata and cannot pass controller validation without matching actual acceptance
bytes. The fixed conservative runtime projection must fit7200 seconds; otherwise
no screen or automatic hardware restart. Smoke returns do not tune settings.

Project38.51487741/150; last official account349.41 zero rate/assignments.
Original shared ledger and2-unit study baseline38.514877408036206 retained.
All owned sessions are stopped; no new usage.


## Actual full-width CPU smokes accepted

Two512 Colab CPU runs from clean39f94a0/defaultPython3.13.16 completed, retrieved
and verified stopped.156 exact frozen action/hidden/cell/repeat/reset predictions
pass; normalization/input rejection, saved selected/final model/optimizer and
checkpoint identities pass. Actual sampling matches. Per-arm tensors match pinned
starts and CPU RNG matches across widths; cross-width weights differ as intended.

Both full architectures trained:61,513/221,321 parameters,40 nominal PPO epochs
and80 actual Adam steps. Smoke returns did not choose settings. Measured
train/evaluate duration control140.9608s, larger136.9381s; evaluation and hardware
variation prevent interpreting that as a training-speed advantage. The protocol
clamps the duration ratio to at least1.0. Conservative projected screen allocation
3015.9971 seconds (50.3 minutes), including1200-second setup/retrieval/shutdown
reserve, passes7200 seconds. This remains a heuristic, not a runtime guarantee.

The exact smoke-acceptance bytes and runtime projection bind prepared100k plans
and controller pre-allocation checks. Fixed same-source two100kseed11 screen is
next; no width/step/feature/reward adjustment or model promotion from smokes.

All two owned sessions stopped. Project38.53588481/150; study conservative
increment0.02100741 includes setup/idle/downloads/shutdown. Official account
349.40, zero assignments/rate. Original study baseline38.514877408036206/2-unit
allowance and project150/130/account150/170 safeguards remain. No orders/test access.

[Acceptance](smoke-summary.json), [frozen checks](smoke-restoration.json),
[compute](smoke-compute.json), [runtime profile](runtime-profile.json).
