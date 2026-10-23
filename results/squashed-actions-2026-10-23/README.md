# Fixed squashed-action recurrent comparison

Status: policy/version31 integration and actual Colab512 CPU smokes accepted.
Fixed100k paired screen plans prepared, not allocated. No performance result yet. [Protocol](../../configs/experiments/squashed_actions.json),
[method](../../docs/action-boundaries.md), [construction](construction.json).

Compare native Gaussian versus explicitly tanh-squashed diagonal Gaussian during
both rollout sampling and PPO likelihood evaluation. Same full64 actor/critic
LSTM/64x64 heads,61,513 parameters and exact pinned initial tensors/RNG, original
111/26 full-history dataset/seven-field normalization,21 masked observations,
zero-decay Adam/eps1e-5, reward/actions/costs/20-session holding/100k/10k cadence.
No input clipping/refit, gSDE, indicators or simultaneous optimizer change.

125 focused policy/protocol/artifact/report/legacy/account-floor checks pass;
a separate62-check shared-controller/frozen/hybrid regression invocation passes
(including some overlapping policy checks). Actual immutable dataset environments
pass Gymnasium/SB3 checkers and finite full64 forwards. Initial tensor identity,
correct transformed distribution/Jacobian/entropy fallback,128-transition
synthetic recurrent rollout stored/evaluated likelihood parity, save/load/frozen
memory parity and actual optimizer contracts pass without local learning/updates.

Version31 plans/worker/runner retain exact initialization snapshot, actual initial/
selected/final model/Adam/distribution records and measured year sampling. Strict
retrieved artifacts/report reject wrong support/class/Jacobian/algorithm, missing
checkpoint evidence or mismatched source/data/sampling. Long plans require exact
accepted smoke bytes and runtime projection≤7200; existing control defaults and
legacy metadata formats preserved. Two512 CPU Colab smokes required first.

Selected+final relative2pp, final stock/ETF nonworsening, drawdown2pp and separate
stock/ETF/median−2pp versus holding gates unchanged; only a pass allows paired
11/23/37 then prospective confirmation. Failed family closes without transform/
temperature/log_std/model/reward/step grids or extensions. Conditional Gaussian
boundary diagnostic motivates this hypothesis but is not causal/performance proof.

Shared2-unit study baseline38.81646887344713; project150/130/account150/170 and
7200s per-allocation safeguards retained. Official preallocation account349.17,
zero rate/assignments; all prior owned sessions stopped. Retired2024+ unopened;
no provider requests or orders. Overall robust-return objective unachieved.


## Actual Colab smoke acceptance

Both512 CPU runs from cleanfa63d6c90b347c51d460e82e6d08dd6f757e6b23 completed
on actual defaultPython3.13.16, retrieved and verified stopped.156 exact frozen
action/hidden/cell/reset/repeat predictions pass, with immutable normalization and
schema/feed/nonfinite rejection. Actual selected/final models and complete Adam
records restore,80 actual updates/40 nominal epochs each. Both61,513-parameter
policies load exactly the same initial tensors/CPU RNG; symbol/year sampling and
checkpoint proofs match. Candidate actual transformed distribution/Jacobian/
policy flags validate; control remains native. No smoke returns selected settings.

Control duration129.7219s, squashed159.4628s include training/evaluation and do not
prove a speed ranking. Fixed projection3472.1638s (57.9min,1200s reserve included)
passes7200s. Exact accepted-smoke bytes bind two prepared100k seed11 plans; runtime
projection is heuristic, existing minute metering/deadline/aggregate guards remain.

Conservative project38.83857572/150, remaining111.16142428; study increment
0.02210685 including setup/idle/transfers/shutdown versus rounded account debit
0.01. Official account349.16, zero rate/assignments. Both owned sessions stopped;
source/data/models/checkpoints/reports/transports and original ledger retained.
Original2-unit baseline38.81646887344713 and all150/130 project/150/170 account
safeguards retained. No retired-test access or orders. Next fixed100k paired screen;
no promotion or return claim from smoke acceptance.

[Acceptance](smoke-summary.json), [frozen restoration](smoke-restoration.json),
[runtime approval](runtime-profile.json), [metering and cleanup](smoke-compute.json).
