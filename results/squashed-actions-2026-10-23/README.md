# Fixed squashed-action recurrent comparison

Status: policy and version31 study integration implemented; actual immutable
source construction/checkers pass. Colab smokes remain before100k. No performance
result or allocation yet. [Protocol](../../configs/experiments/squashed_actions.json),
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
