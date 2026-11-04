# TRPO engineering feasibility

This is a distinct optimizer-feasibility study. It does not reopen failed PPO
safeguards, SAC experiments, CrossQ's integrity stop or forecast families. No
performance comparison or Colab allocation is admitted yet.

## Motivation and limits

Saved RL endpoints repeatedly deteriorate despite valid artifacts, sometimes
without higher costs; supervised residual forecasts also fail their fixed gates.
These observations establish neither a sole optimizer cause nor an information
edge. They justify a limited engineering question: can the maintained TRPO
implementation provide a reproducible, affordable alternative update mechanism
before another matched trading comparison is specified?

TRPO uses conjugate-gradient curvature calculations and KL-constrained backtracking
line search; PPO clipping and an approximate-KL early-stop check use different
update procedures. The [original TRPO paper](https://arxiv.org/abs/1502.05477)
distinguishes theoretical guarantees from the practical approximations and tests
robotics/Atari tasks. It provides no stock-trading guarantee. The
[installed-version API documentation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/trpo.html)
supports continuous Box actions, CPU, save/load and deterministic prediction,
with no recurrent policy. Project inference: a different update method is worth
bounded feasibility work; it cannot create missing predictive information.

## Fixed contract

[Engineering protocol](../configs/experiments/trpo_feasibility.json) SHA256:
`ba19cae92e039c0f6d86e01d8309203899835596daefa6abc1ea4245b706c7ab`.

| Contract | Fixed value |
| --- | --- |
| Implementations | Existing SB3PPO2.7.0 / SB3-ContribTRPO2.7.0 |
| Input/output | execution_full21 fields / four float32 actions in−1…1 |
| Actor/value | Separate64×64 tanh MLPs, orthogonal initialization, log_std0 |
| Shared rollout |1000 transitions, critic batch50, gamma0.99, GAE0.95 |
| Shared value learning rate |0.0003 |
| PPO |10 epochs, clip0.2, entropy coefficient0 |
| TRPO |CG15, damping0.1, line-search shrink0.8, maximum10 trials |
| TRPO constraint/value |KL target0.01,10 critic passes, no subsampling |
| Seeds |11/23/37 |
| Smoke |1024 transitions, rollout128/batch32, same64×64 networks |
| Research timing |2000 transitions, original rollout1000, two nominal updates |

Learning-rate interpretation differs: TRPO's setting controls its value optimizer;
its actor uses the constrained step. Equal nominal epochs/critic passes do not
imply equal actor work. Count accepted actor line searches, conjugate-gradient/
Hessian work, nominal updates and actual critic Adam steps separately. A smaller
KL or fewer actor updates is not an economic success criterion.

The existing seven market features, simulator, net-equity reward, costs, whole
shares,100% cap,20-session holding limit and learned stops/target zones remain.
Future performance registration must bind actual data/normalizer/sampler hashes,
fixed training selection and selected/final development gates. Compare matched
PPO plus a strong relevant SAC control with comparable selection; merely beating
a weaker PPO does not satisfy the benchmark/stock/ETF objective. Include all
short-side failures and preserve original multi-seed/prospective requirements.

## Implemented local acceptance

`training/trpo_feasibility.py` validates the fixed protocol and strict21/4 Box
contract. It constructs CPU native TRPO/PPO using existing locked dependencies;
there is no fitting entry point. TRPO remains absent from the generic algorithm
registry so existing commands cannot accidentally start a performance study.

Four focused tests pass without optimization. For all three seeds they prove:
complete common actor/value/log_std initialization,11,465 policy parameters,
empty optimizer state/zero transitions, equal deterministic actions, exact local
prediction repeats, untrained save/load parity, unchanged prediction weights,
and rejection of altered initialization and incompatible geometry/settings.
The fixture refuses any environment step and monkeypatches learning to fail.
These are local engineering checks, not Colab training or cross-machine parity.

## Remaining acceptance before allocation

1. Inspect actual default Colab Python and match it with the current lock; reuse
   the official CLI and bounded bootstrap. No older interpreter or dependency grid.
2. Specify/export frozen inference for the tanh hidden layers and directly clipped
   Gaussian mean. Existing SAC reference actor assumes a different input/activation/
   output contract; it cannot be relabeled as TRPO. Preserve its accepted controls.
3. Verify actual Colab finite updates, critic moments/counters, line-search
   acceptance/rollback, Hessian statistics, save/load/frozen actions and unchanged
   preprocessing. Register one bounded smoke and research-geometry timing study.
4. Require a conservative measured projection covering setup, learning, evaluation,
   transfer and shutdown before any performance screen. Preserve shared150/130
   project and150/170 account protections, prior charges and owned cleanup.
5. Preregister the complete matched experiment and original stock/ETF/median/
   drawdown/selected-final multi-seed gates before observing new development scores.
   Retired2024+ data remains closed; prospective confirmation is still necessary.

No Colab session, model fitting, provider request, retired-test read or new credit
charge accompanied this engineering step. Project usage remains40.213170920/150.


## Frozen actor arithmetic implemented — 2026-11-04

[Reference contract](trpo-reference-inference.md) now implements21-field
tanh-hidden/direct-clipped-mean inference and trusted native PPO/TRPO export.
26 focused checks pass, including existing SAC regressions and isolated
stdlib-only prediction. A retained untrained128-point synthetic probe takes
1.4174s locally; native/reference max difference5.59e−9 is disclosed, not relabeled
exact. Cross-machine/full-path/preprocessing acceptance and actual Colab learning
smoke/profile remain pending. No allocation or optimization occurred.


## Arithmetic probe prepared — 2026-11-04

The [fixed cross-machine probe](trpo-reference-inference.md) binds the retained
untrained actor and128 synthetic vectors/512 reference output coordinates.
It uses one separately capped0.05-unit CPU inference-only allocation, original
shared ledger/account protections and mandatory cleanup; no installation or
optimization.38 focused probe/actor/constructor/lifecycle checks pass. Actual
Colab arithmetic acceptance is next; learning smoke/performance stay unadmitted.
