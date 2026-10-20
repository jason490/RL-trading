# PPO policy architecture research

The direction replay found that the full-state PPO model changed its requested
sign on 14.70% of common unrestricted/short-only decisions with the same market
inputs. Account dependence can be appropriate for transaction costs and risk.
This study tests a narrower information contract rather than assuming that the
feedback causes losses or that the new policy will predict markets better.

## Information paths

All study arms use seven existing normalized market features followed by the
14 `execution_full` account/execution fields, for 21 inputs. Actions remain
signed target exposure, stop distance, target-near distance and zone width.

| Variant | Exposure output | Protective outputs | Critic |
| --- | --- | --- | --- |
| `native` | Standard SB3 actor on all inputs | Same actor | Full inputs |
| `two_branch_unmasked` | Both actor branches | Both actor branches | Full inputs |
| `two_branch_market_exposure` | Market branch only | Both actor branches | Full inputs |

The market actor branch reads only the first seven feature values. A second
actor branch reads the full observation. Each uses two Tanh layers, width 32 for
smokes and 64 for research, and their outputs are concatenated. The value branch
reads the full observation with the same two-layer width.

A fixed connection buffer masks the exposure row's full-branch weights in the
treatment. The unmasked control has the identical layout, allocated parameter
count and initialized tensors, with all connections enabled. The treatment has
fewer **active exposure connections by design**; equal parameter counts do not
mean equal functional capacity. The native arm controls for adding the branch
layout. No state information is removed from learned protective outputs or the
critic. The Gaussian log standard deviations remain trainable and state-free.

Exposure is desired total exposure. Execution still uses current equity,
whole-share affordability, current positions, corporate actions and protection
when turning it into an adjustment. Restricting the actor's exposure inputs does
not bypass accounting or repeatedly add a new full-sized order.

## SB3 implementation and compatibility

`rl/ppo/policies.py` subclasses the installed SB3 `ActorCriticPolicy`. It keeps
SB3's distribution, initialization, optimizer and PPO update implementation. A
wrapper around the existing initialized action layer multiplies its weights by
a registered fixed buffer; it creates no new parameters or random draw. The
optimizer retains the same parameter objects. Existing wide protective biases
are applied through the wrapper's bias property.

`policy_variant='native'` remains the default in `create` and `train`. Native
parameters, factory RNG and action predictions are tested bitwise against the
original SB3 factory path. Branch variants require PPO, flattened vector inputs,
four diagonal-Gaussian actions, two Tanh layers and the full execution profile.
Feature count and mask semantics cannot be overridden through generic kwargs.

Run metadata and frozen manifests record `policy_architecture`. Old bundles
without the field mean native. Loading checks the versioned specification,
market feature count, policy class, input shape, direction flag and exact mask
buffer. A branch policy cannot silently load as native or with the opposite
information contract. Custom models require repository source containing their
policy class; use the source revision recorded by the bundle. Selected/final
weights and immutable training-only normalization remain separate.

## Declared experiment

[exposure_architecture.json](../configs/experiments/exposure_architecture.json)
freezes the hypothesis, dataset, controls, budgets and numeric gates. The same
26-symbol development-only dataset and default costs/reward are used in every
arm. Training ends in 2020; validation is 2021–2023. Retired 2024–September 2026
tests are not loaded. Future declining-stock labels are reporting-only.

First run three 512-transition Colab CPU smokes. Then run all three variants for
50k PPO transitions, seed 11, n_steps 1000, batch 50, wide initialization and
legacy-uniform starts. Retain 10k-step curves plus final post-update evaluation.
Only the market-exposure treatment can be promoted, and only if it independently
passes against **both** controls: selected and final gains at least two percentage
points, no final stock/ETF group deterioration, and mean final drawdown worsening
at most two points. Only a passing screen permits all three arms at seeds 23/37.
Absolute holding comparisons and prospective confirmation remain separate.

Reuse existing study commands with the explicit architecture protocol:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_observation_jobs.py \
  --protocol configs/experiments/exposure_architecture.json \
  --phase smoke --output .runtime/architecture-smoke-jobs.json

PYTHONPATH=src .venv/bin/python scripts/report_observations.py \
  --protocol configs/experiments/exposure_architecture.json \
  --state .runtime/architecture-screen-state.json \
  --output artifacts/exposure-architecture/screen.json
```

The shared reporter verifies each arm's exact resolved configuration, so the
intentional policy kwargs differ while optimizer/data/cost settings match. New
version-10 Colab plans lock CPU, steps, profile and architecture. Archive checks
verify run/frozen architecture agreement and complete curves.

Use one committed source export and the original cumulative usage ledger across
all phases. Initial concurrency is two CPU sessions; these small networks do not
justify GPU allocation. The study allowance is two additional units, with the
existing 150-unit total ceiling, teardown by 130 and each allocation's two-hour
deadline. Check actual Colab Python before setup/training, monitor every minute,
retrieve durable artifacts, and verify all owned stops. Dependencies and CLI
installation are unchanged.

## Validation status

Local tests cover market-only output invariance, retained risk/value sensitivity,
zero account gradients for exposure, fixed mask gradients, exact paired initial
weights, optimizer parameter ownership, native RNG/weights, save/load/frozen
parity, incompatible architecture rejection, locked job plans and dual-control
promotion. Actual training is performed only in Colab. More observability or a
successful wiring smoke is not evidence of improved returns; report unsuccessful
screens and the [short-side diagnostics](short-side.md) as well.

Full offline regression passed **237 tests** (17 existing observation-bound and
deprecation warnings, no skips) before Colab allocation. This includes native
PPO/SAC/TD3/recurrent acceptance and the new policy/frozen contracts. The
architecture-specific screen is now complete and failed both promotion gates.

## Screen outcome

Native full-state selected/final return was 22.59%; unmasked branches 5.41%;
market-only exposure selected -1.02%, final -3.64%. The treatment lost on all
three declining stocks and failed both control comparisons. No confirmation or
longer training was added. All six owned CPU sessions stopped; six bundles pass
468 exact frozen predictions. Native training tensors/checkpoint scores exactly
reproduce the prior control. Project consumption is 35.748563/150.
[Full results](../results/exposure-architecture-2026-10-19/README.md).
