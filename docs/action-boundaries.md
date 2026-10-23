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

## Implementation and verification

`reporting/action_boundaries.py` computes stable analytic lower/upper Gaussian
tails and per-observation joint probabilities. The recurrent reader carries
hidden/cell state per symbol, holds the account flat and restores policy mode.
The CLI binds the accepted summary, exact model/metadata hashes and original
immutable dataset; it fingerprints full Adam, policy tensors, Torch RNG,
normalization and source files before and after. It rejects validation episodes
and unsupported distributions. No policy action is executed or sampled.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
  scripts/diagnose_action_boundaries.py --output artifacts/action-boundaries/summary.json
```

54 focused checks pass: analytic normal tails/joint mass, asymmetry, invalid
parameters, extreme finite scale, real untrained recurrent carry/reset/repeat,
model/optimizer/RNG invariants and existing registered-study/action contracts.
Actual saved-model diagnostic remains to run; these tests are not performance
evidence. Protocol canonical SHA-256:
`f49c54441bec512c6b8170c0d8e437987aba08a503b22eab5cc31559e4036ba4`.


### Completed action-boundary diagnostic

Three fixed Gaussian policies ×111 training prefixes ×32 observations audited
without fitting/policy-action execution/validation-return scoring. Conditional
any-component mass beyond[-1,1]:untrained78.28%, selected79.65%, final79.38%.
Final marginals exposure36.23%, stop32.44%, near37.67%, width24.06%. Full model/
Adam/RNG/normalizer/source invariants pass;54 focused tests. These are conditional
probabilities, not historical rollout frequencies or a causal return diagnosis.
Next separately preregister a same-network squashed-diagonal-Gaussian comparison;
do not conflate with failed market-input clipping or add gSDE simultaneously.
No allocation or extra units. [Evidence](../results/action-boundaries-2026-10-23/README.md).

## Next fixed model comparison preregistered

[Squashed actions](../configs/experiments/squashed_actions.json) fixes two100k
seed11 arms: unchanged Gaussian control and explicit tanh-squashed diagonal
Gaussian recurrent policy. Keep61,513 parameters, identical initial tensors/RNG,
original111/26 dataset and preprocessing, masked21 observations, reward,
zero-decay Adam/eps1e-5, sampling/costs/holding and10k cadence. No gSDE, new
indicators, clipping/refit or simultaneous optimizer change. Correct transformed
log probabilities must be used during training, not just inference.

Both selected/final gains must reach2pp versus control, final stock/ETF results
must not worsen, drawdown worsening≤2pp; selected/final stock/ETF and median-stock
excess must each reach−2pp versus holding. Only a passing screen permits paired
11/23/37 confirmation and prospective evidence. Failure closes the fixed family,
without transform/temperature/log_std grids or extensions. Diagnostic mass is
motivation, not expected return or a causal conclusion.

Shared study allowance2units starts38.81646887344713; project150/130/account
150/170 and7200-second allocation limits unchanged. Full constructor, likelihood,
entropy, numerical saturation, actual rollout/frozen and legacy tests plus two512
Colab CPU smokes must precede100k. Implementation and allocation remain pending.

### Squashed policy implementation acceptance

`rl/recurrent_ppo/squashed.py` adds an explicit policy using SB3's maintained
`SquashedDiagGaussianDistribution` with epsilon1e-6. It transforms samples during
rollout and uses the matching inverse/Jacobian likelihood during PPO evaluation.
The Gaussian action head and four log-standard-deviation parameters retain their
names/shapes; the full64 initial policy has exactly61,513 parameters and the same
pinned tensors as control. Native algorithms remain unchanged. gSDE and unsquashed
overrides reject; the custom distribution requires explicit matching metadata.

The factory, frozen loader and recurrent verifier recognize
`squashed_recurrent_ppo`. Save/load/frozen action and hidden/cell parity pass.
A128-transition synthetic SB3 rollout verifies stored and reevaluated transformed
likelihoods, recurrent padding and finite entropy fallback without learning or
optimizer updates. This is local simulation acceptance, not local model training.
Analytic Jacobian/mean-transform, bounded samples and numerical saturation checks
also pass.70 focused policy/distribution/frozen/legacy checks pass (two existing
Gymnasium unbounded-Box warnings). Dedicated version31 study wiring and real
immutable-dataset construction/Colab smokes remain before100k allocation.

### Version31 integration acceptance

`training/squashed_actions.py` owns fixed source/parameter/distribution/Adam/
initialization/sampling/runtime acceptance. Version31 plans route each algorithm
explicitly; runner and retrieved artifacts retain exact snapshots and initial/
selected/final records. Record validation accepts the custom policy class only
for the declared candidate and keeps legacy defaults unchanged. Strict reports
restore both model/Adam phases, match holding paths and apply the frozen gates.
Long plans bind actual accepted smoke bytes and an under7200s projection.

125 focused new/legacy checks and a separate62-check shared-controller/frozen/
hybrid invocation pass (overlapping policy checks). Both actual immutable-dataset
checkers/full64 recurrent forwards and exact initial tensors/RNG pass without
local learning/optimizer updates. [Evidence](../results/squashed-actions-2026-10-23/construction.json).
Two512 Colab CPU smokes remain before100k. Original project38.81646887 and
account349.17 zero assignments; no allocation or performance result yet.


### Actual squashed-action Colab smokes accepted

Both512 CPU/defaultPython3.13.16 runs from cleanfa63d6c… complete;156 exact frozen
recurrent sequence/reset predictions, immutable normalization, actual saved models/
Adam, distribution flags/Jacobian and identical initial tensors/RNG/symbol/year
sampling pass.80 actual updates/40 nominal epochs,61,513 parameters each. Both
owned sessions retrieved/stopped. Runtime129.7219s control/159.4628s candidate gives
fixed3472.1638s projection including1200s reserve, below7200s. Exact accepted smoke
bytes bind prepared100k plans, not yet allocated. Project38.83857572/account349.16
zero rate/assignments, shared2-unit allowance/original project/account limits
unchanged. No smoke-return selection, promotion or retired-test access.
[Evidence](../results/squashed-actions-2026-10-23/README.md).


### 2026-10-23 — fixed squashed-action screen failed

Two100k Colab CPU runs completed; native control selected/final19.6245/18.4844%,
squashed15.4740/5.1381%, matched holding30.6019%. Relative and absolute stock/ETF/
median gates fail; final stock6.4716%/ETF−2.1964%. Fixed family closed with no
transform/temperature/log_std grid/extension/confirmation.312 frozen checks
including smokes, actual model/Adam/distribution/initialization/year proofs pass;
fresh control exactly reproduces prior tensors/sampling. All four owned sessions
retrieved/stopped. Project38.90354117/account349.10 zero assignments. Large
conditional boundary mass did not demonstrate a causal cure or policy improvement.
Full artifacts retained; publication and saved-path diagnosis next. Robust-return
objective unachieved. [Evidence](../results/squashed-actions-2026-10-23/README.md).
