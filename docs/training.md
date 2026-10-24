# RL training, checkpoints and reproducibility

The shared factory validates an algorithm name (`ppo`, `sac`, `td3`) and retains algorithm-specific parameters in separate owned modules. Models use SB3 `MlpPolicy`, learning rate 3e-4 and gamma .99. These are experimental hypotheses, not proven trading optima.

| Setting | PPO | SAC | TD3 |
| --- | --- | --- | --- |
| Default policy layers | [64,64] | [64,64] | [64,64] |
| Rollout / replay | n_steps 2048 | buffer 100000 | buffer 100000 |
| Batch | 64 | 256 | 256 |
| Updates | 10 epochs | train_freq 1, gradient_steps 1 | train_freq 1, gradient_steps 1 |
| Warmup | On-policy rollout | 1000 transitions | 1000 transitions |
| Specific | GAE .95, clip .2, entropy 0 | tau .005, entropy auto | tau .005, delay 2, smoothing .2, clip .5 |
| Exploration | Stochastic learning policy | Learned entropy | Gaussian noise sigma .1 per action component |

Smoke policy layers are [32,32]. PPO smoke uses rollout 128 / batch 64 / exactly 512 transitions. SAC/TD3 smoke uses batch 64, warmup 64 and exactly 1024 transitions so gradient updates are exercised. Rollout/batch divisibility and exact PPO transition multiples are required.

Long configuration is 50,000 transitions for seeds 11/23/37, via `train ... --long --seed SEED`. For an exact 50,000 PPO budget, the CLI resolves rollout 1000 / batch 50. Other long defaults remain as above. The Colab runner defaults to smoke acceptance; `--research ppo` appends one seed-11, 50,000-transition model after all smoke jobs succeed, within the same budget and deadline. Additional seeds remain separate experiments. See [verification.md](verification.md) for completed execution.

## Runtime selection

Run dependency checks, environment checkers and tiny PPO/SAC/TD3
train/evaluate/save/load demonstrations on CPU first. Use T4 or another suitable
lower-credit GPU for GPU-specific smoke checks when needed. Prepare the verified
dataset, dependency lock and resolved configuration before requesting higher-credit
hardware. See [Colab runtime selection](colab.md#runtime-selection-profile-before-choosing-hardware).

Prefer **Colab CPU** for these small MLPs, following the user's 2026-10-15
hardware change. A slower GPU may have similar wall time when sequential
simulation/validation dominates; compare complete operation times and reported
units rather than GPU names. A100 is an explicit option when profiling justifies
it. CPU/CUDA training can differ numerically; record the device for each run.
All stages retain the 150-unit ceiling and 130-unit teardown threshold.

## Lifecycle

1. Verify the checksummed local dataset; separate training episodes from validation/test.
2. Seed Python, NumPy, torch and spaces; set one CPU thread for the small MLP profile.
3. Run Gymnasium and SB3 checkers; reject unavailable requested CUDA.
4. Create a unique run directory and record status `running`, resolved settings, package versions, dataset, seed/device, Git availability, package-source hash and selection metric.
5. Learn on training episodes only, with validation checkpoints every 256 smoke / 10,000 long transitions.
6. Save periodic weights, best-by-validation weights and validation HTML/JSON/history. Off-policy checkpoints save a resumable replay-buffer snapshot.
7. Validate final post-update weights, check finite parameters and save latest/model replay.
8. Freeze selected best weights, verify deterministic save/load prediction parity, and optionally audit the test after freezing.
9. Record `complete` or `failed`, actual steps and duration. Interrupted weights remain reviewable; no failed run is hidden.

The callback's on-step timing matters for PPO: at a rollout boundary it runs before the optimizer update. Final validation is unconditional after learning, so final updated weights are not accidentally excluded just because a checkpoint shares their transition count.

## Selection and durable artifacts

Checkpoint score is equal-weight mean net validation return. It is declared before training; no callback accesses test episodes. A higher shaped reward never substitutes for portfolio evaluation. Save latest and best separately, and keep unsuccessful metadata for comparisons.

`validation.html`/`.json` describe the most recent validation call, including the
final post-update weights. They can differ from the selected `best.zip` bundle.
New runs automatically save `selected-validation.html`/`.json` when validation
selects better weights, plus the selected transition count and per-symbol metrics
in metadata. Keep a separately labeled report when replaying selected weights,
record its device, and never interpret the final-weights chart as the best bundle's chart. The initial A100 PPO run has a selected-weights CPU replay report linked
in [verification](verification.md).

`RL_CHECKPOINT_ARCHIVE` enables an atomic archive after validation; the remote bootstrap sets this path. The Colab watchdog retrieves it into durable local storage with bounded waits. This is best-effort periodic transfer; loss before the first successful download cannot be repaired from an ephemeral VM after termination. Final archive retrieval and cleanup are separately recorded.

Replay buffers permit continued SAC/TD3 learning with recorded settings. They do not guarantee restoration of Python/NumPy/torch RNG, environment episode state or exact optimizer scheduling across interruption. Such exact-resume support remains a separate extension; do not describe the current replay save as that capability.

Runs record hardware/backend nondeterminism. Save/load parity is checked on the same device/observation; it is not a claim of bit-for-bit parity across CPU/A100 or different torch backends. Training smoke demonstrates wiring, not profitable learning. See [verification.md](verification.md) for completed CPU and remote acceptance.

Colab payloads include the originating Git revision and dirty-state record,
captured at upload. Remote runs use that record even though the payload omits
the Git database, and also compute a hash of the actual package source.

## RecurrentPPO

`--algorithm recurrent_ppo` selects SB3-Contrib's `MlpLstmPolicy`. The same causal
observations and four-component action are used. Actor and critic have separate
single-layer LSTMs (64 hidden units for research, 32 for smoke) and the existing
MLP head widths. The local smoke accepts 512 transitions. Research PPO variants
use rollout length 1000 and batch size 50, so requested steps must be an exact
rollout multiple. RecurrentPPO is on-policy and does not save a replay buffer.

Validation maintains hidden/cell state within each episode and resets it at each
new stock episode. It never changes the training collector's LSTM state. Frozen
export verifies the action and state from an explicitly reset prediction. The
[inference guide](inference.md#recurrent-bundles) covers sequence use and state
validation; [algorithm stability](algorithm-stability.md) describes the controlled
CPU comparison. Memory adds model capacity and is not established as an improvement.

## Checkpoint identity during interruptions

New runs write `checkpoint-validation.json` at every evaluation, including a
worse non-selected checkpoint. `checkpoint-state.json` is atomically published
last, before the existing atomic snapshot archive, and binds latest/selected
model filenames to report filenames with SHA-256 hashes, transition counts and
`on_step`/`post_update` phases. Only the declared final post-update evaluation
sets `final_post_update`. Selected reports/weights remain the best checkpoint;
producing current interruption evidence does not change selection or rewards.

Legacy `validation.json`/HTML can stay at a prior selected checkpoint in compact
runs until final reporting. Do not infer their identity from the newest learning
curve. Use the explicit checkpoint-state mapping and current checkpoint report.
`read_checkpoint_state(run_directory)` rejects corrupted/mismatched files,
non-validation reports, unsafe filenames, invalid phases and inconsistent scores.
A new complete archive containing this manifest must have the final post-update
marker and agree with recorded final/selected steps and curve score. Old archives
without the manifest remain readable; their paths require separate identity
verification and may be unaligned, as in the interrupted regularization study.

Current JSON evidence adds bounded storage per run; HTML generation remains at
existing selected/final cadence. Published research model archives include both
new files when present. Callback regression tests use fake saves/evaluations,
never local optimizer updates. Actual remote acceptance of this new format is
pending further authorized Colab work; the existing budget-stop flag stays set.

For the Gaussian update-guard study, [shared initialization](recurrent-update-guard.md) pins untrained tensors and CPU RNG across hosts before fitting, while retaining exact equality gates and initial snapshots. Independent same-seed construction was insufficient in the rejected first smoke comparison. Initial RNG provenance does not promise exact continued-learning restoration.

[Recurrent-capacity comparison](recurrent-capacity.md) specifies the fixed
64/128-width hypothesis, shared untrained RNG starts and resource gates.

The native `tqc` factory configuration and frozen critic contract are implemented
for the [registered distributional-critic study](distributional-critics.md).
Local construction/serialization checks perform zero optimizer steps. Shared
initialization, central runtime/archive integration and actual Colab smokes remain
pending; this base API acceptance is not permission to skip those study gates or
claim a trained TQC result. Existing SAC defaults are unchanged.
