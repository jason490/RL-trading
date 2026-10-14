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

`RL_CHECKPOINT_ARCHIVE` enables an atomic archive after validation; the remote bootstrap sets this path. The Colab watchdog retrieves it into durable local storage with bounded waits. This is best-effort periodic transfer; loss before the first successful download cannot be repaired from an ephemeral VM after termination. Final archive retrieval and cleanup are separately recorded.

Replay buffers permit continued SAC/TD3 learning with recorded settings. They do not guarantee restoration of Python/NumPy/torch RNG, environment episode state or exact optimizer scheduling across interruption. Such exact-resume support remains a separate extension; do not describe the current replay save as that capability.

Runs record hardware/backend nondeterminism. Save/load parity is checked on the same device/observation; it is not a claim of bit-for-bit parity across CPU/A100 or different torch backends. Training smoke demonstrates wiring, not profitable learning. Dependencies and real smoke acceptance remain pending in this sandbox.
