# Recurrent GAE trace comparison — failed screen

The fixed lambda1 candidate did not improve native RecurrentPPO over lambda0.95.
Both actual100k seed11 Colab CPU runs completed; artifacts were retrieved and
both owned sessions were verified stopped. The family is closed: no extension,
parameter grid or multi-seed confirmation is authorized by this failed screen.
The broader research objective remains unmet; no model is promoted.

## Performance

Average cumulative net returns across26 separate symbol episodes during repeatedly
used2021–2023 development validation; these are not annualized or independent test
results. Checkpoints were selected exclusively on development data.

| Model | Selected checkpoint | Selected return | Final100k return |
| --- | ---: | ---: | ---: |
| Control, lambda0.95 | 80k | 19.624533% | 18.484355% |
| Candidate, lambda1 | 100k before final update | 18.442942% | 12.675114% |
| Matched buy-and-hold | — | 30.601851% | 30.601851% |

The candidate loses1.181592pp at selection and5.809241pp at final versus control.
It fails the required2pp selected/final improvement and absolute holding gates.
Final mean drawdown worsens1.370980pp, within the2pp risk limit; passing that limit
does not offset failed return gates. Selected and final stock-median excesses
are−15.562510pp and−28.185010pp, below the−2pp requirement.

| Final group | Control | Candidate | Matched holding |
| --- | ---: | ---: | ---: |
| 22 stocks | 21.767430% | 14.366325% | 32.626547% |
| Four ETFs | 0.427443% | 3.373453% | 19.466022% |

The candidate's final average long/short contributions are+23.877780/−9.511455pp
for stocks and+10.328397/−6.954944pp for ETFs. Two of three retrospectively declining
stocks have both positive policy return and positive short P&L; the remaining one
does not. These cohort labels are reporting-only and never inputs or selection
criteria. Average short-side results do not establish robust profitable shorting.
Saved-path diagnostics are descriptive, not a causal explanation for deterioration.

## Fixed experiment

[Preregistered protocol](../../configs/experiments/gae_trace_preregistration.json)
SHA256:e8995cc951ffb4884b18d7b5e270f74b4ee987e54200be5494f7b00d40112a59.
Only GAE lambda differs. Same61,513-parameter64-wide actor/critic LSTMs, original
seed11 tensors and sampling RNG, seven market inputs/21 total observations, causal
training-only frozen preprocessing, original107stock+fourETF training episodes,
26-symbol development panel,512-session windows,100k transitions and10k evaluation.
Training ends2020; development2021–2023. Retired2024–September2026 data remains isolated.

Native Gaussian training exposure and unit-magnitude evaluation are unchanged,
as are gamma0.99, rollout1000, batch50, ten epochs, learning rate0.0003, net-log-equity
reward, costs, whole shares,100% cap,20-session maximum hold and protective geometry.
No new provider data, VIX inputs, reward shaping, broker orders or local learning.

All four runs use clean training source315614e6a8d8d470167f0d18d5ac9435c2bb46d5,
source hash e09754c3959202aec0e9aba0b3d0a0f7a710f6d1012edfdf0d375167fcd395d0.
Actual default Colab Python3.13.16 was inspected before dependency synchronization.
CPU training/evaluation durations: control915.28s, candidate996.82s. These are
training metadata durations, not complete allocation or billed durations.

## Acceptance and integrity

[Strict screen](summary.json) includes unchanged gates, complete10k learning curves,
stock/ETF metrics, declining-stock diagnostics and saved execution/accounting paths.
[Independent verification](screen-verification.json) recomputes four reports/104
episodes with zero return/group/drawdown discrepancy and rehashes218 protected files.

Both arms' initial tensors, empty Adam and Torch/NumPy/Python sampling starts agree;
only actual GAE lambda differs. Latest, selected and frozen learned Adam/model state
restore exactly.156 full-run frozen sequence predictions match, including recurrent
memory/reset/repeat behavior and immutable caller state/normalization. Schema/feed/
nonfinite inputs reject. The two earlier512-step smokes add156 predictions:312 total.

Every actual pre-update buffer is retained and independently rechecked:200 full
rollout arrays plus eight smoke arrays. Timeout corrections, terminal/reset masks,
float32 schemas and GAE targets validate; lambda1 segmented MC identity validates.
Largest full return-reconstruction errors:1.49e−7 control and4.50e−7 candidate, below
unchanged absolute2e−6/relative2e−5 tolerances. No optimizer update bypassed checks.

Historical control reproduction is exact: all registered checkpoint means and both
selected/final26-symbol panels have0.0pp discrepancy, within fixed0.01pp/0.05pp
limits. Matched holding paths and actual training sampling agree between arms.
The selected100k candidate is pre-final-update; final evaluation correctly uses
the saved post-update model. No test-driven checkpoint selection occurred.

## Compute, artifacts and cleanup

[Compute ledger summary](compute.json): cumulative conservative39.351004/150 units,
90.648996 working units remain to130; shared GAE study increment0.078547/2 including
smokes, setup, allocated idle time, retrieval and shutdown. Full-screen increment
0.058547 units. Original project ledger and study baseline are preserved.
Official post-cleanup account balance348.74, zero rate and zero active assignments;
account floor150/teardown170 remains enforced. No budget or metering stop occurred.

Full owned sessions `rl-stocks-cpu-aa182d96ba1a` and
`rl-stocks-cpu-eda5699ab896` are verified stopped; earlier smoke sessions
`rl-stocks-cpu-b33e0c990b01` and `rl-stocks-cpu-6b2b7ba4a483` are also stopped.
No cleanup recovery command is needed. Complete retained artifacts remain under
ignored `.runtime/gae-source-v1/artifacts/colab/`; the original ledger is
`.runtime/usage-ledger.json`. [Research release](https://github.com/jason490/RL-trading/releases/tag/gae-traces-2026-10-25)
contains14 server-verified assets, including four frozen bundles,208 actual arrays
and interactive reports. [Release verification](release-verification.json) records
exact sizes/hashes. [Cleanup](cleanup.json) removed nine exact duplicates totaling
290,074,611 bytes (276.64MiB), retaining all four differing checkpoint snapshots.
All1,002 protected files rehashed unchanged.
Models, source, original data, intermediate checkpoints, arrays and recovery records
must remain retained. See [smoke acceptance](smokes.json), [smoke budget](smoke-budget.json)
and [category guide](../../docs/gae-traces.md).

## Decision and next work

Do not extend this trace study, alter its gamma/rollout/epochs/learning rate or
promote a lucky intermediate checkpoint. Lambda1 did not solve the observed
learning instability or holding gap. This comparison does not prove that critic
bias caused prior failures or that all other algorithms fail. Publication and exact-duplicate cleanup are complete. Review a
distinct evidence-backed hypothesis before any further bounded experiment.
A future candidate still requires fixed multi-seed gates and prospective evaluation.

[Post-cleanup reproduction](reproduction.json) confirms exact semantic equality
of the strict screen summary from retained files after transfer deletion. All1,002
protected files rehash unchanged; no training, new sessions or extra units.
