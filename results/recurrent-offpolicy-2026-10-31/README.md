# Completed recurrent off-policy comparison

**The fixed seed11 performance gate failed. No model is promoted, and this
GRU-SAC comparison family is closed without confirmation seeds23/37.** All three
100k cells passed physical artifact acceptance, all owned sessions were stopped
and shutdown verified, and the account is idle. The broader research objective
remains incomplete: no robust or prospective winner has been established.
Execution completed2026-10-30; independent full-matrix acceptance and fresh idle
account settlement were completed2026-10-31.

## Matched development results

These are mean net total returns across26 separate single-symbol episodes on
reused2021–2023 development history. They are not annualized, a combined portfolio,
an untouched test, or prospective trading evidence. Each selected checkpoint is
the earliest strict best mean validation return; the100k endpoint is separate.

| Package | Selected checkpoint | Selected return | Final100k return |
| --- | ---: | ---: | ---: |
| GRU-SAC candidate | 10k | 31.28% | 12.92% |
| Native SAC control | 10k | 32.31% | 30.01% |
| RecurrentPPO reference | 40k | 21.81% | 18.48% |
| Matched buy-and-hold | — | 30.60% | 30.60% |

| Package and phase | Stocks22 | ETFs4 |
| --- | ---: | ---: |
| GRU-SAC selected | 34.28% | 14.81% |
| GRU-SAC final | 13.93% | 7.33% |
| SAC selected | 35.44% | 15.13% |
| SAC final | 32.70% | 15.25% |
| RecurrentPPO selected | 23.08% | 14.81% |
| RecurrentPPO final | 21.77% | 0.43% |
| Matched holding | 32.63% | 19.47% |

Native SAC is strongest in this screen, but its ETF returns lag holding in both
phases. Its final median individual-stock excess is−2.70pp. It is not a robust
winner merely because its overall mean approaches holding.

## Exit-fill win rates

A read-only [diagnostic receipt](exit-fill-diagnostics.json) independently checks
all three archived final-validation reports against the recorded archive hashes.
Pooled winning exit fills are747/1,370 (54.53%) SAC,1,800/3,548 (50.73%)
RecurrentPPO and2,174/5,258 (41.35%) GRU-SAC. Partial exits count separately;
these are not complete-trade win rates or future-success probabilities. Entry
fees are excluded from realized exit-fill diagnostics but included in net equity
returns. These rates do not override failed return gates or establish readiness
for trading. No new training, checkpoint selection or retired-test access occurred.

## Gate outcome and interpretation

GRU-SAC needed at least2pp improvement over each control in both selected and
final phases, plus predefined group-return, median-stock and drawdown conditions.
Against SAC it falls1.03pp selected and17.10pp final. It beats RecurrentPPO9.47pp
selected but trails5.57pp final. Its selected ETF excess−4.66pp and median stock
excess−2.30pp miss the absolute−2pp floors. Final stock/ETF excess−18.70/−12.13pp
and median stock excess−16.91pp also fail. Slightly smaller final drawdowns than
the controls do not compensate for failed return gates.

Final mean maximum drawdown is−33.23% GRU-SAC,−33.87% SAC and−35.10%
RecurrentPPO. In the reporting-only cohort of three stocks with negative holding
returns, positive policy return plus positive short P&L occurs for0/3 GRU-SAC,
1/3 SAC and0/3 RecurrentPPO. Decliner labels never enter inputs, rewards,
sampling, checkpoint selection or promotion.

This is a package comparison, not an isolated causal test of memory. SAC and
GRU-SAC use full21 observations; RecurrentPPO masks execution-state fields.
Architecture, batch and sequence contracts differ. Single-seed failure closes
this fixed hypothesis; it does not establish that memory can never help trading.
More steps, extra indicators or reward/width grids are not authorized as a rescue
of this failed family. Any future hypothesis needs its own bounded protocol,
training-only justification and unchanged promotion safeguards.

## Verification and artifacts

- [acceptance.json](acceptance.json): independently revalidated three-cell matrix,
  fixed gate, ten sparse curve points per model, stock-level excess and short-side diagnostics.
- [run.json](run.json): clean source4a85618, package identity, Python3.13.18/CPU,
  per-session archive checksums/durations, accounting and dashboard checksum.
- [family-closure.json](family-closure.json): immutable failed-screen decision;
  local enforcement record also retained under `.runtime`.
- [cleanup.json](cleanup.json): duplicate removal and protected-file verification.

All eleven snapshots per model, initial tensors/RNG, exact configurations,
weights and optimizer moments, dataset/normalizer identity, saved economic paths,
selected/final reports and GRU direct/frozen carried-state parity pass. GRU-SAC
and SAC each complete99,000 actor/critic/entropy Adam steps; RecurrentPPO completes
20,000 actual Adam steps and1,000 nominal epochs. Finite-gradient/weight checks
and all accounting constraints remain enforced. No final-test data was opened.

Additionally,42 focused protocol/gate/physical-verifier/status tests pass.
The actual matrix revalidation, curve/group arithmetic, receipt bindings,
idle-account accounting, dashboard checksum and documentation links were
independently checked. `plan.html` retains its original SHA256.

The standalone interactive dashboard is
`artifacts/recurrent-screen-acceptance-v1/comparison.html`. It shows selected/final
stock and ETF returns, complete checkpoint curves and individual-stock excess.
Large models/reports/archives remain under
`.runtime/recurrent-screen-source-v1/artifacts/colab/<session>/` and are excluded
from Git. The recorded checksums identify them; the original data and portable
starts remain retained. Earlier profile/control evidence remains in the
[previous dated folder](../recurrent-offpolicy-2026-10-30/README.md).

## Credits and cleanup

Screen conservative estimate/charge: **0.36394199 units**; reported measured debit:
**0.31000000 units**. Use the larger charge, retaining allowance for delayed
metering. Original study usage including both profile attempts is
**0.38972324/2 units**; no baseline or historical charge was reset.

Project usage: **39.92585919/150**, remaining **110.07414081**. Teardown threshold130
and account minimum150/teardown170 remain unchanged. Fresh idle account reading:
**348.27 units**, rate0, assignments0. All three owned session terminations are
verified. No cleanup recovery is required.

Removed137,450,298 bytes of duplicate screen uploads after matching every retained
input. Final GRU cleanup matched814 members and preserved139 protected files,
including models, reports, raw archives, dataset, starts, original claims/stop
records and the shared ledger. Full scientific artifacts remain retained.

## Next research step

Do not allocate confirmation or extend this family. Review the
[completed-control path diagnosis](../recurrent-offpolicy-2026-10-30/control-path-diagnostics.json)
and [historical identity audit](../recurrent-offpolicy-2026-10-30/ppo-historical-identity.json)
before another training hypothesis. RecurrentPPO's identical final model already
has failed fitted-period comparability evidence; repeating that replay adds no
independent evidence. A separately fixed inference-only fitted-period diagnosis
for the new GRU/SAC states can distinguish weak fitted performance from a
training/development contrast, without proving an optimizer cause. Retired2024+
history stays closed; any eventual promotion still requires multi-seed gates
and prospective confirmation. The overall improvement goal remains active.
