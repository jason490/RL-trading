# CrossQ screen stopped: historical-control integrity failure

The fixed50k seed11 SAC/CrossQ screen is **not interpretable for promotion**.
At10k, fresh SAC mean development return is−3.138424% versus registered historical
−3.016847%:0.121578 percentage-point discrepancy exceeds the fixed0.01pp limit.
This fails a mandatory gate that later checkpoints cannot undo. No tolerance was
widened, no settings changed and no automatic retry was started. This is not a
completed performance screen or evidence that CrossQ underperforms.

[Read-only diagnosis](diagnosis.json) compares a pinned retained SAC checkpoint
with the original50k control replay. Initial policy tensors, all RNG streams,
dataset, simulation, reward, observations, sampling, execution and package
versions agree. The first1001 stored actions agree bitwise. First differing
action: zero-based row1001, target-near component2, absolute difference
5.960464477539063e−8/four float32 ULPs. Observations first differ at row1002,
rewards1090; done/timeout masks remain equal across19999 compared rows.
The pinned diagnostic snapshot contains20k on-step weights/19999 replay rows
and retains the failing10k curve point.

This localizes divergence immediately after learning begins; it does not prove
CPU arithmetic, optimizer representation or another implementation detail as
the root cause. Effective displayed optimizer settings agree; initial serialized
parameter-group hashes differ because defaults were made explicit. No local
learning/optimizer step was used to investigate. Source archive/replay/metadata
hashes remain unchanged. Next investigate numerical/backend provenance and
implementation equivalence before considering any separately reviewed protocol;
there is no authorized automatic replay of this failed integrity attempt.

[Cleanup](cleanup.json): both owned full-screen CPU sessions are verified stopped,
including bounded checkpoint retrieval before teardown. SAC20k checkpoint and
an independently pinned diagnostic archive are retained. No CrossQ full-screen
checkpoint was available: its salvage request returned404 before first durable
checkpoint; CLI/setup/stop/verification logs are retained. The two earlier
accepted4096 smokes and all their models/replays remain available.

Conservative project39.518547649665/150, remaining110.481452350335.
Screen increment0.02072982852326; whole CrossQ study0.04598595401644 against
unchanged2-unit baseline39.47256169564856. Official account348.60, rate0,
zero assignments. No budget or metering failure; original150/130 project and
150/170 account guards remain. No retired2024+ evaluation or broker orders.
Training source36e4dd4; protocol and research failures remain preserved.


The persistent `.runtime/crossq-integrity-stop.json` now blocks new CrossQ
shared-budget controllers and root-validated runtime profiles, including those
using earlier accepted smoke evidence. Even an empty/malformed marker fails
closed. It does not reset the original ledger or study allowance, or mark an
account/project budget failure. The marker and failed attempt must remain
preserved through any separately reviewed future protocol.
