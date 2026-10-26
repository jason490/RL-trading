# CrossQ implementation feasibility

The initial untrained source/serialization audit passes, and the separate fixed
performance protocol below is now preregistered. CrossQ now has restricted factory/frozen support. Generic training remains
blocked until registered learned-state/runtime acceptance is connected.
Treasury/VIX/GAE failures stay closed; no rate inputs or reward changes are mixed
into this algorithm proposal. Existing project150/130 and account150/170 safeguards
continue, with original usage39.472562 units after Treasury cleanup.

## Evidence and rationale

The [CrossQ paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf)
uses normalization and removal of target networks in continuous-control tasks.
It supplies no stock-return guarantee. The installed implementation is checked
against [SB3-Contrib2.7.0 documentation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/crossq.html).

Installed native defaults differ from SAC beyond target networks: CrossQ actor/
critic Adam beta1 is0.5 versus SAC0.9; CrossQ policy delay3 versus SAC updating
its actor each critic update. CrossQ's native learning rate is0.001 versus our
common audit override0.0003. Default actor256×256/critic1024×1024 substantially
exceed the existing small research networks. Recurrent policies are unsupported.
A future comparison must explicitly distinguish an algorithm-package comparison
from a claim that normalization alone caused improvement.

## Fixed untrained audit

[Audit specification](../configs/experiments/crossq_implementation_audit.json)
fixes synthetic21-dimensional observations, four bounded actions, seed11,64×64
actor/critic linear widths, five deterministic predictions and a tiny empty256-row
replay allocation. No environment reset/step, provider, dataset or optimization is
permitted. Tests forbid both learners' learn/train and Adam.step calls.

| Geometry | SAC | CrossQ |
| --- | ---: | ---: |
| Actor parameters | 6,088 | 6,386 |
| Live critic parameters | 11,778 | 12,390 |
| Separate target critic | 11,778 | 0 |
| Normalization modules | 0 | 9 |

Same linear widths do not give identical trainable parameter counts: CrossQ adds
910 scale/bias parameters. Nine BatchRenorm layers track means/variances/counters;
all counters stay0 under evaluation. Native warmup100,000 updates means a short
screen can remain in BatchNorm behavior; actor/critic update rates differ, and the
renormalization branch uses a strict greater-than warmup check. Do not claim that
an unexercised branch was tested or retune its warmup after seeing returns.

The audit checks actual tensors/buffers, empty actor/critic/entropy optimizer state,
empty replay round-trip, deterministic predictions, finite shape and caller RNG
restoration. Temporary untrained model/replay files are removed after comparison.
This proves neither learned optimizer restoration nor throughput. Record installed
implementation hashes and actual audit evidence only from a clean committed source:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_crossq_implementation.py \
  --output results/crossq-feasibility-2026-10-26/audit.json
```

## Before any training allocation

Fix the algorithm comparison, linear geometry and all optimizer/update/normalization
choices; bind the existing seven-field data and preprocessing without new features.
Preserve net-equity reward, action execution, costs and holding constraints. Define
shared initial linear tensors/RNG where comparable, explicit normalized parameters,
actual optimizer/BatchRenorm counters and replay/timeout restoration requirements.
Enable dedicated factory/frozen contracts and reject schema mismatches. Require
immutable normalization during frozen evaluation and actual learned Colab smokes.
Profile suitable Colab hardware before selecting it; do not infer speed from low
update-to-data ratio. Fix transitions, seeds11/23/37, selected/final group/holding
and drawdown gates plus a persistent study allowance before performance runs.
No full training protocol or extra allocation is authorized by this audit artifact.


## Recorded result

[Actual audit](../results/crossq-feasibility-2026-10-26/audit.json) passes from clean
sourceedf7018. Runtime Python3.13.16, exact native source hashes and all geometry/
optimizer/buffer records are retained. Two original and two restored untrained
models are constructed; zero environment transitions, optimizer steps or Colab
units. Original ledger checksum/39.47256169564856 units stays unchanged. The
performance protocol, factory/frozen integration and actual learned acceptance
remain next; no new return is claimed.


## Fixed SAC/CrossQ performance protocol

[Preregistration](../configs/experiments/crossq_comparison_preregistration.json)
and [actual data/reference binding checks](../results/crossq-feasibility-2026-10-26/preregistration.json)
fix two fresh50k seed11 arms, followed only on passing gates by seeds11/23/37
confirmation and prospective evaluation. Data672a216… is the same22-stock/four-ETF
train/development panel used by the historical SAC control, with seven fields and
original frozen preprocessing. This method comparison preserves the historical
control rather than adding the111-symbol panel or Treasury features simultaneously.

Both arms use64×64 linear widths, gamma0.99, learning rate0.0003,100k replay,
batch256,1000 warmup transitions and one critic update per collected transition.
SAC keeps tau0.005, actor every update and Adam beta1=.9; CrossQ has no target,
actor/entropy update every3critics, beta1=.5 and nine native normalization layers.
Entropy begins at1, target entropy auto−4; entropy optimizer native beta1=.9.
CrossQ batch-renorm momentum0.01/epsilon0.001 stay native; warmup is fixed1000
updates before any learning. This short-warmup/small-network variant is not a
replication of the paper's large networks or a normalization-only causal ablation.

Both4096-transition smokes retain full64 geometry, same renorm warmup, batch64/
learning-start64 and128-session episodes, evaluating every1024 steps. Expected
post-update counts: SAC4032 actor/critic/entropy; CrossQ4032 critics and1344
actor/entropy. Strict warmup comparison gives3031 critic/343 actor renorm forward
calls per layer. Actual Adam and normalization counters must corroborate this;
nominal counters alone are insufficient. On-step selected checkpoints exclude
the current transition's optimizer update. Full screens evaluate every10k with
512-session episodes and preserve sparse percentage-point-change curves.

Shared starts reuse only unfitted original full64 SAC archives for11/23/37,
also for smokes. Copy actor/head and ordered live-critic Linear tensors into
CrossQ; preserve its native normalization zero/one states and absent target.
Restore the original SAC post-initialization Torch/NumPy/Python streams in both.
Portable initializer generation/provenance acceptance precedes factory changes.
No learned SAC/TQC tensors are reused.

Actions, unrestricted train/evaluation exposure, costs,100k equity/whole shares/
cap1/20-session limit, net-log-equity reward, execution_full state and legacy
uniform training starts match historical SAC. Historical control metadata and
selected/final reports are hash-pinned; every checkpoint must reproduce within
0.01pp mean/0.05pp symbol. Never widen tolerances or automatically retry a mismatch.

Promotion requires selected **and** final mean gains≥2pp versus SAC, nonworse
final stock/ETF groups,≤2pp additional drawdown, both phases' stock/ETF and median
stock excess≥−2pp versus holding. Failure closes this fixed CrossQ family without
a warmup/size/optimizer/delay/batch/reward/feature/step rescue grid. Historical
mismatch disables interpretation and requires integrity investigation, rather than
being mislabeled an algorithm-performance failure. Retired2024+ test stays isolated.

A persistent2-unit study allowance shares the original150/130 project and150/170
account guards,7200s inclusive allocation deadlines and central minute metering.
CPU preference remains; actual Colab kernel and learned-smoke throughput must be
verified before a screen. Profile any proposed GPU benefit/cost explicitly.
Factory/frozen/runtime integration, shared starts and actual learned/replay/buffer
acceptance remain before any allocation. This registration performs no training.


## Shared Linear mapping implemented

`training/crossq_mapping.py` validates native untrained SAC/CrossQ types, seeded
21-input/four-action/full64 geometry, actual actor/critic counts, exact optimizer
ownership/defaults, empty state, initial alpha and source SAC target equality
before any target mutation. It copies ten ordered actor/head/twin-critic Linear
modules' weights and biases into existing CrossQ Parameters. All nine native
normalization layers retain zero means/bias/counters, unit variance/scale and the
fixed1000-update warmup. Mapping takes no random draws and preserves optimizer
parameter ownership. Nine guarded tests cover seeds11/23/37 and reject trained,
nonfinite, changed optimizer/delay/normalization and missing ownership before
mutation. Fixtures are untrained; actual pinned source-archive mapping, portable
initializer generation and runtime integration remain before allocation.


## Portable initialization generation

`scripts/build_crossq_initializations.py` requires a clean committed source and a
new directory under `configs/initializations`. For each11/23/37 seed it reloads the
original unfitted full64 SAC archive, maps common Linears into a fresh CrossQ
scaffold, restores original Torch/NumPy/Python streams and writes a CrossQ archive.
The SAC archive is copied byte-for-byte. Both phases later reuse these full64
assets. A manifest written last records source bindings, mappings, tensor/RNG
fingerprints and original generation-source snapshots. The generator refuses to
overwrite partial or completed directories; retain and inspect failed output.

All original SAC generation-source hashes are validated before snapshots are
written. Portable `.txt` copies preserve that historical provenance when the
live factory gains CrossQ support. Independent NumPy reads check all20 mapped
weight/bias tensors and complete zero/one native states across nine normalization
modules. The generator preserves caller RNG and never collects a transition,
reads a market dataset or steps an optimizer.25 focused mapping/archive/spec
checks pass. Actual committed-source generation is the next acceptance step.


## Actual pinned assets and portable validation accepted

[Initialization evidence](../results/crossq-feasibility-2026-10-26/initializations.json)
and [binding](../configs/experiments/crossq_initialization_binding.json) pin six
archives generated from clean source4e32ad7. Three original SAC archives remain
byte-exact; three CrossQ archives contain copied common Linears and complete
native normalization state. All source Torch/NumPy/Python streams are exact for
11/23/37. Independent NumPy reopening creates no models, changes no caller RNG
or ledger, and validates all six original-generation source snapshots.

`training/crossq_assets.py` validates the bound manifest/archives, portable paths,
all original source/sampling identities and snapshot hashes without constructing
a model. It reuses full64 entries in both phases while recording requested smoke
phase separately.37 focused checks include validation in a miniature asset tree
with a deliberately different live factory: historical `.txt` snapshots, not the
mutable registry, provide original provenance. No fitting/data/provider calls or
Colab units. Actual training dispatch, frozen normalization, saved learned state/
replay/counters and remote acceptance still remain before allocation.


## Restricted factory and frozen inference support

The factory now exposes only the fixed CrossQ parameter set and preserves full64
geometry in smoke mode. Existing SAC/TQC defaults, tensors and RNG are unchanged.
The saved action distribution is explicitly four-component squashed Gaussian.
CrossQ frozen bundles require actual geometry and all nine normalization state
fingerprints/counters; renamed or missing contracts reject even with valid new
bundle hashes. Loading validates the expected configuration and finite running
moments before prediction.

Frozen prediction captures a separate expectation from mutable display metadata,
requires evaluation mode, checks running state before each call and compares it
afterward. Any injected update restores original normalization state and raises
without returning an action. Caller observations are copied. Untrained/manual
counter fixtures cover exact save/load parity, immutable weights/statistics/input,
shape/nonfinite rejection, geometry/state tampering and post-prediction restoration.
The generic training entry point rejects CrossQ before dataset/environment/model
construction while the registered initializer, optimizer/replay/counter and runtime
contracts remain unfinished. This is inference integration, not learned acceptance
or permission to bypass the fixed Colab study.

86 targeted CrossQ/TQC/Treasury/recurrent regressions pass, including the final
display-metadata isolation check. No local learning, provider reads, allocation or unit use.
