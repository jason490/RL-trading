# CrossQ implementation feasibility

This is an untrained source/serialization audit, preceding any performance
protocol. CrossQ is not registered in the training factory or frozen loader yet.
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
