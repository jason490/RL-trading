# Research model releases

Large research models are published as GitHub release assets. Source, reviewed
experiment protocols and compact results remain in Git. Original provider bars,
replay buffers and private runtime logs remain local.

## Initial four-ETF baseline

[research-baseline-2026-10-14](https://github.com/jason490/RL-trading/releases/tag/research-baseline-2026-10-14)
contains five completed CPU/CUDA models from the original A100 session, including
PPO 50,000 transitions, seed 11, on SPY/QQQ/IWM/DIA. Their training source revision
is recorded in metadata; the release tag identifies the later packaging tools.

| Asset | Contents |
| --- | --- |
| `research-models.tar.gz` | Exact frozen bundles, selected/final weights, run metadata and checkpoint histories |
| `validation.html` | Derived selected-model equity versus buy-and-hold and metrics; original PPO chart is a CPU replay |
| `model-catalog.json` | All five completed model records |
| `SHA256SUMS` | Checksums for the three assets above |

All four server-reported release asset digests were checked against local
SHA-256 values after upload. Extract models into a separate directory and retain
each frozen bundle's complete content-addressed directory; adding/removing files
inside a bundle fails inference integrity checks. Use [FrozenPolicy](inference.md)
with the declared schema/feed and reviewed dependencies.

These are research candidates with `paper_eligible=false`. Validation selected
the checkpoints. This release contains no held-out test evaluation and no
expanded-stock, supply/demand or sector-rotation trained model. Subsequent study
releases will identify those datasets explicitly; see [experiments](experiments.md)
and [verification](verification.md).
