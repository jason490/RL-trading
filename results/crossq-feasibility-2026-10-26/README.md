# CrossQ untrained implementation audit — passed

[Actual audit](audit.json) uses clean sourceedf7018, Python3.13.16, SB3/Contrib2.7.0
and existing Torch2.8.0. No learning, environment transitions, optimizer steps,
data/provider reads or Colab allocation. Original ledger remains39.47256169564856.

Synthetic21-input/four-action64×64 models confirm:

| Actual geometry | SAC | CrossQ |
| --- | ---: | ---: |
| Actor | 6,088 | 6,386 |
| Live critics | 11,778 | 12,390 |
| Separate target critic | 11,778 | 0 |
| BatchRenorm modules | 0 | 9 |

Matching widths adds910 native scale/bias parameters to CrossQ; do not claim exact
trainable-capacity equality. Effective native actor beta1 differs0.9/0.5; CrossQ
policy delay3 and normalization warmup100,000 updates require explicit treatment
before performance comparisons. All nine evaluation counters remain0; actual
weights/buffers are unchanged by five deterministic synthetic predictions. Native
save/load reproduces predictions, complete tensor/buffer fingerprints and empty
optimizer/replay state exactly. Caller RNG preservation and rejected version drift
pass guarded tests; learner learn/train and Adam.step calls are forbidden.

Two original and two restored untrained models are constructed; temporary model/
replay files are removed. This is not learned optimizer acceptance, real-data
performance, or hardware-throughput evidence. No CrossQ factory/frozen dispatch
or performance protocol is enabled. Next: preregister a fixed matched SAC/CrossQ
comparison, including explicit update/normalization choices, original seven-field
data/actions/reward, actual shared-start and replay/buffer acceptance, bounded
Colab profiling/smokes and unchanged selected/final multi-seed/prospective gates.
Treasury/VIX/GAE and prior failed families remain closed; retired test stays isolated.
See [category guide](../../docs/crossq-feasibility.md) for source citations and limits.
