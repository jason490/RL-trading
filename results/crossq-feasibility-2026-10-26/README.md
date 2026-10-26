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


## Subsequent fixed performance registration

The [separate protocol](../../configs/experiments/crossq_comparison_preregistration.json)
now fixes a50k two-arm screen with4096-step learned smokes and a persistent2-unit
allowance. [Binding checks](preregistration.json) validate original26-symbol data,
preprocessing and historical SAC metadata/report hashes without model construction,
optimization or allocation. CrossQ retains delay3/beta1=.5 versus SAC1/.9;
renorm warmup1000 is fixed before learning and exercised by both smoke actor/
critic paths. Same64 widths, original execution/reward and exact shared native
SAC linear starts are required; this is an algorithm-package comparison. Nine
tests check phase-sensitive expected updates, immutable parameters and strict
absolute/relative/reproduction gates. Learned models/counters/replay and runtime
integration are still pending. No new model return or credit use is claimed.


## Actual shared initialization assets accepted

[Independent initialization proof](initializations.json) pins six untrained
archives and six portable original-generation source snapshots from clean
source4e32ad7. SAC bytes are original; all20 common weight/bias tensors and all
Torch/NumPy/Python streams match for11/23/37. Native CrossQ scale/variance are1,
means/bias/counters0. Full64 assets are reused by smokes. Independent NumPy reads
construct no models and leave RNG/ledger unchanged.37 focused checks validate
archives, path/phase rejection, unchanged ownership and provenance after a fake
live-factory change. No new return, local optimization or Colab allocation.
Next: factory/frozen/runtime and actual learned optimizer/replay/normalization
acceptance, preserving this fixed protocol and existing usage ledger.


## Restricted inference integration

CrossQ factory/frozen support now checks the fixed full64 native geometry,
squashed action contract and explicit saved normalization. Deterministic untrained
save/load parity and manual running-state fixtures pass; injected normalization
updates are restored and rejected. Original SAC/TQC default weights/RNG regressions
pass. Generic training is blocked before data/model construction pending dedicated
shared-start and learned optimizer/replay/counter/runtime acceptance. No CrossQ
trading return or further credit use is claimed.


## Initializer and state contract implementation

Pinned initialization now restores original tensors/global streams and validates
all settings/ownership before mutation. Actual Adam moment/step and normalization
records enforce checkpoint phase; final replay hashes preserve filled rows and
correct timeout/terminal semantics.108 focused regressions pass. Native optimizer
save/load fixtures are manually populated, with optimization forbidden; these
are not learned acceptance. Generic training remains blocked pending observed
forward evidence, registered runner and Colab runtime/archive integration. No
new return or Colab charge is claimed.
