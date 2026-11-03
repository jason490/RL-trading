# Routed SAC reference comparison — 2026-11-03

The complete fixed seed11 comparison failed its original final training gate.
No development scoring, confirmation seeds, model promotion or retired-test access
occurred. The family is closed; the broader improvement objective remains unmet.

All returns below are mean net total returns across26 separate single-symbol
episodes on the fixed2018–2020 training window. They are neither annualized nor
a portfolio return, out-of-sample evidence or a trading-readiness result.

| Reference snapshot | Overall | Stocks22 | ETFs4 | Mean maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Routed selected90k |61.62%|64.24%|47.26%|−28.62%|
| Routed final100k |54.14%|56.05%|43.63%|−30.42%|
| Tagged shared final100k |46.03%|50.37%|22.18%|−35.37%|
| Masked shared final100k |27.61%|34.89%|−12.47%|−36.06%|
| Matched holding |56.45%|58.15%|47.15%|—|

The routed final snapshot beats both controls by26.53/8.11 percentage points,
passing the required2pp gains. It fails all three absolute excess floors:
stock−2.10pp, ETF−3.52pp and median stock−6.16pp, versus the fixed−2pp limit.
The stronger90k checkpoint does not override the predefined final admission gate.
Native training-curve selection remains90k; reference inference did not reselect.
Both routed phases achieve positive total return plus positive short P&L for0/2
reporting-only declining stocks. Future decliner labels did not enter learning,
selection or gates. No additional steps or tuning rescue is authorized for this family.

## Acceptance evidence

[Complete acceptance](reference-candidate-acceptance.json) binds the original
transport SHA, captured clean training revision368e4f3, actual Python3.13.18/CPU,
44 routed initial arrays mapped exactly to32 retained shared arrays, all11
checkpoints, saved optimizer/replay/selection records, and selected/final frozen
bundles. Training completed100,000 transitions and99,000 actor/critic/entropy
updates. Both reference points independently replay26 episodes/19,136 decisions;
lossless path hashes, group metrics, short attribution and unchanged gate were
rechecked after acceptance. Post-stop physical/reference acceptance took168.72s.
Original native arithmetic failures/claims remain unchanged.

The original controller completed transport and verified shutdown. Its process
handle was unavailable after context recovery and no saved offline acceptance
existed; the offline-only acceptance entry point then ran against the exact stopped
state and retained artifacts. No allocation or learning was repeated.

Interactive reports: `artifacts/task-reference-candidate-acceptance-v1/selected.html`
and `final.html`; complete native learning curves remain in the retrieved
`artifacts/colab/rl-stocks-cpu-07e11d409cec/retrieved/artifacts/task-screen/curve.html`.
Raw model/replay/source archives and differing snapshots remain outside Git.

## Credits and cleanup

[Usage](usage.json): measured0.060000 units, conservative estimate/charge
0.07024609. Project40.19698237/150, remaining109.80301763; original study
0.27112317/2, with unchanged baseline. Fresh account548.05, rate0, assignments0.
The owned session was terminated and verified; no recovery command is needed.
[Retention audit](retention.json) records protected evidence and confirms no
partial downloads remain. Original upload/transport and differing snapshots are
retained for provenance; generated reports use compressed lossless paths.

Next: diagnose the fitted endpoint decline and long/short participation using
these saved paths before specifying a distinct hypothesis. Do not reopen this
family, tune its gates, or infer prospective performance from the selected result.


## Saved-path endpoint diagnosis

[Read-only diagnosis](saved-path-diagnosis.json) covers all six accepted reference
points,156 episodes/114,816 saved decisions. Existing participation/cost accounting
reconciles fills, borrow, long/short P&L and net equity. Input/model/ledger evidence
is unchanged; no model was loaded, no dataset opened and no Colab allocated.

From routed90k to100k, mean stock return declines8.19pp. Incremental cost drag
rises1.52pp, while saved-path non-cost excess declines6.66pp. ETF return falls
3.63pp despite cost drag improving4.31pp; saved-path non-cost excess falls7.94pp.
These are exact fixed-path identities, not a zero-cost replay or a causal
explanation of training deterioration. Costs added back would not preserve fills.

Final stock long/short contributions are+55.24/+0.81pp; ETF contributions are
+51.73/−8.10pp. Selected ETF short contribution was+0.55pp. Final ETF reversal
costs average7.26pp of initial equity; small same-side resizing costs only0.10pp.
Stock small-resize costs are0.05pp. Thus a small-resize filter alone does not
account for the observed gap. End-of-session long fractions are80.01% stocks and
79.11% ETFs; these differ from intraday touched fractions in the receipt.

The next distinct hypothesis needs training-only support for better direction/exit
quality and short participation, plus matched controls, cost profiling and the
existing multi-seed/prospective gates. These observations do not authorize a
short ban, reward grid, extended steps or reselection within the closed family.
