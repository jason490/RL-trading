# Compact market-context coverage audit

Status: fixed cached-data coverage/availability preflight passed. No new dataset,
normalization fit, policy inference, model training, validation-performance score
or Colab allocation. [Protocol](../../configs/experiments/market_context_audit.json),
[method](../../docs/market-context.md), [complete evidence](coverage.json).

| Existing split | Rows checked | Context available | Unavailable |
| --- | ---: | ---: | ---: |
| Training,111 symbols |298,946|298,946|0|
| Development,26 symbols |19,058|19,058|0|

Each original source row matches its cached raw bar exactly. All contributing
own/SPY windows contain the same21 sessions, avoid known gaps, and are available
by that decision bar's recorded availability. Covered source calendars are
verified with no missing sessions. The immutable raw snapshot ends before2024;
retired-test bars were not opened. Source/raw/protocol/implementation hashes are
recorded and unchanged. No row, date or stock was selected by future returns.

Three fixed candidate fields: SPY20-session price log return, its20-return
population volatility, and own-minus-SPY20-session price log return. Forward-only
split continuity; dividends excluded from these price features and retained in
simulation cash accounting. The test does not establish economic alpha or vendor
vintage correctness. Revised Yahoo/survivorship/assumed-availability limitations
remain explicit. All existing seven-field preprocessing and source payloads stay
unchanged; the audit fits nothing.

60 focused feature/core/audit checks pass, including causal prefix/future-suffix
parity, warmup, delayed publication, session mismatch, known-gap rejection and
split/dividend conventions. No new Colab units: original project38.90354117,
last official account349.10 zero rate/assignments; previous sessions stopped.

Next: separately preregister a bounded full/masked/legacy-control study, build
immutable opt-in payloads with training-only extra-field normalization and
preserved original seven statistics, validate matched rows/initializations and
frozen inference, then run Colab smokes before any research training. Coverage
acceptance alone does not select a model or authorize automatic allocation.
Closed failed feature/optimizer/distribution families remain closed. Robust-return
objective remains unachieved.
