# Compact broad-market context

The current recurrent source has seven inputs from the traded symbol only.
The failed sector study supplied a large proxy/rank feature set, with optional
issuer classifications. This proposed family instead has three fixed price-only
fields: SPY20-session log return, SPY20-session realized volatility, and the
traded symbol's20-session log return minus SPY's. No sector label, return-ranked
universe or future declining-stock label is needed. Benefit is unproven.

## Coverage and causal preflight

[Fixed audit protocol](../configs/experiments/market_context_audit.json) pins the
original111/26 source and cached Yahoo snapshot. Both raw windows must contain
the exact same21 sessions and no known missing session; all contributing bars
must be available by the source decision bar's recorded availability. Never carry
stale context, backfill it, or use the last historical window for earlier rows.
Use forward-only split continuity and price returns; dividend accounting stays
in the simulator. A delayed or missing benchmark row makes the context unavailable.

Audit every original train/development row and prove its source bar identity.
No training, inference, normalization fit, validation-performance scoring,
provider request or Colab allocation. The snapshot ends before2024; the retired
test remains unopened. Source coverage acceptance does not prove predictive power.
Revised Yahoo prices, assumed availability and survivor-universe limitations remain.

Only a passing audit permits a separate dataset/training preregistration, with
training-only normalization for additional fields, preserved original seven-field
statistics and matched full/masked/legacy controls. Existing failed session,
clipping, capacity, L2 and squashed-action families remain closed.

## Implemented feature and audit

`features/market/context.py` groups the available series once, applies each split
only when effective and precomputes each completed20-session window. It uses log
price differences for numerically stable log ratios. Only SPY needs volatility;
other symbols need their own momentum and matched availability/session windows.
`at` returns three fields or unavailable, never a stale substituted value. Explicit
missing-session metadata is required for every input series. Generic bar validation
rejects unsorted/duplicate/mixed-feed or invalid inputs.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
  scripts/audit_market_context.py --output artifacts/market-context/coverage.json
```

The fixed audit verifies original/raw/protocol/implementation SHA-256 identities,
source calendars and every original row. Actual results:298,946/298,946 training
rows and19,058/19,058 development rows available, with no exclusions or source
changes.60 focused checks pass. [Evidence](../results/market-context-2026-10-23/README.md).
No new preprocessing fit, dataset/model or performance result was produced.
A matched feature study remains to be specified before any Colab allocation.
