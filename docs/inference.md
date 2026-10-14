# Frozen inference bundles

A bundle is a content-addressed directory under `artifacts/frozen/BUNDLE_HASH`. It contains selected model weights, canonical manifest, dependency lock when present (otherwise clearly recorded direct pins), and checksums for every file. The manifest includes feature/action schemas, ordered names, immutable preprocessing, exact source/split assumptions, resolved model/simulation settings, versions, seeds, selection history and model checksum.

The bundle identity hashes the manifest; the manifest hashes the weights. Loading validates the file set, each checksum, directory identity, expected feature/action schema and expected feed. Extra files, corrupt weights or mismatched feeds/features fail before prediction. Checksums provide integrity, not cryptographic publisher authentication.

## API

```python
from rl_stocks.inference.bundle import FrozenPolicy
policy = FrozenPolicy(bundle_path, feature_schema='daily-indicators-v1', feed='yahoo', device='cpu')
action = policy.predict(normalized_market_features_plus_portfolio_state)
```

`predict` requires exactly feature_count+5 finite float32-compatible values and requests deterministic SB3 prediction. The model policy is placed in evaluation mode. Loading never attaches a replay buffer; inference has no learning or normalizer update path.

`decide(features, names, portfolio_state, symbol, decision_time, expiry_time, reference, atr, equity, quantity, available_time, now=None, max_age_seconds=86400)` transforms raw market features with the frozen normalizer, appends the same five portfolio-state ratios as the environment, predicts and decodes a typed research decision.

It rejects an unknown symbol, mismatched name/order, malformed observation, nonpositive equity/reference/ATR, unavailable features, future decision time, expired signal or source age beyond the configured maximum. Required ordering is availability <= decision <= current < expiry. UTC timestamps are mandatory. Set an explicit maximum age consistent with your session/holiday scheduling; the default 24 hours is conservative and may reject stale weekend data.

Returned fields include immutable bundle ID plus action schema, decision/expiry/reference, raw and clipped action, desired exposure/whole-share quantity, notional, direction/side, stop and ordered target zone, and reconciliation intent. These are research proposals. A future broker adapter must separately reconcile actual holdings and enforce account/feed/quantity/shortability/order restrictions.

## Parity and limitations

Training compares selected pre-export deterministic action with the reloaded frozen action on the same observation using exact array equality. Integration tests also check repeated predictions, normalizer immutability, schema/feed rejection and checksum corruption. These checks pass with the reviewed Python 3.13 lock. All five downloaded A100-session bundles also passed local CPU save/load parity, repeated predictions and unchanged normalization/file checks. This does not claim bit-for-bit CPU/CUDA parity. See [verification](verification.md) for artifacts.

All current candidates, including the initial 50,000-transition PPO model, have `paper_eligible=false`. No inference call places an order, updates replay or retrains. Shared batch feature functions support causal historical recomputation; a durable live-data scheduler, incremental feature state, drift monitoring and restart reconciliation remain the shadow-inference milestone. Alpaca paper execution remains later still.
