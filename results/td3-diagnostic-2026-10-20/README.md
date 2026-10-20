# Saved baseline TD3 audit

Three owned 50k TD3 seeds (11/23/37), six saved selected/final models and six
2021–2023 development reports were inspected without fitting, predictions,
dataset access or Colab allocation. Only saved owned model bundles are loaded.
All source hashes remain unchanged. Retired datasets are never loaded merely
because their identities occur in old metadata.

All three models use 12 legacy inputs (seven market features plus five portfolio
fields), omitting nine active execution fields. Restored objects confirm fixed
four-component NormalActionNoise with mean 0, sigma 0.1, policy delay 2, target
noise 0.2/clipping 0.5 and 49,000 critic updates. Missing noise/observation fields
in old metadata do not establish absent exploration or a 21-input model.

| Three-seed mean | Stocks | ETFs |
| --- | ---: | ---: |
| Selected net return | 4.84% | 0.57% |
| Final net return | -0.08% | -0.37% |
| Matched holding | 32.63% | 19.47% |
| Final incremental cost drag | 10.60pp | 8.85pp |
| Final saved-path non-cost excess | -22.11pp | -10.98pp |

These are cumulative returns across separate symbol episodes, not annualized
or portfolio returns. Final end-of-session short fractions average 18.62% stocks
and 16.85% ETFs. Those fractions do not establish positive net short P&L; old
reports lack directional attribution and this audit deliberately does not open
corporate-action bars to reconstruct it. Cost add-back is accounting, not a
zero-cost replay. Old source/data/GPU contracts are not new matched controls.

A fixed legacy/masked/full-state TD3 study is preregistered to test the verified
input omission under delayed deterministic learning, retaining negative prior
PPO/SAC information. Missing state may matter, but no improvement is promised.
No longer training, extra data, temperature tuning or noise sweep is combined.
[Protocol](../../configs/experiments/td3_execution_state.json),
[guide](../../docs/td3-execution-state.md), [full audit](summary.json).

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/diagnose_td3.py \
  --output artifacts/td3-diagnostic/new-summary.json
```

Twelve locked-protocol/off-policy accounting checks pass. The protocol is fixed;
runtime integration, initial-network/noise verification, smoke and screen are
not complete and no new session has been allocated. Cumulative project remains
36.069198/150. The overall comparable-return/profitable-short objective is unmet.
