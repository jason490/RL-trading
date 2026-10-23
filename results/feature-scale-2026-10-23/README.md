# Frozen market-feature scale audit

Completed outcome-free distribution audit on cached full-history111 training/
original26 development inputs. Original normalizer unchanged, no clipping/refit,
model/provider access, fitting, Colab allocation or retired-test access.
[Summary and source hashes](summary.json), [compact distributions](distributions.csv),
[category method](../../docs/feature-preprocessing.md).

| Feature | Train107-stock max abs z | Train fraction abs z>5 | Development22-stock max abs z |
| --- | ---: | ---: | ---: |
| One-day log return |50.55|0.604%|9.92|
| Volatility20 |26.66|1.114%|4.39|
| EMA20 ratio |25.21|0.580%|6.72|
| EMA50 ratio |17.95|0.599%|6.05|
| RSI14 |3.90|0%|3.47|
| ATR14 ratio |51.13|1.197%|4.94|
| Relative volume20 |80.73|0.574%|22.50|

Training rows:288,146 stock and10,800 ETF; original stock subset59,400 rows,
additional85 stocks228,746. Development16,126 stock/2,932 ETF rows. These are
available-row distributions before float32 cast, not realized training-year
visitation or uniform-symbol weights. All four broad ETFs reported separately.
Original normalization comes from the original training panel and stays frozen
in expanded-source comparisons; extra-stock mean/scale shifts are not refitted.

Extreme source examples include APA2020-03-09 log return, APA2020-03-17 ATR ratio,
and EQIX2010-10-06 relative volume. All nine top examples recompute from source
arithmetic. Relative volume excludes today's observation from its prior20 mean,
so37.83× prior volume is possible and not itself a formula error. This is source
consistency, not independent-vendor verification or proof of real economic causes.
No stocks/dates removed and no favorable future outcomes used to form groups.

The audit supports investigating an explicit bounded input transformation, not a
claim that clipping will improve trading. Any single fixed-cap comparison needs
its own frozen protocol, immutable schema/normalizer/bundle handling, identical
models/costs/steps/return/risk gates and failure stop. Raw prices/ATR execution/
corporate actions/rewards/baselines must remain untouched. Existing failed families
remain closed; no threshold sweep or normalizer fit on validation/test.

Reproduce with a new output path:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
 scripts/audit_feature_scale.py --output .runtime/feature-scale-reproduced.json
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_feature_scale.py
```

Seven checks pass; actual source and normalizer hashes unchanged, all nine extreme
arithmetic witnesses match. No new Colab units: project38.74617914/150, last
verified account349.23 zero rate/assignments. All prior owned sessions stopped.
