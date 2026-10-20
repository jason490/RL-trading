# Saved SAC execution-state and participation diagnosis

Exactly three frozen100k SAC models (seeds11/23/37) and six selected/final
reports from the completed algorithm-stability study are inspected. All model,
metadata and report hashes remain unchanged. No dataset is opened, no model is
fitted, and no Colab allocation or broker action occurs.

All models load with12 inputs: seven market indicators and five legacy portfolio
fields. They omit held stop/target distances, protection-present/partial-exit
state, previous protective actions and share-unit scale (nine extra fields in
`execution_full`). This verifies an observability gap, not its performance effect.
Earlier full-state PPO experiments do not establish the effect for SAC.

| Group, 3-seed mean | Selected net return | Final net return | Holding |
| --- | ---: | ---: | ---: |
| Stocks | 22.07% | 17.69% | 32.63% |
| ETFs | 6.69% | 9.97% | 19.47% |

Final stock shortfall14.94pp splits into5.98pp incremental cost drag and8.96pp
saved-path non-cost gap. ETF shortfall9.50pp splits into3.91pp costs and5.59pp
non-cost gap. This is an accounting identity with both policy and holding costs
added back, not a cost-free replay or causal attribution.

Final positions are long at90.10% of stock closes and92.74% of ETF closes;
short at2.53%/0.25%. Mean absolute exposure is83.12%/87.25%. Those close fractions
miss intraday positions and cannot prove profitable shorts. These old reports
lack corporate-action bars needed for exact direction P&L reconstruction; the
diagnostic deliberately does not fabricate it or reopen the retired dataset.
New matched training must retain the current complete short-side accounting.

[Summary and frozen-input proof](summary.json), [group table](groups.csv),
[original study](../algorithm-stability-2026-10-16/README.md).
Ten focused saved-path/participation tests pass, including unmatched horizons,
invalid costs/equity, retired-date and nonfinite rejection. Actual three-model
restoration and six report audits pass. No original files changed; project usage
remains35.917535/150. Repeated2021–2023 data is development-only evidence.

Next preregister a fixed SAC legacy/masked/full execution-state comparison using
the existing train/validation-only dataset. Match information/capacity controls,
steps, costs, preprocessing and seeds; require both selected/final stock/ETF
benchmarks before confirmation. This diagnosis alone promotes no model.
