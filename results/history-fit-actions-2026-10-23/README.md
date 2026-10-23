# Saved training/development action comparison

Eight accepted paths (full/recent×selected/final×training/development),208 symbol
episodes, compared without fitting or dataset/provider access. Training report
hashes match the fixed replay; development hashes and model files match the
accepted study. Original26 panel and train/validation labels verified; source
files unchanged. [Summary](summary.json), [group CSV](groups.csv).

## Final recent model: same weights, different market periods

| Stock metric | Eligible2018–2020 training window | Reused2021–2023 development |
| --- | ---: | ---: |
| Mean net return |46.66%|4.46%|
| Long request fraction |76.43%|76.11%|
| Mean closing short exposure |19.36%|19.56%|
| Net short return contribution |−4.07pp|−10.54pp|
| Incremental cost drag vs holding |8.95pp|7.26pp|
| Residual noncost excess vs holding |−2.53pp|−20.90pp|

Similar aggregate participation does not imply similar timing or market payoffs.
Cost drag is higher in the training window despite better returns. The direction
frequency/cost totals alone cannot identify a cause or predict a remedy. Net
short contributions already include allocated costs; never subtract costs twice.
Different dates/regimes/durations prohibit causal raw-return comparisons. No
bar-level visitation claim, checkpoint selection, promotion, short ban or future
stock ranking. The failed history family remains closed.

The next step is to inspect saved optimizer/parameter behavior before defining a
distinct learning hypothesis. This report is not evidence that a particular
regularizer, model, input or longer run will improve out-of-period returns.

## Reproduce and scope

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python  scripts/compare_history_fit_actions.py --output .runtime/history-fit-actions.json
```

Existing diagnostics remain development-only by default. The new explicit
`period='training-window'` scope accepts only train-labelled2018-01-02 through
2020-12-02 rows; test/future periods reject. All existing equity, cost, quantity
and holding-path reconciliation remains active.29 focused action/audit/
participation tests pass, including opt-in training and unchanged default/future
rejection. This changes reporting scope only, not the simulator or training.

Project38.68329334/150 unchanged, last official account349.27 zero rate/assignments;
all prior owned sessions remain stopped. Model weights and original reports are
retained in the published history releases; no new Colab allocation or orders.
