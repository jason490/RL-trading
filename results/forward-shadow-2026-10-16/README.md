# Frozen forward shadow campaign

The user authorized persistent shadow observation of the saved promising model.
The [recipe](../../configs/shadow/forward_validation.json) fixes nine selected
bundles: combined SAC seeds 11/23/37 (including the historical 67.24% validation
checkpoint), basic PPO seeds 11/23/37, and basic SAC seeds 11/23/37.

The campaign uses a fixed 126-settled-session research horizon, unchanged
simulation costs and a buy-and-hold/cash comparison. Each model/stock has an
independent $100,000 simulated account. It has no broker-order path and cannot
automatically become paper-eligible. All policies and seeds remain reported;
forward data must not select a replacement checkpoint.

[Software acceptance](software-acceptance.json) checks the actual nine bundles
against previously inspected validation inputs: features/ATR match the stored
research data and observations match the training environment for all 26 symbols.
Normalization is unchanged and frozen actions are finite. These checks do not
produce fresh performance claims. Full local acceptance passed 156 tests.

Actual registration and process status are recorded separately. The first
eligible signal must use a bar closing after registration; older bars are warmup.
There are no forward returns at initialization. The retired 2024–September 2026
interval is not reopened as an independent audit. Historical membership and
delisting coverage remain unverified.

See [operation, reports and recovery](../../docs/shadow-inference.md). Runtime
journals, raw data and process logs remain private/ignored. A persistent host is
required for daily uptime; installing the provided service template is an
operator deployment step, not something the template performs automatically.
