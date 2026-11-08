# Stock and index context expansion — 2026-11-08

Downloaded all22 additional stock histories:71,909 daily bars, bounded to
2010 through2023. The new configured universe contains132 stocks,4 broad ETFs
and11 sector proxies. Existing datasets/models/experiments remain unchanged.

Two additional names per declared current sector:

| Sector | Stocks |
| --- | --- |
| Communication services | CHTR, FOX |
| Consumer discretionary | BBY, DHI |
| Consumer staples | SYY, ADM |
| Energy | PSX, MPC |
| Financials | SCHW, TROW |
| Health care | CVS, CI |
| Industrials | CSX, EMR |
| Materials | DOW, IP |
| Real estate | DLR, ESS |
| Technology | AMD, AMAT |
| Utilities | ETR, CMS |

No XNYS sessions are missing inside any stock's observed range. CHTR begins
2010-01-05, MPC2011-06-24, PSX2012-04-12, FOX2019-03-13 and DOW2019-03-20;
short histories are retained explicitly. No prelisting prices are manufactured.
Current labels/selection do not prove historical sector membership or remove
survivorship bias. Price coverage alone does not admit a training panel.

MOVE:3,693 observations from2009-01-02 to2023-12-29, with79 absent XNYS dates
in2010–2023. VIX:3,774 retained observations with zero absent XNYS dates over
that interval. VIX checksum matches its original retained artifact. Both are
context-only; availability/lag-aware coverage remains separate from raw-date
coverage. Model inputs are disabled.

[Acquisition receipt](acquisition.json) binds source paths, checksums, exact
missing dates and counts. Local content-addressed raw histories and original
acquisition receipt remain retained; large source data is excluded from Git.
No failed acquisition, data fallback, dependency installation, training, Colab
allocation or retired-test access occurred. No extra staging copies were created.
The unclaimed collector requires a fresh output and writes a durable claim; do
not rerun the completed November8 output. API credentials were unnecessary.

The optional relationship component computes20-session correlations with SPY
and a supplied sector proxy plus sector/relative momentum. It remains detached
from training until historical mapping, coverage and experiment admission.
