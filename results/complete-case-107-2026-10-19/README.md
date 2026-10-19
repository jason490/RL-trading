# Explicit complete-case stock dataset — 2026-10-19

Prepared a verified Yahoo-only research dataset with **107 stocks plus four
broad ETFs** for training and the original **22-stock/four-ETF validation panel**.
The eleven sector ETFs remain audited context symbols. The original 110-stock
request is preserved separately; EA/AVB/EQR are explicit exclusions because
Yahoo returned no history. Their partial Alpaca histories remain unmerged.

| Item | Verified result |
| --- | --- |
| Dataset | `c2e6c58c4d4b0a9853b5b37851c731f559f42e597e9eefcf77796d874db31398` |
| Immutable parent snapshot | `09bc69ec15798c6bf2987c3332397104ce3066846bad638b1512e29e41f1028f` |
| Training symbols / feature rows | 111 / 298,946 |
| Validation symbols / feature rows | 26 / 19,058 |
| Test rows | 0 |
| Validation match to original control | Exact bars, features, ATR and episode keys |
| Unverified-data override | False |

The new explicit `coverage_scope='universe'` checks every configured tradable
and context symbol. It retains full original source provenance and records
all excluded source failures. Default snapshot-wide quality checks are unchanged.
Missing context coverage/bars, incomplete retained symbols and unverified
dividends still fail. Existing successful default dataset identities are unchanged.

The seven basic features, chronological boundaries and 20-session purge are
unchanged. Preprocessing is fit on the new training panel only and frozen.
Existing models trained on the smaller panel still require **their own saved
normalizer**; this expanded normalizer must not be substituted during inference.
`dataset.json` records identities, counts, the coverage audit and validation hashes.

Reproduce without network or model training:

```python
from rl_stocks.data.generation.datasets import generate
from rl_stocks.data.universe.config import read_universe, symbols

generate(
    "data/raw/universe-expansion/09bc69ec15798c6bf2987c3332397104ce3066846bad638b1512e29e41f1028f",
    "data/datasets",
    universe=read_universe("configs/universes/stock_research_107_complete.json"),
    evaluation_symbols=symbols(read_universe("configs/universes/sector_research.json"), "tradable"),
    include_test=False,
    coverage_scope="universe",
)
```

This preparation does not train another model or change the dataset used by the
ongoing action-initialization screen. Sector labels are current annotations;
the sample remains retrospective survivor research. Historical membership,
delisting outcomes and representative market performance remain separate gates.
No new provider calls, interpreter changes or separate Colab allocation were
needed. Full local acceptance passes 193 tests, with 13 existing warnings and
no skips/failures; scope/context failure checks also pass independently.

## Training-only declining-regime coverage

The existing trailing-price audit uses a 252-session price return below -20%
and a negative 63-session price return. It inspects training episodes only;
validation changes cannot affect these diagnostics. It does not label future
recovery or include dividends in its trend classification.

| Measure | Original 26 tradables | Expanded 111 tradables |
| --- | ---: | ---: |
| Eligible trailing observations | 63,648 | 270,974 |
| Observations with trailing yearly decline below -20% | 1,851 (2.91%) | 18,829 (6.95%) |
| Also negative over the trailing quarter | 1,468 (2.31%) | 13,964 (5.15%) |
| Symbols with at least one such persistent-decline window | 16 | 93 |

The broader training set contains substantially more observed downtrend periods.
That supports a future matched experiment about learning downside behavior; it
does not prove additional stocks improve a policy or establish short profits.
`control-regimes.json` and `expanded-regimes.json` retain exact settings, symbol
counts and dates; their CSVs are compact per-symbol diagnostics. The original
22-stock/four-ETF validation panel remains fixed for any matched comparison.
