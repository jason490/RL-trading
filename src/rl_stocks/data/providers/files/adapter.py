import csv
from dataclasses import fields, replace
from pathlib import Path
from rl_stocks.core.contracts import Bar, Capabilities, Request
from rl_stocks.data.validation.bars import validate

class FileProvider:
    capabilities = Capabilities("files", limitations="Canonical schema required; provenance preserved")
    def __init__(self, path):
        self.path = Path(path)
    def bars(self, request: Request):
        if self.path.suffix == ".parquet":
            import pandas as pd
            rows = pd.read_parquet(self.path).to_dict("records")
        else:
            with self.path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        numeric = {"open", "high", "low", "close", "volume", "split", "dividend"}
        names = {f.name for f in fields(Bar)}
        result = [Bar(**{k: float(v) if k in numeric else v for k, v in row.items() if k in names})
                  for row in rows if row["symbol"] in request.symbols and
                  str(request.start) <= row["session"] < str(request.end)]
        validate(result)
        self.capabilities=replace(self.capabilities,feed=result[0].feed,adjustments=result[0].adjustments)
        return result
