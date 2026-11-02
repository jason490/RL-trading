"""Inspect the actual default kernel before installing any runtime dependencies."""

import json
from pathlib import Path
import sys

record = {
    "python": list(sys.version_info[:3]),
    "executable": sys.executable,
    "supported": sys.version_info[:2] == (3, 13),
}
Path("/content/task-reference-kernel.json").write_text(json.dumps(record, sort_keys=True))
print(json.dumps({"python": record["python"], "supported": record["supported"]}))
