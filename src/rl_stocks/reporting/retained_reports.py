"""Load the single retained selected/final report pair for an owned task."""

import json


def run_reports(root):
    paths = list((root / "runs").glob("*/metadata.json"))
    if len(paths) != 1:
        raise ValueError("one retained run per owned task required")
    run = paths[0].parent
    return (
        run,
        json.loads(paths[0].read_text()),
        {
            phase: json.loads((run / name).read_text())
            for phase, name in (
                ("selected", "selected-validation.json"),
                ("final", "validation.json"),
            )
        },
    )

