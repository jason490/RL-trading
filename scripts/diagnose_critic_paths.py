"""Bounded paired accepted-report audit; no models or provider requests."""

import argparse
import json
import math
from pathlib import Path
import signal
from statistics import mean
import time

from rl_stocks.core.contracts import digest
from rl_stocks.core.files import write_once
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.action_execution import diagnose_episode
from rl_stocks.reporting.path_divergence import compare_paths

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "9b13dd7c2ac791c3052052fb47bf0f407f2afbe12173689448c12f39c1f7b5c4"


def journal(value):
    path = ROOT / ".runtime/critic-path-progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def audit(output):
    started = time.monotonic()
    config_path = ROOT / "configs/experiments/critic_path_diagnostic.json"
    spec = json.loads(config_path.read_text())
    if digest(spec) != PROTOCOL_SHA256 or output.exists():
        raise ValueError("fixed protocol and new output required")
    summary_path, state_path = ROOT / spec["source_summary"], ROOT / spec["state"]
    if checksum(summary_path) != spec["source_summary_sha256"]:
        raise ValueError("fixed accepted critic result required")
    summary, state = json.loads(summary_path.read_text()), json.loads(state_path.read_text())
    if (
        summary["screen"]["passed"] is not False
        or summary["test_evaluated"] is not False
        or state["status"] != "complete"
        or state.get("cleanup_verified") is not True
    ):
        raise ValueError("failed fixed study and verified owned shutdown required")
    cell = summary["cells"][spec["arm"]]
    root = Path(state["completed"]["critic-" + spec["arm"] + "-11"]["root"])
    run = root / "runs" / cell["run"]
    metadata_path = run / "metadata.json"
    if (
        checksum(metadata_path)
        != cell["restoration"]["protected_source_sha256"][str(metadata_path)]
    ):
        raise ValueError("accepted metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata["selected_step"] != 50000
        or metadata["critic_selected"]["updates"] != 48999
        or metadata["critic_final"]["updates"] != 49000
    ):
        raise ValueError("exact accepted one-update endpoints required")
    files = [
        config_path,
        summary_path,
        state_path,
        metadata_path,
        Path(__file__),
        ROOT / "src/rl_stocks/reporting/path_divergence.py",
    ]
    reports = {}
    for phase, filename in (("selected", spec["before"]), ("final", spec["after"])):
        path = run / filename
        if checksum(path) != cell["reports"][phase]["sha256"]:
            raise ValueError("accepted report checksum mismatch")
        files.append(path)
        reports[phase] = json.loads(path.read_text())
    protected = {str(p): checksum(p) for p in files}
    before, after = reports["selected"], reports["final"]
    if set(before) != set(after) or len(before) != spec["symbols"]:
        raise ValueError("all original26 paired development episodes required")
    records, diagnoses = {}, {}
    for name in sorted(before):
        left, right = before[name], after[name]
        if left["symbol"] in records:
            raise ValueError("one episode per symbol required")
        diagnoses[left["symbol"]] = (diagnose_episode(left), diagnose_episode(right))
        records[left["symbol"]] = compare_paths(left, right)
        journal(dict(status="running", completed_symbols=list(records)))
    etfs = set(spec["etfs"])
    if not etfs <= set(records) or len(set(records) - etfs) != spec["stocks"]:
        raise ValueError("original22stock/fourETF panel required")
    groups = {}
    for kind in ("stock", "etf"):
        symbols = [s for s in records if (s in etfs) == (kind == "etf")]
        rows = [records[s] for s in symbols]
        bucket_keys = {k for r in rows for k in r["buckets"]}
        buckets = {
            k: dict(
                mean_sessions=mean(r["buckets"].get(k, {}).get("sessions", 0) for r in rows),
                mean_difference_pp=mean(
                    r["buckets"].get(k, {}).get("net_difference_pp", 0) for r in rows
                ),
            )
            for k in sorted(bucket_keys)
        }
        delta = mean(r["final_minus_selected_pp"] for r in rows)
        expected = 100 * (
            cell["final"]["groups"][kind]["net_return"]
            - cell["selected"]["groups"][kind]["net_return"]
        )
        if not math.isclose(delta, expected, abs_tol=1e-8, rel_tol=0):
            raise ValueError("accepted group return bridge mismatch")
        keys = rows[0]["first_differences"]
        contributions = {
            side: 100 * mean(diagnoses[s][1][side] - diagnoses[s][0][side] for s in symbols)
            for side in ("long_return_contribution", "short_return_contribution")
        }
        groups[kind] = dict(
            symbols=len(rows),
            mean_final_minus_selected_pp=delta,
            buckets=buckets,
            symbols_with_difference={
                k: sum(r["first_differences"][k] is not None for r in rows) for k in keys
            },
            mean_fraction_sessions_different={
                k: mean(r["difference_session_counts"][k] / r["sessions"] for r in rows)
                for k in keys
            },
            contribution_change_pp=contributions,
        )
    total = mean(r["final_minus_selected_pp"] for r in records.values())
    expected = 100 * (cell["final"]["score"] - cell["selected"]["score"])
    if not math.isclose(total, expected, abs_tol=1e-8, rel_tol=0):
        raise ValueError("accepted overall return bridge mismatch")
    if any(checksum(Path(p)) != h for p, h in protected.items()):
        raise ValueError("source files changed")
    result = dict(
        protocol_sha256=PROTOCOL_SHA256,
        per_symbol=records,
        groups=groups,
        mean_final_minus_selected_pp=total,
        source_files_sha256=protected,
        source_files_unchanged=True,
        duration_seconds=time.monotonic() - started,
        training_performed=False,
        optimizer_steps=0,
        models_loaded=False,
        provider_requested=False,
        dataset_opened=False,
        test_evaluated=False,
        checkpoint_selected=False,
        colab_allocated=False,
        interpretation=spec["interpretation"],
    )
    write_once(output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    progress = ROOT / ".runtime/critic-path-progress.json"
    if progress.exists():
        raise ValueError("existing journal requires review; no automatic restart")
    write_once(progress, b'{"status":"starting"}\n')

    def expire(*_):
        raise TimeoutError("fixed120-second saved-path audit deadline")

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(120)
    try:
        result = audit(args.output)
        journal(
            dict(
                status="complete",
                symbols=len(result["per_symbol"]),
                duration_seconds=result["duration_seconds"],
            )
        )
    except BaseException as error:
        journal(dict(status="failed", error_type=type(error).__name__))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    print("Reconciled all26 paired development paths; no models, training or new returns.")


if __name__ == "__main__":
    main()
