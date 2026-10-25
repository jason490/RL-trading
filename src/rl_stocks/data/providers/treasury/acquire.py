"""Bounded official annual acquisition; one collector owns pacing and operation lock."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.providers.treasury.xml import annual_url, parse_annual, MAX_BYTES


def cached_year(cache, year):
    cache = Path(cache)
    url = annual_url(year)
    index = cache / f"year-{year}.json"
    if not index.exists():
        return None
    pointer = json.loads(index.read_text())
    receipt_id = pointer["receipt_id"]
    if (
        not isinstance(receipt_id, str)
        or len(receipt_id) != 64
        or any(c not in "0123456789abcdef" for c in receipt_id)
    ):
        raise ValueError("invalid immutable Treasury receipt identity")
    receipt = json.loads((cache / "receipts" / (receipt_id + ".json")).read_text())
    if digest(receipt) != receipt_id or receipt["url"] != url or receipt["year"] != year:
        raise ValueError("immutable Treasury receipt mismatch")
    sha = receipt["raw_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid Treasury raw identity")
    raw = (cache / "raw" / (sha + ".xml")).read_bytes()
    if receipt["bytes"] != len(raw) or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("immutable Treasury raw bytes changed")
    if receipt["http_status"] != 200 or receipt["complete"] is not True:
        raise ValueError("incomplete Treasury source is not reusable")
    return dict(
        receipt_id=receipt_id,
        receipt=receipt,
        rows=parse_annual(raw, year),
        network_requested=False,
    )


def fetch_year(cache, year, *, deadline, get=None, clock=time.monotonic):
    """No retries, redirects or pagination; complete raw bytes precede parsing."""
    cache = Path(cache)
    url = annual_url(year)
    if clock() >= deadline:
        raise TimeoutError("Treasury operation deadline")
    cached = cached_year(cache, year)
    if cached is not None:
        return cached
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError("Treasury operation deadline")
    response = (get or requests.get)(
        url,
        timeout=(min(5.0, remaining), min(30.0, remaining)),
        stream=True,
        allow_redirects=False,
        headers={
            "User-Agent": "rl-stocks-research/0.1",
            "Accept": "application/atom+xml, application/xml, text/xml",
            "Accept-Encoding": "identity",
        },
    )
    raw = bytearray()
    complete = False
    failure = None
    try:
        if response.status_code != 200:
            raise ValueError(f"Treasury HTTP status{response.status_code}; no retry")
        if response.headers.get("Content-Encoding", "identity").strip().lower() not in (
            "",
            "identity",
        ):
            raise ValueError("Treasury encoded response forbidden")
        declared = response.headers.get("Content-Length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_BYTES):
            raise ValueError("Treasury declared response limit")
        for chunk in response.iter_content(chunk_size=65536):
            if clock() >= deadline:
                raise TimeoutError("Treasury streaming deadline")
            if not isinstance(chunk, bytes):
                raise ValueError("Treasury transport must yield bytes")
            raw.extend(chunk[: max(0, MAX_BYTES + 1 - len(raw))])
            if len(raw) > MAX_BYTES:
                raise ValueError("Treasury streaming response limit")
        if declared is not None and len(raw) != int(declared):
            raise ValueError("Treasury incomplete Content-Length")
        complete = True
    except BaseException as error:
        failure = error
    finally:
        response.close()
    content = bytes(raw)
    raw_sha = hashlib.sha256(content).hexdigest()
    receipt = dict(
        schema="treasury-annual-response-v1",
        year=year,
        url=url,
        http_status=response.status_code,
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        bytes=len(content),
        raw_sha256=raw_sha,
        complete=complete,
        error_type=None if failure is None else type(failure).__name__,
        historical_publication_vintages_verified=False,
    )
    receipt_id = digest(receipt)
    write_once(cache / "raw" / (raw_sha + ".xml"), content)
    write_once(cache / "receipts" / (receipt_id + ".json"), canonical(receipt))
    if failure is not None:
        raise failure
    rows = parse_annual(content, year)
    write_once(cache / f"year-{year}.json", canonical(dict(receipt_id=receipt_id)))
    return dict(receipt_id=receipt_id, receipt=receipt, rows=rows, network_requested=True)
