"""No live provider calls: immutable cache, bounded streaming and failure retention."""

import pytest

from test_treasury_xml import document, row
from rl_stocks.data.providers.treasury.acquire import fetch_year


class Response:
    def __init__(self, raw, status=200, length=None):
        self.raw = raw
        self.status_code = status
        self.headers = {} if length is None else {"Content-Length": length}
        self.closed = False

    def iter_content(self, chunk_size):
        for i in range(0, len(self.raw), chunk_size):
            yield self.raw[i : i + chunk_size]

    def close(self):
        self.closed = True


def test_exact_raw_receipt_cache_and_no_second_request(tmp_path):
    response = Response(document([row()]))
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    first = fetch_year(tmp_path, 2020, deadline=10, get=get, clock=lambda: 0)
    second = fetch_year(
        tmp_path,
        2020,
        deadline=10,
        get=lambda *a, **k: pytest.fail("unexpected network"),
        clock=lambda: 0,
    )
    assert first["rows"] == second["rows"] and first["receipt_id"] == second["receipt_id"]
    assert first["network_requested"] and not second["network_requested"]
    assert response.closed and len(calls) == 1
    assert calls[0][1]["allow_redirects"] is False and calls[0][1]["timeout"] == (5, 10)
    assert (
        tmp_path / "raw" / (first["receipt"]["raw_sha256"] + ".xml")
    ).read_bytes() == response.raw


@pytest.mark.parametrize(
    "kind", ["http", "redirect", "oversize", "declared", "short", "html", "encoded"]
)
def test_failure_retained_no_index_no_retry(tmp_path, kind):
    raw = document([row()])
    status = 200
    length = None
    if kind == "http":
        status = 403
    if kind == "redirect":
        status = 302
    if kind == "oversize":
        raw = b"x" * (2 * 1024 * 1024 + 1)
    if kind == "declared":
        length = str(3 * 1024 * 1024)
    if kind == "short":
        length = str(len(raw) + 10)
    if kind == "html":
        raw = b"<html>access denied</html>"
    response = Response(raw, status, length)
    if kind == "encoded":
        response.headers["Content-Encoding"] = "gzip"
    calls = []

    def get(*a, **k):
        calls.append(1)
        return response

    with pytest.raises(ValueError):
        fetch_year(tmp_path, 2020, deadline=10, get=get, clock=lambda: 0)
    assert calls == [1] and response.closed
    assert not (tmp_path / "year-2020.json").exists()
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_corruption_rejected_without_provider_request(tmp_path):
    r = fetch_year(
        tmp_path,
        2020,
        deadline=10,
        get=lambda *a, **k: Response(document([row()])),
        clock=lambda: 0,
    )
    raw = tmp_path / "raw" / (r["receipt"]["raw_sha256"] + ".xml")
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fetch_year(
            tmp_path, 2020, deadline=10, get=lambda *a, **k: pytest.fail("network"), clock=lambda: 0
        )


def test_deadline_before_request_and_during_stream(tmp_path):
    with pytest.raises(TimeoutError):
        fetch_year(
            tmp_path, 2020, deadline=1, get=lambda *a, **k: pytest.fail("network"), clock=lambda: 2
        )
    times = iter([0, 0, 2])
    response = Response(document([row()]))
    with pytest.raises(TimeoutError):
        fetch_year(
            tmp_path, 2020, deadline=1, get=lambda *a, **k: response, clock=lambda: next(times)
        )
    assert response.closed and not (tmp_path / "year-2020.json").exists()
