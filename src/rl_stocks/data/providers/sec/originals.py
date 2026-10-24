"""Bounded single-request original XBRL acquisition for fixed source review."""
import fcntl
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import _NoRedirect


class OriginalMissing(Exception):
    """The selected source returned404; retain the case without replacement."""


def source_url(record):
    cik, accession, filename = record['cik'], record['accession'], record['instance_filename']
    if (not re.fullmatch(r'[1-9][0-9]{0,9}', cik)
            or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession)
            or not re.fullmatch(r'[A-Za-z0-9_.-]+\.xml', filename) or '..' in filename
            or not re.fullmatch(r'20(?:0[9]|1[0-9]|2[0-3])\d{4}', record['archive_filed'])):
        raise ValueError('fixed pre2024 original source metadata required')
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"
    if record['source_url'] != url:
        raise ValueError('original endpoint differs from fixed ledger')
    return url


def acquire_original(root, record, user_agent):
    """One request, no retries/redirects,20MB and60s checked deadline, private identity."""
    url = source_url(record)
    if (not isinstance(user_agent, str) or not user_agent or len(user_agent) > 256
            or any(ord(c) < 32 or ord(c) > 126 for c in user_agent)
            or not re.search(r'\S+@\S+\.\S+', user_agent)):
        raise ValueError('valid private SEC request identity required')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    receipt_path = root / 'original-receipts' / (key + '.json')
    deadline = time.monotonic() + 60
    with (root / '.original-download.lock').open('a+b') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('original acquisition lock deadline') from None
                time.sleep(0.2)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            sha = receipt.get('source_sha256', '')
            if (receipt.get('source_url') != url or not re.fullmatch(r'[a-f0-9]{64}', sha)
                    or receipt.get('schema') != 'sec-original-acquisition-v1'):
                raise ValueError('original cache receipt corruption')
            path = root / sha / 'instance.xml'
            raw = path.read_bytes() if path.stat().st_size <= 20_000_000 else b''
            if hashlib.sha256(raw).hexdigest() != sha or len(raw) != receipt.get('bytes'):
                raise ValueError('original cache source corruption')
            return path
        time.sleep(2)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('original acquisition deadline')
            request = Request(url, headers={'User-Agent': user_agent, 'Accept': 'application/xml'})
            with build_opener(_NoRedirect()).open(request, timeout=min(20, remaining)) as response:
                if response.status != 200:
                    raise ValueError('unexpected original source HTTP status')
                length = response.headers.get('Content-Length')
                if length is not None and (not length.isdigit() or int(length) > 20_000_000):
                    raise ValueError('invalid or excessive original source length')
                chunks, total = [], 0
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('original acquisition deadline')
                    chunk = response.read(min(65536, 20_000_001 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 20_000_000:
                        raise ValueError('original source exceeds byte limit')
                    chunks.append(chunk)
                raw = b''.join(chunks)
                if length is not None and len(raw) != int(length):
                    raise ValueError('incomplete original source response')
                if not raw:
                    raise ValueError('empty original source response')
        except HTTPError as error:
            if error.code == 404:
                raise OriginalMissing('selected original source HTTP404') from None
            raise ValueError(f'original source HTTP{error.code}; collection must stop') from None
        except (URLError, OSError):
            raise ValueError('original source transport failed; collection must stop') from None
        sha = hashlib.sha256(raw).hexdigest()
        path = root / sha / 'instance.xml'
        write_once(path, raw)
        receipt = dict(schema='sec-original-acquisition-v1', source_url=url,
                       source_sha256=sha, bytes=len(raw),
                       retrieved_utc=datetime.now(timezone.utc).isoformat(), causal_ready=False)
        write_once(receipt_path, (json.dumps(receipt, indent=2) + '\n').encode())
        return path
