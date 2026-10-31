"""Single-request immutable SEC filing HTML acquisition for historical review."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import _NoRedirect


def validate_endpoint(url, cik, accession):
    if (not re.fullmatch(r'[1-9][0-9]{0,9}', cik)
            or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession)):
        raise ValueError('explicit normalized SEC filing identity required')
    year = int(accession[11:13])
    if 24 <= year <= 92:
        raise ValueError('retired or unsupported filing accession rejected')
    prefix = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace("-", "")}/'
    if not url.startswith(prefix):
        raise ValueError('filing endpoint differs from declared identity')
    name = url[len(prefix):]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.html?', name) or '..' in name:
        raise ValueError('plain original SEC HTML endpoint required')
    return url


def _acquire_filing_html(root, *, url, cik, accession, user_agent):
    """Reuse exact cached bytes, otherwise one request; no redirects/retries.

    Callers serialize collections across processes, retain every selected failure
    and stop on rejection.20MB/60s bounds; this function never admits metadata.
    """
    validate_endpoint(url, cik, accession)
    if (not isinstance(user_agent, str) or not user_agent or len(user_agent) > 256
            or any(ord(c) < 32 or ord(c) > 126 for c in user_agent)
            or not re.search(r'\S+@\S+\.\S+', user_agent)):
        raise ValueError('valid private SEC request identity required')
    root = Path(root)
    for receipt_path in sorted(root.glob('*/receipt.json')):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('source_url') != url:
            continue
        sha = receipt.get('sha256', '')
        if (not re.fullmatch(r'[a-f0-9]{64}', sha) or receipt_path.parent.name != sha):
            raise ValueError('immutable SEC HTML receipt mismatch')
        path = receipt_path.parent / 'filing.htm'
        with path.open('rb') as stream:
            raw = stream.read(20_000_001)
        if (len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != sha
                or len(raw) != receipt.get('bytes')):
            raise ValueError('immutable SEC HTML cache corruption')
        return path, receipt, True
    deadline = time.monotonic() + 60
    time.sleep(2)  # Well below SEC's aggregate limit; do not parallelize collectors.
    try:
        request = Request(url, headers={'User-Agent': user_agent, 'Accept': 'text/html'})
        with build_opener(_NoRedirect()).open(request, timeout=20) as response:
            if response.status != 200:
                raise ValueError('unexpected SEC HTML status')
            declared = response.headers.get('Content-Length')
            if declared is not None and (not declared.isdigit() or int(declared) > 20_000_000):
                raise ValueError('SEC HTML declared size rejected')
            chunks, total = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('SEC HTML collection deadline')
                chunk = response.read(min(65536, 20_000_001 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > 20_000_000:
                    raise ValueError('SEC HTML source exceeds byte limit')
            raw = b''.join(chunks)
            if not raw or (declared is not None and len(raw) != int(declared)):
                raise ValueError('incomplete SEC HTML response')
            if b'<html' not in raw.lower():
                raise ValueError('original HTML source required')
    except HTTPError as error:
        raise ValueError(f'SEC HTML HTTP{error.code}; stop collection without retry') from None
    except (URLError, OSError):
        raise ValueError('SEC HTML transport failed; stop collection without retry') from None
    sha = hashlib.sha256(raw).hexdigest()
    path = root / sha / 'filing.htm'
    receipt = {'schema': 'sec-filing-html-acquisition-v1', 'source_url': url, 'sha256': sha,
               'bytes': len(raw), 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
               'accession': accession, 'cik': cik, 'historical_metadata_admitted': False}
    write_once(path, raw)
    write_once(path.parent / 'receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
    return path, receipt, False


def acquire_filing_html(root, *, url, cik, accession, user_agent):
    """Reject concurrent collectors immediately; never overlap owned SEC requests."""
    validate_endpoint(url, cik, accession)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.html-download.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('SEC HTML collector busy; no request issued') from None
        return _acquire_filing_html(root, url=url, cik=cik, accession=accession,
                                    user_agent=user_agent)
