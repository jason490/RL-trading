"""Bounded, identified SEC archive acquisition; no model or broker operations."""
import fcntl
import hashlib
import json
import os
import re
import shlex
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rl_stocks.core.files import write_once


def identity(env_file='.env'):
    value = os.environ.get('SEC_USER_AGENT', '')
    if not value:
        path = Path(env_file)
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip().removeprefix('export ').lstrip()
                name, sep, raw = line.partition('=')
                if sep and name.strip() == 'SEC_USER_AGENT':
                    tokens = shlex.split(raw, comments=True)
                    if len(tokens) != 1:
                        raise ValueError('SEC_USER_AGENT must be a quoted single value')
                    value = tokens[0]
    if (not value or len(value) > 256 or any(ord(c) < 32 or ord(c) > 126 for c in value)
            or not re.search(r'\S+@\S+\.\S+', value)):
        raise ValueError('Configure SEC_USER_AGENT with application name and authorized contact email')
    return value


def archive_url(quarter):
    if not re.fullmatch(r'\d{4}q[1-4]', quarter) or not '2009q2' <= quarter <= '2023q4':
        raise ValueError('quarter outside fixed pre2024 audit scope')
    return f'https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip'


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('SEC redirect rejected; review source endpoint explicitly')


def download_archive(root, quarter, user_agent, *, max_bytes=256_000_000):
    """Acquire one pre2024 archive, with local cross-process request serialization.

    Maximum three attempts and a 120-second operation deadline. Access rejection
    and rate limiting stop immediately. The lock coordinates this cache root,
    not other machines; callers must coordinate any additional SEC clients.
    """
    url = archive_url(quarter)
    # Validate without reading or exporting an environment identity.
    if (not isinstance(user_agent, str) or not user_agent or len(user_agent) > 256
            or any(ord(c) < 32 or ord(c) > 126 for c in user_agent)
            or not re.search(r'\S+@\S+\.\S+', user_agent)):
        raise ValueError('invalid SEC request identity')
    if not isinstance(max_bytes, int) or not 0 < max_bytes <= 256_000_000:
        raise ValueError('invalid archive byte limit')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 120
    opener = build_opener(_NoRedirect())
    with (root / '.sec-download.lock').open('a+b') as lock:
        # Do not block indefinitely waiting for an unrelated acquisition.
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('SEC acquisition lock deadline exceeded') from None
                time.sleep(0.2)
        for receipt_path in sorted(root.glob('*/acquisition-*.json')):
            receipt_bytes = receipt_path.read_bytes()
            if hashlib.sha256(receipt_bytes).hexdigest() != receipt_path.stem.removeprefix('acquisition-'):
                raise ValueError('immutable SEC acquisition receipt corruption')
            receipt = json.loads(receipt_bytes)
            if receipt.get('quarter') != quarter or receipt.get('source_url') != url:
                continue
            archive = receipt_path.parent / 'archive.zip'
            with archive.open('rb') as cached:
                cached_hash = hashlib.file_digest(cached, 'sha256').hexdigest()
            if (cached_hash != receipt_path.parent.name
                    or cached_hash != receipt.get('archive_sha256')
                    or archive.stat().st_size != receipt.get('bytes')):
                raise ValueError('immutable SEC archive corruption')
            return archive
        for attempt in range(3):
            time.sleep(2)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('SEC acquisition deadline exceeded')
            temporary = None
            try:
                request = Request(url, headers={'User-Agent': user_agent, 'Accept': 'application/zip'})
                with opener.open(request, timeout=min(20, remaining)) as response:
                    if response.status != 200:
                        raise ValueError('unexpected SEC HTTP status')
                    size = response.headers.get('Content-Length')
                    if size is not None and (not size.isdigit() or int(size) > max_bytes):
                        raise ValueError('SEC response exceeds byte limit or invalid length')
                    with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                        temporary = Path(stream.name)
                        total = 0
                        sha = hashlib.sha256()
                        while True:
                            if time.monotonic() >= deadline:
                                raise TimeoutError('SEC acquisition deadline exceeded')
                            chunk = response.read(min(65536, max_bytes - total + 1))
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes:
                                raise ValueError('SEC response exceeds byte limit')
                            stream.write(chunk)
                            sha.update(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if size is not None and total != int(size):
                        raise ValueError('incomplete SEC response')
                    with temporary.open('rb') as downloaded:
                        signature = downloaded.read(4)
                    if total < 4 or signature != b'PK\x03\x04':
                        raise ValueError('SEC response is not a ZIP archive')
                    target = root / sha.hexdigest()
                    target.mkdir(exist_ok=True)
                    archive = target / 'archive.zip'
                    try:
                        os.link(temporary, archive)
                    except FileExistsError:
                        with archive.open('rb') as existing:
                            if hashlib.file_digest(existing, 'sha256').hexdigest() != sha.hexdigest():
                                raise ValueError('immutable SEC archive corruption') from None
                    receipt = dict(schema='sec-archive-acquisition-v1', quarter=quarter,
                                   source_url=url, archive_sha256=sha.hexdigest(), bytes=total,
                                   retrieved_utc=datetime.now(timezone.utc).isoformat(),
                                   attempt=attempt + 1, causal_ready=False)
                    receipt_bytes = (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode()
                    receipt_id = hashlib.sha256(receipt_bytes).hexdigest()
                    write_once(target / f'acquisition-{receipt_id}.json', receipt_bytes)
                    return archive
            except HTTPError as error:
                if error.code not in {500, 502, 503, 504} or attempt == 2:
                    raise ValueError(f'SEC HTTP {error.code}; acquisition stopped') from None
            except (URLError, TimeoutError, OSError):
                if attempt == 2 or time.monotonic() >= deadline:
                    raise ValueError('SEC transport failure; bounded acquisition stopped') from None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        raise ValueError('SEC bounded acquisition exhausted')
