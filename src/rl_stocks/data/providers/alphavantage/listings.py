"""Dated listing candidates are not verified point-in-time index constituents."""
import csv
from datetime import date
import io
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from rl_stocks.core.contracts import canonical, digest, utcnow
from rl_stocks.core.files import write_once
from rl_stocks.data.validation.bars import timestamp

FIELDS = {'symbol','name','exchange','assetType','ipoDate','delistingDate','status'}


def parse_listing_csv(text, asof_date, state, retrieved_time):
    if date.fromisoformat(asof_date).isoformat() != asof_date or date.fromisoformat(asof_date) <= date(2010,1,1) or state not in ('active','delisted'):
        raise ValueError('invalid Alpha Vantage date/state')
    if date.fromisoformat(asof_date) > timestamp(retrieved_time).date():
        raise ValueError('future listing snapshot requested')
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if set(reader.fieldnames or ()) != FIELDS:
        raise ValueError('listing response is not the documented CSV; check quota/entitlement')
    records, seen = [], set()
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('malformed listing CSV')
        row = {k:v.strip() for k,v in row.items()}
        if not row['symbol'] or not row['exchange'] or row['assetType'] not in ('Stock','ETF'):
            raise ValueError('invalid listing identity/type')
        if row['status'].lower() != state:
            raise ValueError('response state differs from requested state')
        dates = {key: None if row[key].lower() in ('','null','none') else date.fromisoformat(row[key]).isoformat()
                 for key in ('ipoDate','delistingDate')}
        identity = (row['symbol'], row['exchange'], dates['ipoDate'])
        if identity in seen:
            raise ValueError('duplicate listing candidate')
        seen.add(identity)
        records.append({**row, **dates, 'asof_date':asof_date, 'available_time':retrieved_time,
                        'retrieved_time':retrieved_time, 'source':'alphavantage-listing-status',
                        'stable_security_id':None, 'identity_verified':False,
                        'historical_publication_vintage_verified':False,
                        'terminal_return_verified':False})
    if not records:
        raise ValueError('empty listing response is not complete coverage evidence')
    return records


def fetch_listings(asof_date, state, output, *, opener=None):
    # Validate before issuing any metered request. Never log a credential-bearing URL.
    if date.fromisoformat(asof_date).isoformat() != asof_date or date.fromisoformat(asof_date) <= date(2010,1,1) or state not in ('active','delisted'):
        raise ValueError('invalid Alpha Vantage date/state')
    if date.fromisoformat(asof_date) > timestamp(utcnow()).date():
        raise ValueError('future listing snapshot requested')
    key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not key:
        raise ValueError('set ALPHAVANTAGE_API_KEY privately; do not put it in arguments or chat')
    query = urlencode({'function':'LISTING_STATUS','date':asof_date,'state':state,'apikey':key})
    try:
        with (opener or urlopen)('https://www.alphavantage.co/query?'+query, timeout=30) as response:
            raw = response.read(50_000_001)
        if len(raw) > 50_000_000:
            raise ValueError('oversized listing response')
        text = raw.decode('utf-8-sig')
    except Exception:
        raise RuntimeError('Alpha Vantage listing request failed; credentials and response URL suppressed') from None
    if key in text:
        raise ValueError('response contains credential text; refusing to persist it')
    retrieved = utcnow()
    records = parse_listing_csv(text, asof_date, state, retrieved)
    payload = {'version':1,'provider':'alphavantage','asof_date':asof_date,'state':state,
               'retrieved_time':retrieved,'records':records,'raw_csv':text,
               'usable_as_verified_membership':False,
               'limitations':'Listing existence discovery only; no stable-ID mapping, index membership, publication vintage or terminal-return proof.'}
    target = Path(output)/digest(payload)
    target.mkdir(parents=True,exist_ok=True)
    write_once(target/'listings.json',canonical(payload))
    return target
