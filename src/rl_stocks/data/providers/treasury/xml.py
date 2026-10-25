"""Strict annual Treasury XML normalization; no network access or model inputs."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import xml.etree.ElementTree as ET

ATOM = "http://www.w3.org/2005/Atom"
DATA = "http://schemas.microsoft.com/ado/2007/08/dataservices"
META = "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
FIELDS = ("BC_2YEAR", "BC_10YEAR")
MAX_BYTES = 2 * 1024 * 1024
ENDPOINT = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"


def annual_url(year):
    if type(year) is not int or not 2009 <= year <= 2023:
        raise ValueError("Treasury audit only permits annual2009–2023 requests")
    return ENDPOINT + f"?data=daily_treasury_yield_curve&field_tdr_date_value={year}"


def _one(parent, name):
    values = parent.findall(name)
    if len(values) != 1:
        raise ValueError("one exact Treasury source element required")
    return values[0]


def _yield(properties, name):
    fields = properties.findall(f"{{{DATA}}}{name}")
    if not fields:
        return None
    if len(fields) != 1:
        raise ValueError("duplicate Treasury yield field")
    field = fields[0]
    null = field.get(f"{{{META}}}null")
    text = (field.text or "").strip()
    if null is not None and null not in ("true", "false"):
        raise ValueError("invalid Treasury null flag")
    if null == "true":
        if text:
            raise ValueError("conflicting null Treasury value")
        return None
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("invalid Treasury yield") from exc
    if not number.is_finite() or not -100 <= number <= 100:
        raise ValueError("nonfinite or impossible percent yield")
    return str(number)


def parse_annual(raw, year):
    """Retain decimal source units and missingness; source dates are not UTC times."""
    annual_url(year)
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("bounded Treasury XML bytes required")
    try:
        text_xml = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Treasury XML must be UTF8") from exc
    if "<!DOCTYPE" in text_xml.upper() or "<!ENTITY" in text_xml.upper():
        raise ValueError("Treasury XML declarations/entities forbidden")
    try:
        root = ET.fromstring(text_xml)
    except ET.ParseError as exc:
        raise ValueError("invalid Treasury XML") from exc
    if root.tag != f"{{{ATOM}}}feed":
        raise ValueError("Treasury Atom feed required")
    if any(link.get("rel") == "next" for link in root.findall(f"{{{ATOM}}}link")):
        raise ValueError("annual Treasury pagination forbidden")
    entries = root.findall(f"{{{ATOM}}}entry")
    if not 0 < len(entries) <= 300:
        raise ValueError("bounded nonempty Treasury year required")
    rows, seen = [], set()
    for entry in entries:
        content = _one(entry, f"{{{ATOM}}}content")
        properties = _one(content, f"{{{META}}}properties")
        stamp = _one(properties, f"{{{DATA}}}NEW_DATE")
        if stamp.get(f"{{{META}}}null") == "true":
            raise ValueError("Treasury observation date cannot be null")
        text = (stamp.text or "").strip()
        try:
            parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
        except ValueError as exc:
            raise ValueError("Treasury date schema changed") from exc
        if parsed.isoformat() != text or parsed.time().isoformat() != "00:00:00":
            raise ValueError("Treasury date must be source-local midnight label")
        day = parsed.date()
        if day.year != year or day >= date(2024, 1, 1) or day.weekday() >= 5:
            raise ValueError("Treasury observation outside requested business year")
        if day in seen:
            raise ValueError("duplicate Treasury observation date")
        seen.add(day)
        values = {name: _yield(properties, name) for name in FIELDS}
        rows.append(dict(observation_date=day.isoformat(), **values))
    if not any(all(r[name] is not None for name in FIELDS) for r in rows):
        raise ValueError("no complete requested Treasury maturities in year")
    return sorted(rows, key=lambda r: r["observation_date"])
