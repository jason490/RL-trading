"""Feature-specific type evidence; never a historical security or financial master."""
from datetime import date
import re

from rl_stocks.core.contracts import canonical
from rl_stocks.data.validation.bars import timestamp


def verification_scope():
    return {'version': 1, 'verified_property': 'initial_coarse_stock_etf_type',
            'series_identity': 'existing_dataset_provider_association_assumed',
            'continuity': 'fixed_kind_research_contract; reject_documented_mixed_kind_paths',
            'historical_price_link_verified': False, 'financial_comparability_verified': False,
            'historical_membership_verified': False, 'numeric_financial_inputs_enabled': False}


def validate_scope(evidence):
    if canonical(evidence.get('verification_scope')) != canonical(verification_scope()):
        raise ValueError('explicit coarse type-only verification scope required')


def validate_source_point(record):
    if (record.get('kind') not in ('stock', 'etf')
            or record.get('source_fact_kind') != record['kind']
            or record.get('evidence_kind') != 'historical_primary_source'
            or not record.get('source')
            or not re.fullmatch(r'[a-f0-9]{64}', record.get('source_sha256', ''))):
        raise ValueError('historical source type fact and checksum required')
    known = date.fromisoformat(record['known_from'])
    available = timestamp(record['source_available_time'])
    if known.isoformat() != record['known_from'] or available.date() != known:
        raise ValueError('known-from date must match documented source availability')
    return known
