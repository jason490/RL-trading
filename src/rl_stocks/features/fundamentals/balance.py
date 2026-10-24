"""Audit reported balance identities; never impute an unreported liability amount."""
from collections import defaultdict
from decimal import Decimal, localcontext

EQUITY = 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
TEMPORARY = 'TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests'


def inspect_balance(payload):
    if payload.get('schema') != 'sec-original-balance-audit-v1':
        raise ValueError('separate original balance-audit payload required')
    groups = defaultdict(list)
    taxonomies = set()
    for fact in payload['facts']:
        if fact['exclusion_reasons'] or fact['period_kind'] != 'instant' or fact['end'] != payload['report_end']:
            continue
        if (fact['accession'] != payload['accession'] or fact['cik'] != payload['cik']
                or fact['unit_measures'] != ['{http://www.xbrl.org/2003/iso4217}USD']):
            raise ValueError('coherent issuer/accession/period/USD scope required')
        namespace, concept = fact['concept'].split('}', 1)
        taxonomies.add(namespace)
        value = Decimal(fact['value'])
        if not value.is_finite():
            raise ValueError('finite reported balance amount required')
        groups[concept].append(value)
    conflicts = sorted(k for k, values in groups.items() if len(set(values)) != 1)
    amounts = {k: values[0] for k, values in groups.items() if len(set(values)) == 1}
    result = dict(schema='reported-balance-inspection-v1', accession=payload['accession'],
                  cik=payload['cik'], report_end=payload['report_end'], source_sha256=payload['source_sha256'],
                  reported_concepts=sorted(groups), conflicting_concepts=conflicts,
                  arithmetic_scope_ready=not conflicts and len(taxonomies) <= 1,
                  liability_inference_permitted=False, financial_features_enabled=False,
                  missing_temporary_equity_is_zero=False)
    if not result['arithmetic_scope_ready']:
        return dict(result, reason='conflicting_values_or_taxonomies')
    assets, liabilities = amounts.get('Assets'), amounts.get('Liabilities')
    equity, parent = amounts.get(EQUITY), amounts.get('StockholdersEquity')
    temporary = amounts.get(TEMPORARY)
    result['reported_total_temporary_equity'] = str(temporary) if temporary is not None else None
    nonzero = [v for v in amounts.values() if v]
    precision = max(50, max((v.adjusted() for v in nonzero), default=0)
                    - min((v.as_tuple().exponent for v in nonzero), default=0) + 4)
    if precision > 10000:
        raise ValueError('unsupported source numeric precision span')
    with localcontext() as context:
        context.prec = precision
        if assets is not None and liabilities is not None:
            if equity is not None:
                residual = assets - liabilities - equity
                result['assets_minus_liabilities_minus_equity_including_nci'] = str(residual)
                result['three_term_reported_identity_holds'] = residual == 0
                if temporary is not None:
                    result['residual_after_reported_total_temporary_equity'] = str(residual - temporary)
                    result['four_term_reported_identity_holds'] = residual == temporary
            if parent is not None:
                result['assets_minus_liabilities_minus_parent_equity'] = str(assets - liabilities - parent)
        else:
            result['reason'] = 'missing_direct_assets_or_liabilities_no_inference'
    return result
