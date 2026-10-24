"""Coherent-accession financial arithmetic; publication/price joins remain separate."""
import math
from collections import defaultdict
from datetime import date
from decimal import Decimal, DecimalException, localcontext

NAMES = ('filing_revenue_growth', 'filing_net_margin', 'filing_liabilities_assets')
REVENUE = frozenset({'Revenues', 'SalesRevenueNet',
                     'RevenueFromContractWithCustomerExcludingAssessedTax',
                     'RevenueFromContractWithCustomerIncludingAssessedTax'})
BOUNDS = {'10-K': (330, 380), '10-K/A': (330, 380), '10-Q': (28, 330), '10-Q/A': (28, 330)}
USD = ['{http://www.xbrl.org/2003/iso4217}USD']


def build_snapshot(payload, form):
    """Original three-feature contract, including direct reported liabilities."""
    return _build_snapshot(payload, form, include_balance=True)


def build_flow_snapshot(payload, form):
    """Separate two-flow research family; never infer or fill balance amounts."""
    return _build_snapshot(payload, form, include_balance=False)


def _build_snapshot(payload, form, *, include_balance):
    if payload.get('schema') != 'sec-original-money-facts-v1' or form not in BOUNDS:
        raise ValueError('supported original fact payload and filing form required')
    end = date.fromisoformat(payload['report_end'])
    filed = date.fromisoformat(payload['filed'])
    if end > filed or filed >= date(2024, 1, 1):
        raise ValueError('invalid or retired original filing dates')
    result = dict(schema=('filing-financial-snapshot-v1' if include_balance
                          else 'filing-flow-snapshot-v1'),
                  names=list(NAMES if include_balance else NAMES[:2]), values=None,
                  source_sha256=payload['source_sha256'], accession=payload['accession'],
                  cik=payload['cik'], form=form, report_end=payload['report_end'], filed=payload['filed'],
                  arithmetic_ready=False, financial_features_enabled=False,
                  publication_verified=False, historical_price_link_verified=False)

    def unready(reason):
        return dict(result, reason=reason)

    groups = defaultdict(lambda: defaultdict(list))
    instants = defaultdict(list)
    taxonomies = set()
    for fact in payload['facts']:
        if fact['exclusion_reasons']:
            continue
        if (fact['accession'] != payload['accession'] or fact['cik'] != payload['cik']
                or fact['unit_measures'] != USD):
            raise ValueError('mixed accession/issuer/unit in unqualified source facts')
        finish = date.fromisoformat(fact['end'])
        if finish > end:
            raise ValueError('unexcluded forward source fact')
        value = Decimal(fact['value'])
        if not value.is_finite():
            raise ValueError('nonfinite source amount')
        namespace, concept = fact['concept'].split('}', 1)
        if not namespace.startswith(('{http://fasb.org/us-gaap/', '{https://fasb.org/us-gaap/',
                                     '{http://xbrl.us/us-gaap/', '{https://xbrl.us/us-gaap/')):
            raise ValueError('standard US-GAAP source concepts required')
        taxonomies.add(namespace)
        if fact['period_kind'] == 'instant':
            if finish == end and concept in {'Assets', 'Liabilities'}:
                instants[concept].append(value)
        elif fact['period_kind'] == 'duration':
            start = date.fromisoformat(fact['start'])
            if start > finish:
                raise ValueError('reversed source period')
            if concept in REVENUE or concept == 'NetIncomeLoss':
                groups[(start, finish)][concept].append(value)
        else:
            raise ValueError('unsupported source period kind')

    if len(taxonomies) > 1:
        return unready('mixed_us_gaap_taxonomy_versions')
    lower, upper = BOUNDS[form]
    current_candidates = [period for period, concepts in groups.items()
                          if period[1] == end and lower <= (period[1] - period[0]).days + 1 <= upper
                          and set(concepts) & REVENUE and 'NetIncomeLoss' in concepts]
    if not current_candidates:
        return unready('missing_current_comparable_flows')
    current = min(current_candidates, key=lambda period: period[0])
    current_days = (current[1] - current[0]).days + 1

    def unique(values):
        amounts = set(values)
        return next(iter(amounts)) if len(amounts) == 1 else None

    def revenue(period):
        concepts = groups[period]
        return unique([v for name in REVENUE for v in concepts.get(name, [])])

    current_revenue = revenue(current)
    income = unique(groups[current]['NetIncomeLoss'])
    if current_revenue is None or income is None:
        return unready('conflicting_current_flow_values')
    prior_candidates = [period for period, concepts in groups.items()
                        if set(concepts) & REVENUE
                        and 350 <= (current[0] - period[0]).days <= 380
                        and 350 <= (current[1] - period[1]).days <= 380
                        and abs(((period[1] - period[0]).days + 1) - current_days) <= 7]
    if len(prior_candidates) != 1:
        return unready('missing_or_ambiguous_prior_interval')
    prior = prior_candidates[0]
    if not set(groups[current]) & set(groups[prior]) & REVENUE:
        return unready('no_same_revenue_concept_comparable')
    prior_revenue = revenue(prior)
    assets, liabilities = unique(instants['Assets']), unique(instants['Liabilities'])
    if prior_revenue is None or (include_balance and (assets is None or liabilities is None)):
        return unready('missing_or_conflicting_prior_or_balance_values' if include_balance
                       else 'conflicting_prior_revenue_values')
    if (current_revenue <= 0 or prior_revenue <= 0
            or (include_balance and (assets <= 0 or liabilities < 0))):
        return unready('invalid_denominator_or_liabilities' if include_balance
                       else 'invalid_revenue_denominator')
    try:
        with localcontext() as context:
            context.prec = 50
            values = [float(current_revenue / prior_revenue - 1),
                      float(income / current_revenue)]
            if include_balance:
                values.append(float(liabilities / assets))
    except (DecimalException, OverflowError):
        return unready('nonfinite_ratio')
    if not all(math.isfinite(v) for v in values):
        return unready('nonfinite_ratio')
    return dict(result, arithmetic_ready=True, reason=None, values=values,
                current_interval=[d.isoformat() for d in current],
                prior_interval=[d.isoformat() for d in prior], interval_days=current_days,
                comparability_scope='Reported same-accession periods; not organic or merger-adjusted growth')
