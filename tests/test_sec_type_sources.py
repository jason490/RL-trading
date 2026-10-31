from rl_stocks.data.providers.sec.type_sources import select_stock_sources


def test_selection_prefers_annual_but_excludes_unavailable_and_wrong_cik():
    types = [{'symbol': 'A', 'kind': 'stock', 'first_training_session': '2010-03-16'}]
    candidates = [{'cik': '1', 'discovery_symbols': ['A']}]
    def row(cik, form, day, accession):
        return dict(cik=cik, form=form, filed=day, accession=accession)
    records = [row('1', '10-K', '2010-02-01', 'annual'),
               row('1', '10-Q', '2010-03-01', 'quarter'),
               row('1', '10-K', '2010-03-15', 'too-late'),
               row('2', '10-K', '2010-03-01', 'wrong-issuer')]
    result = select_stock_sources(types, candidates, records)
    assert result[0]['master_record']['accession'] == 'annual'
    assert result[0]['collection_needed']
    assert not result[0]['historical_identity_verified']
    assert not select_stock_sources(types, candidates, records, {'A'})[0]['collection_needed']


def test_unresolved_identity_never_gets_fallback_candidate():
    result = select_stock_sources([{'symbol': 'A', 'kind': 'stock',
                                    'first_training_session': '2010-03-16'}], [], [])
    assert result[0]['master_record'] is None
    assert not result[0]['collection_needed']
