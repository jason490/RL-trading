"""Synthetic evidence: no source records here establish real market coverage."""
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import io
from rl_stocks.core.contracts import Bar
from rl_stocks.data.universe.historical.schema import ATTESTATIONS, validate
from rl_stocks.data.universe.historical.registry import member_asof, ticker_asof, resolve_ticker
from rl_stocks.data.universe.historical.storage import put_history, load_history
from rl_stocks.data.universe.historical.coverage import audit, terminal_proceeds
from rl_stocks.reporting.historical import report
from rl_stocks.data.providers.alphavantage.listings import parse_listing_csv, fetch_listings


def fixture():
    import exchange_calendars as xc
    common = {'source':'synthetic-fixture-only','revision':'v1',
              'available_time':'2019-12-31T12:00:00+00:00',
              'retrieved_time':'2026-10-17T00:00:00+00:00'}
    payload = {'version':1,'universe':{'name':'fixture-index','start':'2020-01-02',
               'end':'2020-01-10','calendar':'XNYS','source':'synthetic-fixture-only',
               'selection_rule':'Synthetic rotating index; no real market evidence'},
               'attestations':{k:{'verified':True,'source':'synthetic-only','revision':'v1'} for k in ATTESTATIONS},
               'securities':[], 'aliases':[], 'membership':[], 'terminals':[], 'prices':[]}
    for sid, listed, last, status in [('old-security','2010-01-01','2020-01-07','delisted'),
                                      ('new-security','2020-01-08',None,'active')]:
        payload['securities'].append({**common,'security_id':sid,'issuer_id':'issuer-'+sid,
           'exchange':'XNYS','instrument_type':'stock','listed_from':listed,
           'last_trade_session':last,'status':status})
        payload['aliases'].append({**common,'event_id':'alias-'+sid,'security_id':sid,
                                   'ticker':'ACME','effective_from':listed})
        payload['membership'].append({**common,'event_id':'add-'+sid,'security_id':sid,
                                      'universe':'fixture-index','member':True,
                                      'effective_from':max(listed,'2020-01-02')})
    payload['membership'].append({**common,'event_id':'remove-old','security_id':'old-security',
                                  'universe':'fixture-index','member':False,'effective_from':'2020-01-08'})
    payload['terminals'].append({**common,'event_id':'terminal-old','security_id':'old-security',
         'available_time':'2020-01-10T12:00:00+00:00', 'last_trade_session':'2020-01-07',
         'settlement_time':'2020-01-10T12:00:00+00:00','kind':'cash_per_share',
         'reference_price':100.,'value':0.,'currency':'USD','verified':True,
         'basis':'additional_to_last_close'})
    cal = xc.get_calendar('XNYS',start='2020-01-02',end='2020-01-10')
    for session in cal.sessions:
        day = str(session.date())
        if day >= '2020-01-10':continue
        sid = 'old-security' if day <= '2020-01-07' else 'new-security'
        op, close = cal.session_open(session), cal.session_close(session)
        bar = Bar('ACME',day,op.isoformat(),close.isoformat(),
                  (close+timedelta(minutes=15)).isoformat(),common['retrieved_time'],
                  'synthetic','synthetic','v1',100.,101.,99.,100.,100000.)
        payload['prices'].append({'security_id':sid,'bar':asdict(bar)})
    return payload


class HistoricalUniverseTests(unittest.TestCase):
    def test_full_fixture_covers_dead_member_and_recycled_ticker(self):
        payload = fixture();result = audit(payload)
        self.assertTrue(result['coverage_passed'])
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['totals']['selected_securities'],2)
        self.assertEqual(result['totals']['expected_member_sessions'],6)
        self.assertEqual(result['totals']['terminal_outcomes_required'],1)
        self.assertEqual(resolve_ticker(payload,'ACME','2020-01-03','2020-01-03T21:15:00+00:00'),'old-security')
        self.assertEqual(resolve_ticker(payload,'ACME','2020-01-09','2020-01-09T21:15:00+00:00'),'new-security')

    def test_future_membership_and_alias_revisions_do_not_leak_backwards(self):
        payload = fixture();row = deepcopy(payload['membership'][0])
        row.update(member=False, revision='v2', available_time='2021-01-01T12:00:00+00:00')
        payload['membership'].append(row)
        payload['aliases'].append({**payload['aliases'][1], 'event_id':'rename-new',
            'ticker':'BETA','effective_from':'2020-01-09','available_time':'2020-01-09T21:30:00+00:00'})
        validate(payload)
        self.assertTrue(member_asof(payload,'old-security','2020-01-03','2020-01-03T21:15:00+00:00'))
        self.assertFalse(member_asof(payload,'old-security','2020-01-03','2021-01-03T21:15:00+00:00'))
        self.assertEqual(ticker_asof(payload,'new-security','2020-01-09','2020-01-09T21:15:00+00:00'),'ACME')
        self.assertEqual(ticker_asof(payload,'new-security','2020-01-09','2020-01-09T21:30:00+00:00'),'BETA')
        # Changing future outcome labels cannot change member selection.
        other = deepcopy(payload);other['terminals'][0]['value'] = 100.
        self.assertEqual(audit(payload)['eligible_panel'],audit(other)['eligible_panel'])

    def test_missing_failed_stock_prices_are_reported_not_dropped(self):
        payload = fixture();payload['prices'] = [r for r in payload['prices'] if r['security_id']!='old-security']
        result = audit(payload)
        self.assertFalse(result['coverage_passed'])
        self.assertEqual(result['totals']['selected_securities'],2)
        self.assertEqual(result['issue_counts']['missing_member_price'],4)
        self.assertIn('missing_last_trade_price',result['issue_counts'])

    def test_missing_unresolved_non_cash_and_bad_reference_outcomes_block(self):
        for mode in ('missing','unresolved','non_cash','wrong_reference'):
            with self.subTest(mode=mode):
                payload = fixture()
                if mode=='missing':payload['terminals']=[]
                elif mode=='wrong_reference':payload['terminals'][0]['reference_price']=99.
                else:payload['terminals'][0].update(kind=mode,value=None,reference_price=None)
                self.assertFalse(audit(payload)['coverage_passed'])
        payload=fixture();payload['securities'][0]['last_trade_session']=None
        self.assertIn('missing_last_trade_session',audit(payload)['issue_counts'])

    def test_zero_settlement_short_gain_buyout_and_missing_codes(self):
        event=fixture()['terminals'][0]
        self.assertEqual(terminal_proceeds(event,100.),0.)
        self.assertEqual(9000.+10*terminal_proceeds(event,100.),9000.)
        self.assertEqual(11000.-10*terminal_proceeds(event,100.),11000.)
        event.update(kind='delisting_return',value=-1.)
        self.assertEqual(terminal_proceeds(event,100.),0.)
        event.update(value=-.5)
        self.assertEqual(terminal_proceeds(event,100.),50.)
        event.update(kind='cash_per_share',value=150.)
        self.assertEqual(terminal_proceeds(event,100.),150.)
        for value in (-55.,-66.,-88.,-99.,float('nan')):
            payload=fixture();payload['terminals'][0].update(kind='delisting_return',value=value)
            with self.assertRaises(ValueError):validate(payload)

    def test_attestations_current_only_metadata_and_ambiguity_block(self):
        payload=fixture();payload['attestations']['membership_complete']['verified']=False
        self.assertIn('unverified_membership_complete',audit(payload)['issue_counts'])
        payload=fixture()
        for row in payload['membership']:row['available_time']=row['retrieved_time']
        self.assertIn('empty_eligible_universe',audit(payload)['issue_counts'])
        payload=fixture();row=deepcopy(payload['membership'][0]);row.update(event_id='conflict',member=False)
        payload['membership'].append(row)
        self.assertIn('ambiguous_membership',audit(payload)['issue_counts'])

    def test_immutable_import_reports_and_schema_validation(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);payload=fixture();path=put_history(payload,root/'data')
            self.assertEqual(load_history(path),payload)
            report(audit(payload),root/'report')
            self.assertTrue((root/'report/eligible-sessions.csv').exists())
            self.assertIn('Historical universe coverage',(root/'report/coverage.html').read_text())
            p=path/'history.json';original=p.read_bytes();p.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):load_history(path)
            p.write_bytes(original);value=json.loads(p.read_text());value['securities'][0]['issuer_id']='wrong'
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'checksum'):load_history(path)
        payload=fixture();payload['membership'].append(deepcopy(payload['membership'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):validate(payload)
        payload=fixture();duplicate=deepcopy(payload['prices'][0]);duplicate['bar']['symbol']='OTHER'
        payload['prices'].append(duplicate)
        with self.assertRaisesRegex(ValueError,'duplicate security/session'):validate(payload)
        payload=fixture();payload['prices'][0]['bar']['adjustments']='split_adjusted_with_dividends'
        with self.assertRaisesRegex(ValueError,'raw_with_actions'):validate(payload)


CSV = 'symbol,name,exchange,assetType,ipoDate,delistingDate,status\nOLD,Old Co,NYSE,Stock,2000-01-03,2020-01-07,Delisted\n'


class ListingCandidateTests(unittest.TestCase):
    def test_listing_candidates_are_never_silently_certified(self):
        rows=parse_listing_csv(CSV,'2020-01-08','delisted','2026-10-17T00:00:00+00:00')
        self.assertIsNone(rows[0]['stable_security_id'])
        self.assertFalse(rows[0]['terminal_return_verified'])
        self.assertEqual(rows[0]['available_time'],'2026-10-17T00:00:00+00:00')
        with self.assertRaisesRegex(ValueError,'future'):
            parse_listing_csv(CSV,'2027-01-08','delisted','2026-10-17T00:00:00+00:00')
        for text in ('{"Information":"rate limit"}',CSV+CSV.splitlines()[1]+'\n',''):
            with self.assertRaises(ValueError):parse_listing_csv(text,'2020-01-08','delisted','2026-10-17T00:00:00+00:00')

    def test_transport_persists_no_key_and_suppresses_errors(self):
        with TemporaryDirectory() as directory,patch.dict('os.environ',{'ALPHAVANTAGE_API_KEY':'private-key-value'}):
            path=fetch_listings('2020-01-08','delisted',directory,opener=lambda *a,**k:io.BytesIO(CSV.encode()))
            self.assertNotIn('private-key-value',(path/'listings.json').read_text())
            def fail(*args,**kwargs):raise OSError('URL contains private-key-value')
            with self.assertRaisesRegex(RuntimeError,'suppressed') as error:
                fetch_listings('2020-01-08','delisted',directory,opener=fail)
            self.assertNotIn('private-key-value',str(error.exception))
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'set ALPHAVANTAGE_API_KEY'):
                fetch_listings('2020-01-08','delisted','unused')
