from datetime import date
import sys
import pytest

from rl_stocks.core.contracts import Request
from rl_stocks.data.generation.datasets import generate, read_dataset
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.storage.immutable import put


def test_explicit_universe_scope_retains_source_failures_and_gates_context(tmp_path):
    request=Request(('SPY','QQQ'),date(2020,1,1),date(2023,6,1))
    bars=SyntheticProvider().bars(request)
    good={'missing':[],'calendar_verified':True}
    coverage={'SPY':dict(good),'QQQ':dict(good), 'MISSING':{'missing':['2020-01-02'],'calendar_verified':False}}
    universe={'instruments':[{'symbol':'SPY','role':'tradable'}, {'symbol':'QQQ','role':'context'}]}
    options={'universe':universe,'include_test':False}
    snapshot=put(tmp_path/'raw',bars,{'coverage':coverage,'dividends_verified':True})
    with pytest.raises(ValueError,match='coverage/corporate'):
        generate(snapshot,tmp_path/'datasets',**options)
    path=generate(snapshot,tmp_path/'datasets',coverage_scope='universe',**options)
    data=read_dataset(path)
    assert data['manifest']['coverage_audit']['excluded_failed_coverage']==['MISSING']
    assert data['manifest']['provenance']['coverage']['MISSING']['missing']==['2020-01-02']
    assert data['manifest']['unverified_override'] is False
    assert {e['split'] for e in data['episodes'].values()}=={'train','validation'}
    for bad_coverage,verified in [({**coverage,'QQQ':coverage['MISSING']},True),(coverage,False)]:
        bad=put(tmp_path/'raw',bars,{'coverage':bad_coverage,'dividends_verified':verified})
        with pytest.raises(ValueError,match='coverage/corporate'):
            generate(bad,tmp_path/'datasets',coverage_scope='universe',**options)
    absent=put(tmp_path/'raw',bars,{'coverage':{'SPY':good},'dividends_verified':True})
    with pytest.raises(ValueError,match='coverage entries absent'):
        generate(absent,tmp_path/'datasets',coverage_scope='universe',**options)
    no_context=put(tmp_path/'raw',[b for b in bars if b.symbol=='SPY'],
                   {'coverage':coverage,'dividends_verified':True})
    with pytest.raises(ValueError,match='universe absent'):
        generate(no_context,tmp_path/'datasets',coverage_scope='universe',**options)
    for bad_options in ({'coverage_scope':'universe'}, {'coverage_scope':'unknown','universe':universe}):
        with pytest.raises(ValueError,match='coverage scope'):
            generate(snapshot,tmp_path/'datasets',include_test=False,**bad_options)
    verified=put(tmp_path/'raw',bars,{'coverage':{'SPY':good,'QQQ':good},'dividends_verified':True})
    assert generate(verified,tmp_path/'datasets',**options)==generate(
        verified,tmp_path/'datasets',coverage_scope='snapshot',**options)


def test_complete_case_universe_declares_all_three_exclusions():
    from rl_stocks.data.universe.config import read_universe, symbols
    complete=read_universe('configs/universes/stock_research_107_complete.json')
    requested=read_universe('configs/universes/stock_research_110.json')
    assert len([i for i in complete['instruments'] if i['kind']=='stock'])==107
    assert set(symbols(requested))-set(symbols(complete))=={'EA','AVB','EQR'}
    assert {e['symbol'] for e in complete['exclusions']}=={'EA','AVB','EQR'}


def test_development_cli_forwards_explicit_scope_and_common_panel(monkeypatch,capsys):
    from rl_stocks.runtime.local import main
    observed={}
    def fake(snapshot,output,weekly,**options):
        observed.update(options);return 'prepared'
    monkeypatch.setattr('rl_stocks.data.generation.datasets.generate',fake)
    monkeypatch.setattr(sys,'argv',['rl-stocks','dataset','unused','--universe',
        'configs/universes/sector_research.json','--coverage-scope','universe',
        '--development-only','--evaluation-symbols','SPY','QQQ'])
    main()
    assert observed['coverage_scope']=='universe'
    assert observed['include_test'] is False
    assert observed['evaluation_symbols']==['SPY','QQQ']
    assert not observed['allow_unverified']
    assert capsys.readouterr().out.strip()=='prepared'
