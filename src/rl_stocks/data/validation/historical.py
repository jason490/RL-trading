"""Source-consistency and chronological regime audit, never model selection."""
from collections import defaultdict
from datetime import date
import math
from rl_stocks.data.validation.bars import validate


def audit_history(bars, raw_parts, symbols, expected_sessions):
    validate(bars)
    expected=set(expected_sessions);symbols=tuple(symbols)
    if not expected or not symbols or len(set(symbols))!=len(symbols):
        raise ValueError('nonempty unique symbols and expected sessions required')
    if {b.symbol for b in bars}!=set(symbols):raise ValueError('declared symbol panel incomplete')
    raw={}
    for part in raw_parts:
        for row in part['rows']:
            key=(part['symbol'],str(row['date'])[:10])
            date.fromisoformat(key[1])
            if key in raw:raise ValueError('duplicate raw source row')
            raw[key]=row
    normalized={(b.symbol,b.session):b for b in bars}
    if set(raw)!=set(normalized):raise ValueError('raw and normalized row sets differ')
    records=[]
    for symbol in symbols:
        rows=[b for b in bars if b.symbol==symbol]
        if {b.session for b in rows}!=expected:raise ValueError('incomplete or extra exchange sessions')
        splits=[];dividends=[];by_year=defaultdict(lambda:dict(rows=0,ready63=0,negative63=0,ready252=0,negative252=0,joint_negative=0))
        for i,b in enumerate(rows):
            if (b.provider,b.feed,b.adjustments)!=('yahoo','yahoo','split_adjusted_with_dividends') or b.split!=1:
                raise ValueError('require Yahoo split-adjusted units without repeated split')
            r=raw[(symbol,b.session)]
            for name,key in [('open','Open'),('high','High'),('low','Low'),('close','Close'),('volume','Volume'),('dividend','Dividends')]:
                value=float(r[key])
                if not math.isfinite(value) or value!=getattr(b,name):raise ValueError('source field mismatch')
            split=float(r['Stock Splits']);capital=float(r.get('Capital Gains',0))
            if not math.isfinite(split) or split<0 or not math.isfinite(capital) or capital!=0 or b.dividend<0:
                raise ValueError('unsupported or invalid corporate action')
            if split:splits.append(dict(session=b.session,source_factor=split,normalized_factor=b.split))
            if b.dividend:dividends.append(dict(session=b.session,cash_per_adjusted_share=b.dividend))
            v=by_year[b.session[:4]];v['rows']+=1
            for horizon in (63,252):
                if i>=horizon:
                    v[f'ready{horizon}']+=1;v[f'negative{horizon}']+=int(b.close<rows[i-horizon].close)
            if i>=252:v['joint_negative']+=int(b.close<rows[i-63].close and b.close<rows[i-252].close)
        records.append(dict(symbol=symbol,rows=len(rows),first=rows[0].session,last=rows[-1].session,
            source_splits=splits,dividends=dividends,regime_years=dict(by_year)))
    return dict(accepted_source_consistency=True,coverage_complete=True,independent_action_verification=False,
        point_in_time_vintages_verified=False,stock_delisting_coverage_verified=False,
        training_performed=False,automatic_merge=False,records=records)
