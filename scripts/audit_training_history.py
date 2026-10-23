"""Audit cached training dates only, without outcomes, inference or training."""
import argparse,json
from pathlib import Path
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.history_coverage import window_coverage


def audit(dataset_path, recent_start='2016-01-01'):
    before=checksum(dataset_path)
    data=json.loads(dataset_path.read_text())
    # Never inspect validation/test bar values. Require the retired test absent.
    if set(data['manifest']['included_splits'])!={'train','validation'}:
        raise ValueError('train/development-only dataset required')
    records=[]
    for episode in data['episodes'].values():
        if episode['split']!='train':continue
        sessions=[b['session'] for b in episode['bars']]
        recent=[s for s in sessions if s>=recent_start]
        records.append(dict(symbol=episode['symbol'],full=window_coverage(sessions),recent=window_coverage(recent)))
    if not records or len({r['symbol'] for r in records})!=len(records):
        raise ValueError('unique training symbols required')
    groups={}
    etfs={'SPY','QQQ','IWM','DIA'}
    for kind in ['stock','etf']:
        rows=[r for r in records if (r['symbol'] in etfs)==(kind=='etf')]
        if not rows:raise ValueError('both groups required')
        groups[kind]={}
        for arm in ['full','recent']:
            years=sorted({y for r in rows for y in r[arm]['expected_execution_year_fraction']})
            groups[kind][arm]=dict(symbols=len(rows),minimum_bars=min(r[arm]['bars'] for r in rows),
                expected_execution_year_fraction={y:sum(r[arm]['expected_execution_year_fraction'].get(y,0) for r in rows)/len(rows) for y in years})
    assert checksum(dataset_path)==before
    return dict(dataset_file_sha256=before,recent_start=recent_start,window_transitions=512,
        groups=groups,records=records,source_unchanged=True,training_performed=False,
        provider_called=False,test_evaluated=False,actual_year_visitation_measured=False,
        interpretation='Date-only availability and expected complete-window execution mass; not actual sampled-year counts, predictive evidence or approved experiment. Existing normalization unchanged; no dataset generated.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output required')
    result=audit(a.dataset);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['groups'],indent=2))


if __name__=='__main__':main()
