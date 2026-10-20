"""Describe saved internal-probe forecasts without fitting or changing gates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rl_stocks.reporting.forecast_errors import decompose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('use a new output path')
    before=hashlib.sha256(a.predictions.read_bytes()).hexdigest();records=[]
    with np.load(a.predictions,allow_pickle=False) as z:
        expected={'target','sessions','symbols','constant','ridge_masked','ridge_full',*[f'{arm}-{seed}' for arm in ('masked','full') for seed in (11,23,37)]}
        if set(z.files)!=expected:raise ValueError('complete fixed forecast set required')
        y=z['target'];days=z['sessions'];symbols=z['symbols']
        if (len(y)!=18616 or len(set(symbols))!=26 or len(set(zip(days,symbols)))!=len(y)
            or set(str(d)[:4] for d in days)!={'2018','2019','2020'}):raise ValueError('internal probe only')
        for key in sorted(expected-{'target','sessions','symbols'}):
            for year in ('all','2018','2019','2020'):
                take=np.ones(len(y),dtype=bool) if year=='all' else np.char.startswith(days,year)
                records.append(dict(predictor=key,year=year,**decompose(y[take],z[key][take],days[take])))
    assert hashlib.sha256(a.predictions.read_bytes()).hexdigest()==before
    result=dict(source_prediction_sha256=before,source_unchanged=True,training_performed=False,
        correction_applied=False,checkpoint_selection=False,trading_profitability_established=False,
        scope='Internal training-era probe; negative target signs are reporting only, not profitable shorts.',records=records)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'{len(records)} descriptive decompositions; source unchanged, no fitting.')


if __name__=='__main__':main()
