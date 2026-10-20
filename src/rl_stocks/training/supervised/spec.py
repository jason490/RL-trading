"""Frozen feasibility contracts; changing one requires a new reviewed protocol."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.training.supervised.targets import FIT_CUTOFF,PROBE_CUTOFF
from rl_stocks.data.validation.bars import timestamp

SIGNAL_DATASET='84b1e38843f8e623c3abbee3d1373c4e20556b39b85891ba8201c1ff54723235'


def protocol():
    path=Path(__file__).resolve().parents[4]/'configs/experiments/supervised_signal.json'
    value=json.loads(path.read_text())
    if digest(value)!=PROTOCOL_SHA256:raise ValueError('supervised protocol drift')
    return value


def validate_payload(data):
    from rl_stocks.features.registry.assemble import NAMES as BASE
    from rl_stocks.features.trend.context import NAMES as EXTRA
    if (digest(data)!=SIGNAL_DATASET or data.get('schema')!='completed-training-signal-v1'
        or data['names']!=list(BASE)+list(EXTRA)):raise ValueError('signal dataset identity/schema mismatch')
    for r in data['rows']:
        available=timestamp(r['label_available_time'])
        if (r['fold']=='fit' and available>=FIT_CUTOFF or r['fold']=='probe' and (r['decision_session']<'2018-01-01' or available>=PROBE_CUTOFF)
            or r['fold'] not in ('fit','probe')):raise ValueError('signal fold leakage')
    return data

# Filled from the committed numeric protocol before any fitting.
PROTOCOL_SHA256='54d032fbfb152d03f1a1eac6f6026dc76a37645ad75df1c06b3191c43b016a44'
