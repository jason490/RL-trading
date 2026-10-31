"""Fixed final fitted-period admission, never a performance promotion claim."""
import math
import re
from copy import deepcopy

from rl_stocks.training.task_conditioning import PROTOCOL_SHA256, protocol


def training_gate(cells):
    spec = protocol()
    if set(cells) != set(spec['roles']):
        raise ValueError('complete three-role final fitted matrix required')
    reference = cells[spec['candidate']]
    holding = None
    for cell in cells.values():
        if (cell['steps'] != spec['steps_per_cell'] or cell['period'] != 'training-window'
                or cell['phase'] != 'final' or cell['seed'] != spec['screen_seed']
                or cell['protocol_sha256'] != PROTOCOL_SHA256):
            raise ValueError('fixed complete final training cells required')
        if not re.fullmatch(r'[a-f0-9]{64}', cell['holding_path_sha256']):
            raise ValueError('holding path digest required')
        metrics = cell['metrics']
        values = [metrics['score'], metrics['groups']['stock']['excess_return'],
                  metrics['groups']['etf']['excess_return'],
                  metrics['groups']['stock']['median_excess_return']]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
            raise ValueError('finite fitted metrics required')
        if (metrics['groups']['stock']['count'] != 22
                or metrics['groups']['etf']['count'] != 4):
            raise ValueError('original22-stock/four-ETF panel required')
        if holding is None:
            holding = cell['holding_path_sha256']
        elif holding != cell['holding_path_sha256']:
            raise ValueError('matched holding paths required')
    limits = spec['training_admission']
    gains = {role: 100 * (reference['metrics']['score'] - cell['metrics']['score'])
             for role, cell in cells.items() if role != spec['candidate']}
    groups = reference['metrics']['groups']
    floors = {'stock': 100 * groups['stock']['excess_return'],
              'etf': 100 * groups['etf']['excess_return'],
              'median_stock': 100 * groups['stock']['median_excess_return']}
    passed = (min(gains.values()) >= limits['minimum_gain_pp_vs_each_control']
              and min(floors.values()) >= limits['minimum_stock_etf_and_median_stock_excess_pp'])
    return {'schema': 'task-training-admission-v1', 'passed': passed,
            'development_allowed': passed, 'candidate': spec['candidate'],
            'gain_pp_vs_controls': gains, 'excess_pp': floors,
            'holding_path_sha256': holding, 'inputs': deepcopy(cells),
            'objective_achieved': False, 'performance_promotion': False,
            'automatic_extension_allowed': False}
