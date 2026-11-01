"""Fixed training-only checkpoint selection; never grants development access."""
from copy import deepcopy
import math
import re

from rl_stocks.training.task_conditioning import protocol


def select_training_checkpoint(history):
    spec = protocol()
    expected = spec['selection']['checkpoints']
    if (not isinstance(history, list) or len(history) != len(expected)
            or [point.get('steps') for point in history] != expected):
        raise ValueError('complete ordered fixed training checkpoints required')
    holding = None
    best = None
    for point in history:
        score = point.get('score')
        if (point.get('phase') != 'training-window'
                or type(score) not in (int, float) or not math.isfinite(score)
                or not re.fullmatch(r'[a-f0-9]{64}', point.get('holding_path_sha256', ''))):
            raise ValueError('finite training-only matched holding scores required')
        if holding is not None and point['holding_path_sha256'] != holding:
            raise ValueError('checkpoint holding paths must match')
        holding = point['holding_path_sha256']
        if best is None or score > best['score']:
            best = point
    return {'selected': deepcopy(best), 'final': deepcopy(history[-1]),
            'selection_period': 'training-window', 'development_allowed': False,
            'selection_rule': 'earliest strict maximum; final100k retained separately'}
