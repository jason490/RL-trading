"""Explicit fitted-period diagnostics, separate from selection/evaluation splits."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.runner import _evaluate

PROTOCOL_SHA256 = '0ea7aa3d793a6f0469234883dba2ddef07b135427989f4eb728e094c0b696196'


def protocol():
    p = json.loads((Path(__file__).resolve().parents[3] / 'configs/experiments/training_fit_diagnostic.json').read_text())
    if digest(p) != PROTOCOL_SHA256:
        raise ValueError('training diagnostic protocol changed')
    return p


def training_window(dataset, *, original_panel_only=False):
    p = protocol()
    if type(original_panel_only) is not bool:raise ValueError('explicit original-panel flag required')
    if (dataset['manifest']['included_splits'] != ['train', 'validation']
            or dataset['manifest']['unverified_override'] is not False
            or any(e['split'] not in ('train', 'validation') for e in dataset['episodes'].values())):
        raise ValueError('verified training/development-only source required')
    expected = set(dataset['manifest']['evaluation_symbols'])
    episodes = {}
    for key, episode in dataset['episodes'].items():
        if episode['split'] != 'train':
            continue
        bars = episode['bars']
        if (len(bars) != len(episode['features']) or len(bars) != len(episode['atr'])
                or [b['session'] for b in bars] != sorted({b['session'] for b in bars})
                or any(b['session'] > '2020-12-31' for b in bars)):
            raise ValueError('aligned chronological training rows required')
        if original_panel_only and episode['symbol'] not in expected:continue
        indices = [i for i,b in enumerate(bars) if p['anchor_session'] <= b['session'] <= p['last_session']]
        if (len(indices) < 2 or bars[indices[0]]['session'] != p['anchor_session']
                or bars[indices[1]]['session'] != p['first_execution_session']
                or bars[indices[-1]]['session'] != p['last_session']):
            raise ValueError('complete fixed training window required')
        episodes[key] = {**episode, **{k: [episode[k][i] for i in indices] for k in ('bars','features','atr')}}
    if set(e['symbol'] for e in episodes.values()) != expected or len(episodes) != len(expected):
        raise ValueError('one training window per declared symbol required')
    return {**dataset, 'episodes': episodes}


def evaluate_training_window(model, window, config, observation_profile):
    p = protocol()
    # Revalidate even a caller-provided view; preserve explicit train labels.
    view = training_window(window)
    if observation_profile != p['observation_profile']:
        raise ValueError('fixed execution observation profile required')
    return _evaluate(model, view, 'train', config, observation_profile)


def evaluate_common_recurrent_training_window(model, window, config, exposure_mode):
    """Original declared panel, masked memory inputs and explicit execution rule."""
    from rl_stocks.rl.registry.prediction import recurrent
    if not recurrent(model):raise ValueError('saved recurrent policy required')
    if exposure_mode not in ('unrestricted','unit_magnitude'):
        raise ValueError('declared training-diagnostic exposure mode required')
    view=training_window(window,original_panel_only=True)
    return _evaluate(model,view,'train',config,'execution_masked',exposure_mode)


def common_training_views(datasets):
    """Require exact common model inputs and accounting bars across paired sources."""
    if not isinstance(datasets,dict) or set(datasets)!={'full','recent'}:
        raise ValueError('both full/recent sources required')
    views={a:training_window(data,original_panel_only=True) for a,data in datasets.items()}
    reference=views['full']
    for view in views.values():
        if any(digest(view[key])!=digest(reference[key]) for key in ('episodes','normalizer','names','schema')):
            raise ValueError('exact common training window and preprocessing required')
    return views
