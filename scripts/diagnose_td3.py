"""Audit three owned baseline TD3 models and saved validation paths, never datasets."""
import argparse
import json
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from rl_stocks.core.contracts import digest
from rl_stocks.environment.observations.state import EXECUTION_FIELDS
from rl_stocks.inference.bundle import FrozenPolicy, checksum
from rl_stocks.reporting.offpolicy import saved_episode

SOURCE = Path('artifacts/colab/rl-stocks-a100-fcf79fe3ffb9/retrieved/artifacts')
RUNS = {11: '20261014T232818-td3-11-ff94e1b5',
        23: '20261014T233608-td3-23-45042c92',
        37: '20261014T234502-td3-37-140a76d1'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('new output required')
    universe = Path('configs/universes/sector_research.json')
    instruments = json.loads(universe.read_text())['instruments']
    stocks = {i['symbol'] for i in instruments if i['kind'] == 'stock'}
    symbols = {i['symbol'] for i in instruments if i['role'] == 'tradable'}
    protected = {str(universe): checksum(universe)}
    records = []; baseline = contract = None
    torch.set_num_threads(1)
    for seed, name in RUNS.items():
        run = SOURCE / 'runs' / name
        path = run / 'metadata.json'
        protected[str(path)] = checksum(path)
        m = json.loads(path.read_text())
        if (m['algorithm'] != 'td3' or m['seed'] != seed or m['actual_steps'] != 50000
                or m['smoke'] or m['status'] != 'complete'):
            raise ValueError('fixed completed baseline TD3 run required')
        current = {k: m[k] for k in ('dataset', 'source_hash', 'git', 'simulation', 'versions', 'resolved_parameters')}
        if contract is not None and current != contract:
            raise ValueError('matched original source contracts required')
        contract = current
        folder = SOURCE / 'frozen' / m['bundle_id']
        manifest = json.loads((folder / 'manifest.json').read_text())
        for f in folder.iterdir():
            protected[str(f)] = checksum(f)
        frozen = FrozenPolicy(folder, manifest['feature_schema'], manifest['data_manifest']['provenance']['capabilities']['feed'])
        models = {}
        for filename in ('best.zip', 'latest.zip'):
            path = run / filename; protected[str(path)] = checksum(path)
            model = TD3.load(path, device='cpu')
            noise = model.action_noise
            if (not isinstance(noise, NormalActionNoise)
                    or not np.array_equal(noise._mu, np.zeros(4))
                    or not np.array_equal(noise._sigma, np.full(4, .1))):
                raise ValueError('verified four-action Gaussian noise required')
            if tuple(model.observation_space.shape) != (frozen.observation_size,):
                raise ValueError('saved input dimensions disagree')
            models[filename] = dict(input_size=frozen.observation_size, n_updates=model._n_updates,
                                    policy_delay=model.policy_delay, target_policy_noise=model.target_policy_noise,
                                    target_noise_clip=model.target_noise_clip,
                                    exploration_noise=dict(kind='NormalActionNoise', mean=noise._mu.tolist(), sigma=noise._sigma.tolist()))
        phases = {}
        for phase, filename in [('selected', 'selected-validation.json'), ('final', 'validation.json')]:
            path = run / filename; protected[str(path)] = checksum(path)
            results = json.loads(path.read_text())
            if {v['symbol'] for v in results.values()} != symbols:
                raise ValueError('configured validation panel required')
            bh = digest({k: v['baselines']['buy_hold'] for k, v in results.items()})
            if baseline is not None and bh != baseline:
                raise ValueError('identical original holding paths required')
            baseline = bh
            rows = [saved_episode(v) for v in results.values()]
            keys = ('net_return', 'buy_hold_return', 'short_fraction', 'long_fraction',
                    'incremental_cost_drag_pp', 'saved_path_noncost_excess_pp')
            phases[phase] = {kind: {k: mean(r[k] for r in rows if (r['symbol'] in stocks) == (kind == 'stock'))
                                   for k in keys} for kind in ('stock', 'etf')}
        records.append(dict(seed=seed, run=name, bundle_id=m['bundle_id'], models=models,
                            observation_profile=frozen.observation_profile,
                            omitted_execution_fields=list(EXECUTION_FIELDS) if frozen.observation_profile == 'legacy' else [],
                            phases=phases, directional_pnl_reconstructed=False))
    if not all(checksum(Path(p)) == h for p, h in protected.items()):
        raise ValueError('source files mutated')
    result = dict(records=records, matched_source_contract=contract, validation_baseline_sha256=baseline,
                  source_hashes=protected, source_unchanged=True,
                  diagnostic_script_sha256=checksum(Path(__file__)), dataset_opened=False,
                  training_performed=False, test_evaluated=False, colab_allocated=False,
                  objective_achieved=False,
                  interpretation='Own saved models only; no fitting, predictions or dataset loading. Input omission is verified, not its performance effect. Noise is verified from restored objects, not absent metadata. Old studies are diagnosis, not matched controls for new training. Close short fraction does not establish profitable shorts.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Verified three TD3 seeds, six saved models and six development reports; no training/data access/Colab.')


if __name__ == '__main__':
    main()
