"""Untrained actual-view Gymnasium/SB3 checker acceptance; no optimization."""
import argparse
import hashlib
from pathlib import Path

from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset, normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.training.recurrent_environment_spec import protocol, validate_view, PROTOCOL_SHA256
from rl_stocks.training.recurrent_environment_worker import INPUT_ID
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training.recurrent_sac_smoke_worker import network_hash, INITIAL_RECURRENT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', required=True, type=Path)
    args = parser.parse_args()
    if args.proof.exists():
        raise FileExistsError('checker proof already exists')
    root = Path(__file__).resolve().parents[1]
    input_path = root/'data/datasets'/INPUT_ID
    paths = [input_path/'dataset.json',root/'.runtime/usage-ledger.json',root/'.runtime/crossq-integrity-stop.json']
    def hashes():
        return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    payload = read_dataset(input_path)
    validate_view(payload)
    train = [ep for ep in payload['episodes'].values() if ep['split']=='train']
    spec = protocol()
    with preserved_rng():
        for checker in (gym_check, sb3_check):
            env = TradingEnv(train,normalizer_from(payload['normalizer']),payload['names'],
                             random_start=True,episode_steps=spec['loop']['episode_steps'],
                             start_sampling=spec['loop']['start_sampling'],observation_profile='execution_full')
            try:
                checker(env)
            finally:
                env.close()
        import torch
        torch.manual_seed(spec['loop']['seed'])
        networks = RecurrentSACNetworks(21, spec['loop']['hidden_size'])
        assert network_hash(networks)==INITIAL_RECURRENT_SHA256
        assert all(p.grad is None for p in networks.parameters())
    assert before==hashes()
    result = dict(schema='recurrent-environment-untrained-acceptance-v1',protocol_sha256=PROTOCOL_SHA256,
                  input_identity=INPUT_ID,gymnasium_checker_passed=True,sb3_checker_passed=True,
                  training_episodes=len(train),observation_shape=[21],action_shape=[4],
                  initial_network_sha256=INITIAL_RECURRENT_SHA256,protected_sha256=before,
                  protected_files_unchanged=True,optimizer_steps=0,backward_calls=0,
                  colab_allocations=0,provider_requests=0,performance_evaluation=False,
                  implementation_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                      ['scripts/check_recurrent_environment_view.py','src/rl_stocks/training/recurrent_environment_worker.py']})
    write_once(args.proof,canonical(result))
    print('Actual-view Gymnasium/SB3 checkers and pinned untrained tensors passed; no optimization/allocation.')


if __name__=='__main__':
    main()
