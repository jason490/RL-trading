"""Verify actual retained smoke inputs and untrained starts; never execute updates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.files import write_once
from rl_stocks.training.recurrent_sac_smoke import PROTOCOL_SHA256, prepare_smoke, protocol
from rl_stocks.training.shared_initialization import tensor_fingerprint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preparation evidence already exists')
    root = Path(__file__).resolve().parents[1]
    spec = protocol()
    input_path = root/'data/datasets'/spec['input_identity']/'dataset.json'
    protected = [input_path, root/'.runtime/usage-ledger.json',
                 root/'.runtime/crossq-integrity-stop.json',
                 *sorted((root/'configs/initializations/crossq_v1').glob('*'))]
    protected = [p for p in protected if p.is_file()]
    before = {str(p.relative_to(root)): sha(p) for p in protected}
    rng = torch.get_rng_state().clone()
    prepared = prepare_smoke(json.loads(input_path.read_text()))
    try:
        model = prepared.learner
        assert model.updates == model.attempts == 0 and not model.failed
        assert all(not opt.state for opt in model.optimizers.values())
        assert prepared.control._n_updates == 0
        assert all(torch.equal(a, b) for a, b in zip(model.networks.critics.parameters(),
                                                     model.networks.targets.parameters(), strict=True))
        assert torch.equal(rng, torch.get_rng_state())
        after = {str(p.relative_to(root)): sha(p) for p in protected}
        assert before == after
        sources = ['configs/experiments/recurrent_sac_smoke.json',
                   'src/rl_stocks/training/recurrent_sac_smoke.py',
                   'src/rl_stocks/rl/recurrent_sac/networks.py',
                   'src/rl_stocks/training/recurrent_replay.py',
                   'src/rl_stocks/training/recurrent_payload.py',
                   'scripts/prepare_recurrent_sac_smoke.py']
        result = dict(schema='recurrent-sac-smoke-preparation-v1', protocol_sha256=PROTOCOL_SHA256,
                      input_identity=spec['input_identity'], replay_rows=1001,
                      done_rows=np.flatnonzero(prepared.arrays['dones'][:, 0]).tolist(),
                      timeout_rows=np.flatnonzero(prepared.arrays['timeouts'][:, 0]).tolist(),
                      loss_rows=[r.tolist() for r in prepared.loss_rows],
                      batches=len(prepared.batches), loss_rows_per_batch=128,
                      initial_network_sha256=tensor_fingerprint({k: v.detach().numpy() for k, v
                                                               in model.networks.state_dict().items()}),
                      initial_control_policy_sha256=prepared.control_initial['policy_tensor_sha256'],
                      recurrent_parameter_counts=dict(actor=sum(p.numel() for p in model.networks.actor.parameters()),
                          critics=sum(p.numel() for p in model.networks.critics.parameters()),
                          targets=sum(p.numel() for p in model.networks.targets.parameters())),
                      empty_optimizers=True, exact_initial_targets=True,
                      caller_torch_rng_unchanged=True, protected_files_unchanged=True,
                      protected_sha256=before, implementation_sha256={s: sha(root/s) for s in sources},
                      model_gradient_updates=0, optimizer_steps=0, colab_allocations=0,
                      performance_evaluation=False, crossq_stop_cleared=False)
        write_once(args.output, (json.dumps(result, indent=2)+'\n').encode())
        print('Actual untrained smoke preparation verified; no optimization or allocation.')
    finally:
        prepared.control.get_env().close()


if __name__ == '__main__':
    main()
