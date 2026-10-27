"""Colab entry for the fixed fresh-collection integration smoke."""
from pathlib import Path
from rl_stocks.evaluation.recurrent_sac import evaluate_recurrent_sac
from rl_stocks.evaluation.runner import selection_score
from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.training.recurrent_environment_spec import protocol, validate_view, PROTOCOL_SHA256
from rl_stocks.training.recurrent_loop import LoopConfig, run_colab_training
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.checkpoint_export import export_artifacts

INPUT_ID = '1e25385c771013865a2c05ec9b31af9b1e7399aadaa482562ac0535478a92906'


def execute_colab(payload, output):
    require_colab_cpu()
    validate_view(payload)
    if digest(payload) != INPUT_ID:
        raise ValueError('exact accepted cached smoke view required')
    spec = protocol()
    lineage = dict(source='recurrent-environment-smoke-'+PROTOCOL_SHA256, dataset=INPUT_ID,
                   observation_schema='execution-full-v1-normalized21')
    history = run_colab_training(payload, output, LoopConfig(**spec['loop']),
                                 exposure_mode=spec['exposure_mode'], lineage=lineage)
    (Path(output)/'dataset.json').write_bytes(canonical(payload))
    for point in history:
        learner, _ = load_checkpoint(Path(output)/point['checkpoint'], expected_lineage=lineage)
        results = evaluate_recurrent_sac(learner.networks, payload, exposure_mode=spec['exposure_mode'])
        if selection_score(results) != point['score']:
            raise ValueError('same-runtime checkpoint evaluation did not reproduce its curve')
        (Path(output)/f"validation-{point['steps']}.json").write_bytes(canonical(results))
    export_artifacts(Path(output).parent)
    return history
