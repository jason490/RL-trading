"""Manual checkpoint/real-simulator fixtures; forbid local differentiation/fitting."""
import json

import pytest
import torch

from test_recurrent_environment_spec import source_fixture
from test_recurrent_sac_smoke_verify import no_fitting, manual_state  # noqa: F401
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.recurrent_sac import evaluate_recurrent_sac
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training import recurrent_environment_spec as spec, recurrent_environment_worker as worker
from rl_stocks.training import recurrent_environment_verify as verifier
from rl_stocks.training.curves import render_curve
from rl_stocks.training.recurrent_collector import RecurrentCollector
from rl_stocks.training.recurrent_loop import _CheckpointHistory
from rl_stocks.training.recurrent_payload import RecurrentPayloadStore
from rl_stocks.training.recurrent_sac_checkpoint import save_checkpoint
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner
from rl_stocks.training.sac_update_worker import preserved_rng


def directory_fixture(root, monkeypatch):
    source = source_fixture(monkeypatch)
    view = spec.build_view(source)
    monkeypatch.setattr(worker,'INPUT_ID',digest(view))
    monkeypatch.setattr(verifier,'protocol',spec.protocol)
    root.mkdir(parents=True)
    with preserved_rng():
        torch.manual_seed(11)
        networks = RecurrentSACNetworks(21,64)
        learner = RecurrentSACLearner(networks,max_updates=4)
        lineage = dict(source='recurrent-environment-smoke-'+spec.PROTOCOL_SHA256,
                       dataset=worker.INPUT_ID,observation_schema='execution-full-v1-normalized21')
        save_checkpoint(learner,root/'initial',lineage=lineage)
        env = TradingEnv([ep for ep in view['episodes'].values() if ep['split']=='train'],
                         normalizer_from(view['normalizer']),view['names'],random_start=True,
                         episode_steps=64,start_sampling='stock_etf_cycle_windows',observation_profile='execution_full')
        collector = RecurrentCollector(env,networks,RecurrentPayloadStore(256,21),seed=11,warmup_steps=64,
                                        max_episode_steps=64)
        history = _CheckpointHistory(root)
        try:
            for step in (64,128):
                for _ in range(64):
                    collector.collect(policy_revision=0)  # untrained collection only
                if step==128:
                    with torch.no_grad():
                        networks.actor.mean.bias.add_(.001)
                        for p in networks.critics.parameters():
                            p.add_(.001)
                        learner.log_alpha.fill_(-.001)
                    learner.updates=learner.attempts=4
                    learner.completed_steps=dict.fromkeys(learner.optimizers,4)
                    for optimizer in learner.optimizers.values():
                        manual_state(optimizer)
                name=f'checkpoint-{step}'
                save_checkpoint(learner,root/name,lineage=lineage)
                results=evaluate_recurrent_sac(networks,view,exposure_mode='unit_magnitude')
                history.record(results,steps=step,updates=learner.updates,
                    training={} if step==64 else dict(update=4,order=['entropy','critic','actor','targets']),checkpoint=name)
                (root/f'validation-{step}.json').write_bytes(canonical(results))
            history.finish(results,steps=128,updates=4)
            render_curve(history.history,root/'learning-curve.html')
            (root/'dataset.json').write_bytes(canonical(view))
            (root/'training-execution.json').write_bytes(canonical(env.execution_statistics()))
            (root/'training-sampling.json').write_bytes(canonical(env.sampling_statistics()))
            (root/'run.json').write_bytes(canonical(dict(status='complete',config=spec.protocol()['loop'],
                lineage=lineage,exposure_mode='unit_magnitude',exact_resume_supported=False,
                actual_transitions=128,actual_updates=4,selected_step=history.best_step,
                selected_score=history.best_score,final_score=history.history[-1]['score'])))
        finally:
            env.close()
    return view


def test_physical_checkpoint_report_curve_and_economic_replay(tmp_path,monkeypatch):
    directory_fixture(tmp_path/'run',monkeypatch)
    result=verifier.verify_directory(tmp_path/'run')
    assert result['saved_artifacts_verified'] and result['recorded_action_economics_verified']
    assert result['selection_final_curve_verified'] and result['frozen_carried_prediction_parity']
    assert result['actual_updates']==4  # manual metadata fixture, not actual training


@pytest.mark.parametrize('change',['curve','pointer','economics','counter','csv','run'])
def test_bad_physical_artifacts_reject(tmp_path,monkeypatch,change):
    root=tmp_path/'run'
    directory_fixture(root,monkeypatch)
    if change=='curve':
        path=root/'learning-curve.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['change_pp']+=1
        path.write_text('\n'.join(json.dumps(v) for v in rows)+'\n')
    elif change=='csv':
        path=root/'learning-curve.csv'
        path.write_text(path.read_text().replace('post_update','forged'))
    else:
        path=root/{'pointer':'final.json','economics':'validation-128.json',
                   'counter':'checkpoint-128/manifest.json','run':'run.json'}[change]
        value=json.loads(path.read_text())
        if change=='pointer':
            value['steps']=64
        elif change=='economics':
            next(iter(value.values()))['rows'][0]['costs']+=1
        elif change=='counter':
            value['updates']=3
        else:
            value['actual_transitions']=129
        path.write_bytes(canonical(value))
    with pytest.raises(ValueError):
        verifier.verify_directory(root)
