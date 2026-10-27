"""Full fixed development cells; actual training is Colab-only and externally admitted."""
from copy import deepcopy
import hashlib
import importlib.metadata
import sys
from pathlib import Path

import torch

from rl_stocks.core.provenance import source_identity,repository_state
from rl_stocks.core.execution_provenance import execution_backend
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.runner import evaluate
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training.curves import render_curve
from rl_stocks.training.native_post_updates import post_update_checkpoints
from rl_stocks.training.recurrent_comparison import protocol,candidate_config,PROTOCOL_SHA256
from rl_stocks.training.recurrent_comparison_profile import validate_panel
from rl_stocks.training.recurrent_comparison_starts import load_start
from rl_stocks.training.recurrent_loop import run_colab_training,_atomic_json,_CheckpointHistory
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.sac_update_worker import preserved_rng


def expected_counts(arm,steps):
    if type(steps) is not int or not 0<=steps<=100000:
        raise ValueError('bounded actual transition counter required')
    if arm=='native_sac':
        updates=max(0,steps-1000)
        return updates,dict.fromkeys(('actor','critic','entropy'),updates)
    if arm=='recurrent_ppo_reference' and steps%1000==0:
        return steps//1000*10,dict(policy=steps//1000*200)
    raise ValueError('native registered arm and complete rollout boundary required')


class NativeAudit:
    """Bound actual completed Adam steps; never replace native optimization."""
    def __init__(self,model,arm):
        self.arm=arm
        self.optimizers=(dict(actor=model.actor.optimizer,critic=model.critic.optimizer,
                              entropy=model.ent_coef_optimizer) if arm=='native_sac'
                         else dict(policy=model.policy.optimizer))
        self.counts=dict.fromkeys(self.optimizers,0)
        self.maximum=expected_counts(arm,100000)[1]
        self.handles=[]
        for role,opt in self.optimizers.items():
            self.handles.append(opt.register_step_pre_hook(
                lambda opt,args,kwargs,role=role:self.before(role,opt)))
            self.handles.append(opt.register_step_post_hook(
                lambda opt,args,kwargs,role=role:self.after(role,opt)))
    def before(self,role,opt):
        parameters=[p for group in opt.param_groups for p in group['params']]
        if self.counts[role]>=self.maximum[role] or any(
                p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError('bounded native optimizer and finite gradients required')
    def after(self,role,opt):
        self.counts[role]+=1
        if any(not bool(torch.isfinite(p).all()) for g in opt.param_groups for p in g['params']):
            raise ValueError('nonfinite native parameter after optimizer step')
    def verify(self,model,steps):
        updates,counts=expected_counts(self.arm,steps)
        if model.num_timesteps!=steps or model._n_updates!=updates or self.counts!=counts:
            raise ValueError('actual post-update transition/epoch/Adam counters mismatch')
        for role,opt in self.optimizers.items():
            parameters=[p for group in opt.param_groups for p in group['params']]
            if counts[role]==0:
                if opt.state:
                    raise ValueError('initial native optimizer state must be empty')
            elif len(opt.state)!=len(parameters) or any(
                    opt.state[p]['step'].item()!=counts[role]
                    or any(opt.state[p][key].shape!=p.shape
                           or not bool(torch.isfinite(opt.state[p][key]).all())
                           for key in ('exp_avg','exp_avg_sq')) for p in parameters):
                raise ValueError('actual native Adam state counters mismatch')
    def close(self):
        for handle in self.handles:
            handle.remove()


def native_snapshot(model,arm,seed,steps,audit,output,lineage):
    audit.verify(model,steps)
    folder=Path(output)/('initial' if steps==0 else f'checkpoint-{steps}')
    folder.mkdir(exist_ok=False)
    model.save(folder/'model.zip')
    _atomic_json(folder/'manifest.json',dict(schema='recurrent-comparison-native-checkpoint-v1',
        arm=arm,seed=seed,transitions=steps,library_updates=model._n_updates,
        optimizer_steps=deepcopy(audit.counts),lineage=lineage,
        sha256=hashlib.sha256((folder/'model.zip').read_bytes()).hexdigest()))
    export_artifacts(Path(output).parent)
    return folder.name


def execute_colab(dataset,starts,output,*,arm,seed):
    require_colab_cpu()  # before model construction, RNG changes, output or gradients
    spec=protocol()
    if arm not in spec['arms'] or type(seed) is not int or seed not in spec['confirmation_seeds']:
        raise ValueError('registered full-development comparison role/seed required')
    normalizer,training=validate_panel(dataset)
    output=Path(output)
    if output.parent.name!='artifacts' or output.exists():
        raise ValueError('fresh owned comparison artifacts output required')
    torch.set_num_threads(1)
    identity=dict(source=source_identity(),dataset=spec['dataset'],
        observation_schema=spec['arm_contracts'][arm]['observation_profile'].replace('_','-')+'-v1-normalized21')
    metadata=dict(protocol_sha256=PROTOCOL_SHA256,arm=arm,seed=seed,
        transitions=100000,evaluation_every=10000,checkpoint_phase='post_update',
        initialization_manifest_sha256=hashlib.sha256((Path(starts)/'manifest.json').read_bytes()).hexdigest(),
        source_git=repository_state(Path.cwd()),lineage=identity,
        python=list(sys.version_info[:3]),backend=execution_backend('cpu'),
        versions={name:importlib.metadata.version(name) for name in
                  ('torch','numpy','gymnasium','stable-baselines3','sb3-contrib')},
        performance_admission_external=True,retired_test_access=False,
        exact_resume_supported=False)
    if arm=='gru_sac':
        learner=load_start(starts,arm,seed)
        try:
            return run_colab_training(dataset,output,candidate_config(seed),exposure_mode='unit_magnitude',
                                      lineage=identity,initial_learner=learner)
        finally:
            if output.is_dir():
                _atomic_json(output/'comparison.json',metadata)
                export_artifacts(output.parent)
    output.mkdir(parents=True)
    env=audit=model=None
    history=_CheckpointHistory(output)
    try:
        _atomic_json(output/'comparison.json',metadata)
        _atomic_json(output/'run.json',dict(status='running',**metadata))
        env=TradingEnv(training,normalizer,dataset['names'],random_start=True,episode_steps=512,
                       start_sampling='stock_etf_cycle_windows',
                       observation_profile=spec['arm_contracts'][arm]['observation_profile'])
        model=load_start(starts,arm,seed,env=env)
        audit=NativeAudit(model,arm)
        native_snapshot(model,arm,seed,0,audit,output,identity)
        latest={}
        def checkpoint(current,steps):
            name=native_snapshot(current,arm,seed,steps,audit,output,identity)
            with preserved_rng():
                results=evaluate(current,dataset,split='validation',
                    observation_profile=spec['arm_contracts'][arm]['observation_profile'],
                    exposure_mode='unit_magnitude')
            validate_development_report(results,dataset,env.config)
            latest['results']=results
            history.record(results,steps=steps,updates=current._n_updates,checkpoint=name,
                training=dict(native_library_counter=current._n_updates,optimizer_steps=deepcopy(audit.counts),
                    counter_kind='gradient_updates' if arm=='native_sac' else 'ppo_epochs'))
            _atomic_json(output/'training-execution.json',env.execution_statistics())
            _atomic_json(output/'training-sampling.json',env.sampling_statistics())
            render_curve(history.history,output/'learning-curve.html')
            export_artifacts(output.parent)
        with post_update_checkpoints(model,every=10000,checkpoint=checkpoint):
            model.learn(total_timesteps=100000)
        audit.verify(model,100000)
        if [r['steps'] for r in history.history]!=list(range(10000,100001,10000)):
            raise ValueError('ten actual post-update full-panel checkpoints required')
        history.finish(latest['results'],steps=100000,updates=model._n_updates)
        _atomic_json(output/'run.json',dict(status='complete',**metadata,actual_transitions=100000,
            actual_updates=model._n_updates,actual_optimizer_steps=audit.counts,
            selected_step=history.best_step,selected_score=history.best_score,
            final_score=history.history[-1]['score']))
        return history.history
    except BaseException as error:
        _atomic_json(output/'failure.json',dict(error_type=type(error).__name__,automatic_retry=False,
            actual_transitions=None if model is None else model.num_timesteps,
            optimizer_steps=None if audit is None else audit.counts))
        raise
    finally:
        try:
            if audit is not None:
                audit.close()
            if env is not None:
                env.close()
        finally:
            export_artifacts(output.parent)
