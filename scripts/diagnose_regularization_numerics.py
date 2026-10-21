"""Fixed saved-model numerical/forward audit; no optimizer updates or Colab."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from rl_stocks.inference.bundle import checksum


def weight_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.policy.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def tensor_counts(tensors):
    counts=dict(elements=0,nonfinite=0,zero=0,subnormal=0)
    for t in tensors:
        if not t.is_floating_point():continue
        counts['elements']+=t.numel()
        counts['nonfinite']+=int((~torch.isfinite(t)).sum())
        counts['zero']+=int((t==0).sum())
        counts['subnormal']+=int(((t.abs()>0)&(t.abs()<torch.finfo(t.dtype).tiny)).sum())
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('new output required')
    source=Path('results/sac-regularization-2026-10-20/interruption-summary.json')
    summary=json.loads(source.read_text());protected={str(source):checksum(source)}
    if summary['status']!='interrupted_no_promotion' or not summary['cleanup_verified']:
        raise ValueError('verified recovered interruption required')
    torch.set_num_threads(1)
    observations=torch.from_numpy(np.random.default_rng(11).standard_normal((256,21)).astype(np.float32))
    records={}
    for arm,cell in summary['records'].items():
        filename='latest.zip' if arm=='control' else 'latest-checkpoint.zip'
        name=next(p for p in summary['source_sha256'] if Path(p).name==filename and cell['run'] in Path(p).parts)
        path=Path(name);protected[name]=checksum(path)
        if protected[name]!=summary['source_sha256'][name]:raise ValueError('retained model checksum mismatch')
        model=SAC.load(path,device='cpu');model.policy.set_training_mode(False)
        before=weight_hash(model);updates=model._n_updates
        torch.set_flush_denormal(False)
        counts={'policy':tensor_counts(model.policy.state_dict().values())}
        for label,optimizer in [('actor_moments',model.actor.optimizer),('critic_moments',model.critic.optimizer)]:
            counts[label]=tensor_counts([v for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')])
        times={};outputs={}
        with torch.inference_mode():
            for flush in (False,True):
                supported=torch.set_flush_denormal(flush)
                if not supported:raise ValueError('CPU flush-denormal mode unavailable')
                def forward():
                    action=model.actor(observations,deterministic=True)
                    return (action,*model.critic(observations,action),*model.critic_target(observations,action))
                for _ in range(10):forward()
                measurements=[]
                for _ in range(3):
                    start=time.perf_counter()
                    for _ in range(100):forward()
                    measurements.append((time.perf_counter()-start)/100)
                times[str(flush)]=dict(seconds_per_forward=median(measurements),blocks=measurements)
                outputs[str(flush)]=[v.detach().numpy().copy() for v in forward()]
        torch.set_flush_denormal(False)
        exact=all(np.array_equal(a,b) for a,b in zip(outputs['False'],outputs['True']))
        delta=max(float(np.max(np.abs(a-b))) for a,b in zip(outputs['False'],outputs['True']))
        if weight_hash(model)!=before or model._n_updates!=updates:raise ValueError('weights/update count mutated')
        curve=cell['curve'];intervals=[]
        previous_step=0;previous_elapsed=0.
        for point in curve:
            intervals.append(dict(steps=point['steps'],phase=point['phase'],transitions=point['steps']-previous_step,
                                  elapsed_seconds=point['elapsed_seconds']-previous_elapsed))
            previous_step=point['steps'];previous_elapsed=point['elapsed_seconds']
        records[arm]=dict(run=cell['run'],restored_steps=model.num_timesteps,counts=counts,
            forward_timings=times,forward_speedup_flush_on=times['False']['seconds_per_forward']/times['True']['seconds_per_forward'],
            outputs_exact_on_fixed_synthetic_batch=exact,max_abs_output_difference=delta,
            weights_sha256=before,weights_updates_unchanged=True,logged_checkpoint_intervals=intervals)
    if not all(checksum(Path(p))==h for p,h in protected.items()):raise ValueError('source files changed')
    result=dict(records=records,source_sha256=protected,script_sha256=checksum(Path(__file__)),torch_version=torch.__version__,
        local_cpu_threads=1,batch_shape=[256,21],warmup_forwards=10,measurement_forwards_per_block=100,blocks_per_mode=3,
        source_files_unchanged=True,training_performed=False,optimizer_steps=0,backward_calls=0,
        colab_allocated=False,test_evaluated=False,numerical_setting_applied_to_training=False,
        interpretation='Isolated local inference arithmetic experiment on synthetic observations, not a Colab training benchmark or performance improvement. Logged intervals include learning/evaluation/export; differing remote hosts confound cross-arm timings. No model/optimizer state exported or training configuration changed. Incomplete study remains unassessable.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:{k:r[k] for k in ('counts','forward_speedup_flush_on','outputs_exact_on_fixed_synthetic_batch','max_abs_output_difference')} for a,r in records.items()},indent=2))


if __name__=='__main__':main()
