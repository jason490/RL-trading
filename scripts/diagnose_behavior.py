"""Validation-only behavior audit; never opens the completed test audit."""
import argparse,json
from pathlib import Path


def diagnose(roots,output):
    import torch
    from rl_stocks.evaluation.behavior import summarize_behavior
    from stable_baselines3 import SAC
    torch.set_num_threads(1);records=[]
    for root in map(Path,roots):
        for path in (root/'runs').glob('*/metadata.json'):
            metadata=json.loads(path.read_text())
            if metadata.get('smoke') or metadata['algorithm']!='sac':continue
            results_path=path.parent/'selected-validation.json'
            if not results_path.exists():continue
            value=summarize_behavior(json.loads(results_path.read_text()))
            model=SAC.load(path.parent/'best.zip',device='cpu')
            entropy=float(model.log_ent_coef.detach().exp().cpu()) if model.log_ent_coef is not None else float(model.ent_coef_tensor)
            records.append({'run':path.parent.name,'dataset':metadata['dataset'],'seed':metadata['seed'],
                            'device':metadata['device'],'entropy_coefficient':entropy,**value})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'split':'validation','records':records},indent=2,allow_nan=False)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('roots',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();print(diagnose(args.roots,args.output))
