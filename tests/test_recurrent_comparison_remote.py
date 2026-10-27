"""Remote dispatch fixture replaces computation; no local training."""
import importlib.util
import json
import os
from pathlib import Path

import torch

from rl_stocks.core.contracts import digest
from rl_stocks.runtime.colab.jobs import recurrent_comparison_profile_plan
from rl_stocks.training import recurrent_comparison_profile as worker


def test_remote_profile_dispatch_preserves_integrity_stop(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('remote_profile',Path(__file__).parents[1]/'scripts/remote_train.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'root',tmp_path)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    for key in ('UV_CACHE_DIR','MPLCONFIGDIR','MPLBACKEND','PYTHONPATH','RL_CHECKPOINT_ARCHIVE'):
        monkeypatch.setenv(key,os.environ.get(key,''))
    payload={'fixture':'read-only dispatch; no market data'}
    plan=recurrent_comparison_profile_plan()
    plan['dataset_contract']['identity']=digest(payload)
    import rl_stocks.runtime.colab.jobs as jobs
    monkeypatch.setattr(jobs,'validate_plan',lambda plan:None)  # fixed-plan validation tested separately
    (tmp_path/'dataset.json').write_text(json.dumps(payload))
    (tmp_path/'job-plan.json').write_text(json.dumps(plan))
    (tmp_path/'source-git.json').write_text(json.dumps(dict(revision='a'*40,dirty='')))
    (tmp_path/'artifacts').mkdir()
    stop=tmp_path/'crossq-integrity-stop.json'
    stop.write_text('retained')
    called=[]
    monkeypatch.setattr(worker,'execute_colab',lambda data,starts,output:called.append((data,starts,output)))
    module.main()
    assert called==[(payload,tmp_path/'comparison-starts',tmp_path/'artifacts/comparison-profile')]
    completion=json.loads((tmp_path/'artifacts/completion.json').read_text())
    assert completion['dataset']==digest(payload) and completion['plan']==plan
    assert completion['gpu'] is None and stop.read_text()=='retained'
