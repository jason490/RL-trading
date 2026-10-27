"""Lossless publication, failure atomicity and checkpoint-before-report recovery."""
import tarfile

import pytest

from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training import recurrent_comparison_profile as profile
from rl_stocks.training.recurrent_comparison_starts import construct
from rl_stocks.training.sac_update_worker import preserved_rng


def test_fast_archive_has_exact_contents_and_failure_preserves_old_target(tmp_path,monkeypatch):
    root=tmp_path/'artifacts'
    root.mkdir()
    payload=b'important checkpoint bytes'*10000
    (root/'state.bin').write_bytes(payload)
    target=tmp_path/'checkpoint.tar.gz'
    monkeypatch.setenv('RL_CHECKPOINT_ARCHIVE',str(target))
    export_artifacts(root)
    with tarfile.open(target) as archive:
        assert archive.extractfile('artifacts/state.bin').read()==payload
    previous=target.read_bytes()
    def broken(*args,**kwargs):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(tarfile.TarFile,'add',broken)
    with pytest.raises(OSError,match='disk failure'):
        export_artifacts(root)
    assert target.read_bytes()==previous and not (tmp_path/'checkpoint.tar.gz.tmp').exists()


@pytest.mark.parametrize('level',[True,0,10,'1'])
def test_bad_compression_level_rejects_before_publication(tmp_path,level):
    with pytest.raises(ValueError):
        export_artifacts(tmp_path,compresslevel=level)


def test_untrained_native_checkpoint_can_be_published_without_report(tmp_path,monkeypatch):
    output=tmp_path/'artifacts'/'profile'
    output.mkdir(parents=True)
    target=tmp_path/'checkpoint.tar.gz'
    monkeypatch.setenv('RL_CHECKPOINT_ARCHIVE',str(target))
    with preserved_rng():
        model=construct('native_sac',11)
        profile.publish_profile_checkpoint('native_sac',model,output,dict(native_sac_updates=0))
    with tarfile.open(target) as archive:
        assert {'artifacts/profile/native_sac.zip','artifacts/profile/partial-profile.json'} <= set(archive.getnames())
    assert model._n_updates==0
