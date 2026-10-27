"""A diagnostic must preserve retained evidence and never recurse into its output."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('archive_benchmark',Path(__file__).parents[1]/'scripts/benchmark_artifact_export.py')
benchmark=importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_read_only_round_trip_and_temporary_cleanup(tmp_path):
    root=tmp_path/'artifacts'
    root.mkdir()
    payload=b'retained scientific bytes'*100
    (root/'checkpoint').write_bytes(payload)
    result=benchmark.benchmark(root,tmp_path)
    assert result['retained_files_unchanged'] and result['temporary_archives_removed']
    assert all(s['exact_member_content_verified'] for s in result['samples'])
    assert (root/'checkpoint').read_bytes()==payload
    assert set(tmp_path.iterdir())=={root}
    with pytest.raises(ValueError,match='outside retained'):
        benchmark.benchmark(root,root)
    assert set(root.iterdir())=={root/'checkpoint'}
