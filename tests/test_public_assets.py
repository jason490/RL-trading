import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


class PublicAssetTests(unittest.TestCase):
    def test_only_allowlisted_models_and_derived_reports_are_packaged(self):
        spec=importlib.util.spec_from_file_location('package_research','scripts/package_research.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'input';run=root/'runs/run';bundle=root/'frozen/id'
            run.mkdir(parents=True);bundle.mkdir(parents=True)
            (run/'metadata.json').write_text(json.dumps({'status':'complete','bundle_id':'id'}))
            (run/'best.zip').write_bytes(b'weights')
            (run/'replay.pkl').write_bytes(b'private features')
            (root/'bootstrap-log.txt').write_text('private runtime output')
            (bundle/'model.zip').write_bytes(b'weights')
            (bundle/'manifest.json').write_text('{}')
            (bundle/'checksums.json').write_text('{}')
            result={'rows':[{'session':'2020-01-02','equity':100001,'price':987654}],
                    'baselines':{'buy_hold':{'rows':[{'equity':100002}]}},'metrics':{'net_return':.01}}
            (run/'selected-validation.json').write_text(json.dumps({'fixture':result}))
            output=Path(directory)/'public'
            path=module.package([root],output)
            with tarfile.open(path) as archive:
                names=archive.getnames()
            self.assertFalse(any('replay' in n or 'bootstrap' in n for n in names))
            self.assertNotIn('987654',(output/'validation.html').read_text())
            (bundle/'unexpected.env').write_text('private')
            with self.assertRaises(ValueError):module.package([root],output)
