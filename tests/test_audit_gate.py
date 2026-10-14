import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class AuditGateTests(unittest.TestCase):
    def test_only_committed_validation_selection_can_open_audit(self):
        spec=importlib.util.spec_from_file_location('audit_selected','scripts/audit_selected.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'decision.json'
            value={'test_evaluated':False};path.write_text(json.dumps(value))
            with patch.object(module.subprocess,'run',return_value=SimpleNamespace(stdout=path.read_bytes())):
                self.assertEqual(module.committed_decision(path,root),value)
            with patch.object(module.subprocess,'run',return_value=SimpleNamespace(stdout=b'{}')):
                with self.assertRaises(ValueError):module.committed_decision(path,root)
            value['test_evaluated']=True;path.write_text(json.dumps(value))
            with patch.object(module.subprocess,'run',return_value=SimpleNamespace(stdout=path.read_bytes())):
                with self.assertRaises(ValueError):module.committed_decision(path,root)
