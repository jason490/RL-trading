"""Verify completed development cells and atomically publish a compact report; never trains."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from rl_stocks.core.contracts import canonical
from rl_stocks.evaluation.recurrent_comparison_acceptance import accept_matrix,validate_manifest
from rl_stocks.reporting.recurrent_comparison_report import render_comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--starts',type=Path,required=True)
    parser.add_argument('--screen-receipt',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.matrix.read_text())
    validate_manifest(manifest)
    output=args.output.resolve()
    inputs=[args.starts.resolve(),args.dataset.resolve().parent]
    inputs += [(args.matrix.parent/entry['folder']).resolve() for entry in manifest.get('cells',[])]
    if output.exists() or any(output==p or output.is_relative_to(p) for p in inputs):
        parser.error('fresh output outside all inputs required')
    prior=None if args.screen_receipt is None else json.loads(args.screen_receipt.read_text())
    receipt=accept_matrix(manifest,json.loads(args.dataset.read_text()),args.starts,
                          root=args.matrix.parent,screen_receipt=prior)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.comparison-report-',dir=output.parent))
    try:
        (temporary/'acceptance.json').write_bytes(canonical(receipt))
        render_comparison(receipt,temporary/'comparison.html')
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(dict(stage=receipt['stage'],gate_passed=receipt['gate']['passed'],
                         output=str(output),objective_achieved=False),sort_keys=True))


if __name__=='__main__':
    main()
