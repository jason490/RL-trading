"""Treasury releases preserve initialization evidence and distinct method labels."""

import importlib.util
import json
import tarfile

import pytest


@pytest.mark.parametrize("arm", ["legacy", "masked", "full"])
def test_treasury_identity_and_initializer_are_published_without_private_data(tmp_path, arm):
    spec = importlib.util.spec_from_file_location("treasury_packager", "scripts/package_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "input"
    run = root / "runs" / arm
    bundle = root / "frozen/id"
    run.mkdir(parents=True)
    bundle.mkdir(parents=True)
    metadata = dict(
        status="complete",
        bundle_id="id",
        algorithm="recurrent_ppo",
        research_variant="native",
        treasury_arm=arm,
        seed=11,
        actual_steps=100000,
        smoke=False,
    )
    (run / "metadata.json").write_text(json.dumps(metadata))
    for name in ("best.zip", "latest.zip", "treasury-initialization.npz"):
        (run / name).write_bytes(name.encode())
    (run / "replay.pkl").write_bytes(b"private replay")
    (run / ".env").write_bytes(b"private credentials")
    for name in ("model.zip", "manifest.json", "checksums.json"):
        (bundle / name).write_bytes(b"{}")
    results = {
        "fixture": dict(
            rows=[dict(session="2021-01-04", equity=100000, price=987654)],
            baselines=dict(buy_hold=dict(rows=[dict(equity=100000)])),
            metrics=dict(net_return=0),
        )
    }
    (run / "selected-validation.json").write_text(json.dumps(results))
    output = tmp_path / "public"
    archive = module.package([root], output)
    with tarfile.open(archive) as stream:
        names = stream.getnames()
        initializer = stream.extractfile(f"runs/{arm}/treasury-initialization.npz").read()
    assert initializer == (run / "treasury-initialization.npz").read_bytes()
    assert not any("replay" in n or ".env" in n or "selected-validation" in n for n in names)
    assert arm in (output / "validation.html").read_text()
    assert arm in (output / "validation" / (arm + ".html")).read_text()
    assert "987654" not in (output / "validation" / (arm + ".html")).read_text()
    assert json.loads((output / "model-catalog.json").read_text())[0]["treasury_arm"] == arm
