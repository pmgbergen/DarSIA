"""The native training page's evidence gate on RAW photographic reviews (package D2e).

Reviewed evidence on a RAW review root passes the gate through the review's
SHA-bound native records instead of V5 sidecars; pixels are never read here.
"""
import hashlib
import json
from types import SimpleNamespace

import pytest

from .test_raw_photographic_review import _write, inputs  # noqa: F401


@pytest.fixture
def photometry(tmp_path, monkeypatch):
    series = tmp_path / "configured_series"
    series.mkdir()
    (series / "series.toml").write_text('schema_version = 1\nseries_id = "ffac"\n[paths]\n', encoding="utf8")
    monkeypatch.setenv("DARSIA_FFAC_PROJECT_ROOT", str(series))
    from darsia.fluidflower import native_training_photometry
    return native_training_photometry


def _bundle(tmp_path, root, manifest_sha):
    decisions = tmp_path / "decisions.json"
    combined = tmp_path / "combined.json"
    decisions.write_text(json.dumps({"decisions": {"ac52": {"annotations": [{"id": "a1"}]}}}), encoding="utf-8")
    combined.write_text(json.dumps({"jobs": [{"run": "ac52", "review_output_dir": str(root),
                                              "native_manifest_sha256": manifest_sha}]}), encoding="utf-8")
    return [(decisions, combined)]


def test_raw_review_evidence_passes_through_its_native_records(tmp_path, inputs, photometry):
    manifest = _write(tmp_path, inputs)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result = photometry.require_saved_sidecars(_bundle(tmp_path, tmp_path / "review", digest),
                                               SimpleNamespace(settings={}))
    assert result["native_manifests"] == 1 and result["declared_existing_sidecars"] == 6  # B and two frames.
    assert result["payload_sha_verified_here"] is False
    with pytest.raises(ValueError, match="SHA changed"):
        photometry.require_saved_sidecars(_bundle(tmp_path, tmp_path / "review", "0" * 64),
                                          SimpleNamespace(settings={}))


def test_a_missing_raw_input_closes_the_gate(tmp_path, inputs, photometry):
    manifest = _write(tmp_path, inputs)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    frame = json.loads(manifest.read_text(encoding="utf-8"))["frames"][1]
    log = json.loads(open(frame["native_record"]["path"], encoding="utf-8").read())["full"]["native_centre"]
    __import__("os").remove(log["valid_mask"]["path"])
    with pytest.raises(FileNotFoundError, match="RAW training input is missing"):
        photometry.require_saved_sidecars(_bundle(tmp_path, tmp_path / "review", digest),
                                          SimpleNamespace(settings={}))
    assert photometry._raw_review_inputs.__doc__.strip()
