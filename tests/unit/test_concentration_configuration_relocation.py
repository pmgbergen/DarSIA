"""Concentration metadata locators follow explicit shared moves; no arrays read."""

import json

import pytest
import toml

from darsia.fluidflower import concentration_configuration as subject
from darsia.fluidflower.series_paths import register_series
from darsia.fluidflower.storage_paths import refresh_storage_paths


@pytest.mark.parametrize("absolute_artifacts", [False, True])
def test_default_profile_and_original_artifact_paths_follow_physical_moves(
        tmp_path, monkeypatch, absolute_artifacts):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    root = tmp_path / "project" / "DarSIA"
    root.mkdir(parents=True)
    manifest = root / "series.toml"
    settings = dict(schema_version=1, series_id="sample", name="Sample")
    manifest.write_text(toml.dumps(settings), encoding="utf-8")
    register_series(root)
    old_profile = root / "concentration.toml"
    old_calibration = root / "Calibration"
    (old_calibration / "fitted").mkdir(parents=True)
    receipt = old_calibration / "receipt.json"
    receipt.write_text(json.dumps(dict(dataset="sample", run="r1", artifacts=[])), encoding="utf-8")
    (root / "protocols").mkdir()
    for name in ("injection.csv", "pressure.csv"):
        (root / "protocols" / name).write_bytes(b"metadata only\n")

    def authored(path):
        return str(path) if absolute_artifacts else path.relative_to(root).as_posix()

    profile = dict(schema_version=1, series_id="sample", runs=dict(r1=dict(enabled=True,
        artifact_record=authored(receipt), calibration_folder=authored(old_calibration / "fitted"),
        injection_protocol="protocols/injection.csv", pressure_temperature_protocol="protocols/pressure.csv")))
    old_profile.write_text(toml.dumps(profile), encoding="utf-8")
    profile_bytes, receipt_bytes = old_profile.read_bytes(), receipt.read_bytes()
    job = dict(dataset="sample", run="r1")
    before = subject.describe_run(job)
    assert before["ready"]

    moved_profile = root / "config" / "concentration.toml"
    moved_profile.parent.mkdir()
    old_profile.rename(moved_profile)
    moved_calibration = root.parent / "Results" / "calibration"
    moved_calibration.parent.mkdir()
    old_calibration.rename(moved_calibration)
    settings["storage"] = dict(path_aliases={str(old_profile): str(moved_profile),
        str(old_calibration): str(moved_calibration)})
    manifest.write_text(toml.dumps(settings), encoding="utf-8")
    refresh_storage_paths(root)

    for request in (job, dict(job, concentration_profile_path=str(old_profile))):
        description = subject.describe_run(request)
        assert description == dict(before, profile_path=str(moved_profile))
        assert subject._profile(request)[4] == profile
    assert subject._path(profile["runs"]["r1"]["artifact_record"], root) == moved_calibration / "receipt.json"
    assert moved_profile.read_bytes() == profile_bytes
    assert subject._path(profile["runs"]["r1"]["artifact_record"], root).read_bytes() == receipt_bytes

    # An alias may locate the profile, but cannot transfer it to another series.
    outside = tmp_path / "another-series" / "concentration.toml"
    outside.parent.mkdir()
    moved_profile.rename(outside)
    settings["storage"]["path_aliases"][str(old_profile)] = str(outside)
    manifest.write_text(toml.dumps(settings), encoding="utf-8")
    refresh_storage_paths(root)
    with pytest.raises(ValueError, match="belong to the selected series"):
        subject.describe_run(job)
