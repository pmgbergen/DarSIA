"""Approved preparation-photo metadata follows aliases; no RAW decoder runs."""
from copy import deepcopy
import os

import pytest
import toml

from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower import painted_adoption_table as subject
from darsia.fluidflower.storage_paths import refresh_storage_paths


def test_actual_preparation_photo_check_accepts_only_explicit_raw_relocation(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    series, old = tmp_path / "DarSIA", tmp_path / "old photographs"
    series.mkdir()
    old.mkdir()
    raw = old / "sample.ARW"
    raw.write_bytes(b"synthetic RAW identity; never decoded")
    stamp = painting.image_stamp(raw)
    raw_stat = raw.stat()
    workspace = series / "current.json"
    workspace.write_text('{"revision_id":"retained"}', encoding="utf-8")
    parameters = series / "camera.json"
    parameters.write_text('{"profile":"retained"}', encoding="utf-8")
    basis = dict(format=subject.FRAME_BASIS_FORMAT, basis="explicit-current-frame-at-preparation",
        historical_frame_bound=False, workspace_revision=subject._bound(workspace),
        photograph=subject._bound(raw), image_stamp=stamp,
        camera_profile=dict(kind="camera_u", parameters=subject._bound(parameters),
            profile_id="camera", camera_wh=[3, 2]),
        source_identity_basis="current-file-at-preparation; saved-stat-stamp-matched")
    binding = dict(record=dict(kind="manual map"), frame_basis=basis,
        approval=dict(sha256="a" * 64, approved_utc="2026-01-01T00:00:00Z"))
    session = dict(image=dict(path=str(raw), stamp=stamp, size=[3, 2]))
    original_binding, original_session = deepcopy(binding), deepcopy(session)
    subject._verify_frame_basis(binding, session)

    moved = tmp_path / "Raw_data"
    old.rename(moved)
    (series / "series.toml").write_text(toml.dumps(dict(schema_version=1, series_id="sample",
        storage=dict(path_aliases={str(old): str(moved)}))), encoding="utf-8")
    refresh_storage_paths(series)
    subject._verify_frame_basis(binding, session)
    assert binding == original_binding and session == original_session
    assert painting.image_stamp(moved / raw.name) == stamp
    assert (moved / raw.name).read_bytes() == b"synthetic RAW identity; never decoded"

    other = tmp_path / "other.ARW"
    other.write_bytes((moved / raw.name).read_bytes())
    os.utime(other, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns))
    assert painting.image_stamp(other) == stamp
    with pytest.raises(ValueError, match="preparation photograph and saved stat stamp"):
        subject._verify_frame_basis(binding, dict(image=dict(session["image"], path=str(other))))
    changed_digest = deepcopy(binding)
    changed_digest["frame_basis"]["photograph"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        subject._verify_frame_basis(changed_digest, session)
    os.utime(moved / raw.name, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match="preparation photograph and saved stat stamp"):
        subject._verify_frame_basis(binding, session)
    assert binding == original_binding and session == original_session
