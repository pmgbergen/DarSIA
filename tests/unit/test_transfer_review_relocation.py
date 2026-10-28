"""Restore retained cohort/review/approval metadata after explicit storage moves."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_adoption_table as files
from darsia.fluidflower import painting_chart_transfer_reader as charts
from darsia.fluidflower import painting_series_use_ui as status
from darsia.fluidflower.painting_series_adoption_ui import SeriesAdoptionController, _version_identity
from darsia.fluidflower.storage_paths import (
    refresh_storage_paths, same_storage_descriptor, same_storage_path,
)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return files._bound(path)


def test_retained_39_run_transfer_findings_and_approvals_restore_after_root_and_photo_moves(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    series = tmp_path / "DarSIA"
    series.mkdir()
    old, raw = series / "geometry_facies", tmp_path / "old photographs"
    raw.mkdir()
    photos = {}
    names = [f"run{number:02d}" for number in range(1, 40)]
    for run in names:
        path = raw / (run + ".png")
        path.write_bytes(("retained photograph identity " + run).encode())
        photos[run] = files._bound(path)
    index = _save(old / "frames/index.json", dict(runs=names, point_bindings=photos))
    approval = dict(version="saved-map", sha256="a" * 64, approved_utc="2026-01-01T00:00:00Z")
    version = dict(folder=str(old / "annotations/layer_painting/versions/run01/saved-map"),
        sha256="a" * 64, name="Saved map", kind="manual map", run_id=names[0], files={}, approval=approval)
    rows = []
    for run in names:
        row = dict(run_id=run, gap=dict(pixels=1), findings=[], reference_image=photos[run])
        component = dict(component_id="1" * 64, identity=dict(sha256="1" * 64), pixels=1)
        component["review_row_sha256"] = gaps._digest(dict(context=gaps._digest(row), component=component))
        row["components"] = [component]
        row["review_row_sha256"] = gaps._row_digest(row)
        rows.append(row)
    review = dict(format=gaps.REVIEW_FORMAT, meaning=gaps.MEANING, map_version=version,
        geometry_index=index, source=dict(reference_run=names[0], point_bindings=photos), runs=rows, issues=[])
    transfer_folder = old / "annotations/map_placement/retained"
    review_path = transfer_folder / "native_gap_review/gap_review.json"
    review_binding = _save(review_path, review)
    retained_approval = dict(format=gaps.APPROVAL_FORMAT, meaning=gaps.MEANING, review=review_binding,
        runs=[dict(run_id=row["run_id"], component_id=row["components"][0]["component_id"],
            review_row_sha256=row["components"][0]["review_row_sha256"], actor="test reviewer",
            approved_utc="2026-01-01T00:00:00Z", reason="Retained synthetic review",
            decision_ref="retained", classification="rig_or_frame", other_text="") for row in rows])
    approval_path = transfer_folder / "native_gap_review/approval.json"
    _save(approval_path, retained_approval)
    transfer = dict(format="darsia-fluidflower-map-placement-v1", created_utc="2026-01-01T00:00:00Z",
        map_version=version, reference_run=names[0], geometry_index=index, gap_review=review_binding,
        runs=[dict(run_id=run, status="placed", point_binding=photos[run]) for run in names])
    _save(transfer_folder / "index.json", transfer)
    frozen = {path.relative_to(old): path.read_bytes() for path in old.rglob("*") if path.is_file()}
    before = gaps.read_gap_review(review_path)
    assert gaps.approval_records(retained_approval, before) == retained_approval

    moved, moved_photos = tmp_path / "Results/setup/Geometry", tmp_path / "Raw_data"
    moved.parent.mkdir(parents=True)
    old.rename(moved)
    raw.rename(moved_photos)
    (series / "series.toml").write_text(toml.dumps(dict(schema_version=1, series_id="sample",
        storage=dict(path_aliases={str(old): str(moved), str(raw): str(moved_photos)}))), encoding="utf-8")
    refresh_storage_paths(series)
    selected_version = dict(version, folder=str(moved / Path(version["folder"]).relative_to(old)))
    progress = status.existing_progress(old / "annotations", old / "overrides/geometry_facies/current.json", selected_version)
    assert progress["errors"] == []
    assert [row["run_id"] for row in progress["transfer"]["runs"]] == names
    assert progress["transfer"]["map_version"]["approval"] == approval
    loaded = gaps.read_gap_review(review_path)
    physical = gaps.read_gap_review(moved / review_path.relative_to(old))
    assert [row["run_id"] for row in physical["runs"]] == names
    assert physical["source"]["point_bindings"] == photos
    assert gaps.approval_records(retained_approval, physical) == retained_approval
    assert same_storage_descriptor(loaded["review_binding"], physical["review_binding"])
    copied = tmp_path / "same-bytes-another-review.json"
    copied.write_bytes((moved / review_path.relative_to(old)).read_bytes())
    with pytest.raises(ValueError, match="another gap review"):
        gaps.approval_records(retained_approval, gaps.read_gap_review(copied))
    for run in names:
        assert same_storage_path(photos[run]["path"], moved_photos / (run + ".png"))
        assert same_storage_descriptor(photos[run], files._bound(moved_photos / (run + ".png")))

    # Use the actual known-review opener's reader; no window, RAW or raster is opened.
    view = SimpleNamespace(project=SimpleNamespace(root=moved), _selected_version=lambda: selected_version,
        series_use_status=SimpleNamespace(result=progress))
    controller = SimpleNamespace(view=view, current_gap_review=lambda: None, _registered_gap_reviews=[],
        open_gap_review=lambda source, review_reader: review_reader(source))
    restored = SeriesAdoptionController.open_latest_gap_review(controller)
    assert [row["run_id"] for row in restored["runs"]] == names
    selection = SimpleNamespace(index_path=str(moved / "frames/index.json"),
        sha256=index["sha256"], size_bytes=index["size_bytes"])
    reviewed = SimpleNamespace(_current=lambda: True, context=dict(version=_version_identity(selected_version),
        approval=approval, selection=selection), view=view, _findings_context_snapshot=lambda: {},
        gap_dialog=None, findings_changed=SimpleNamespace(emit=lambda: None))
    SeriesAdoptionController._review_context(reviewed, restored)
    assert reviewed._findings_review_binding == restored["review_binding"]
    monkeypatch.setattr(charts, "ChartTransferSession", lambda source, **options: source)
    restored_chart = charts.selected_chart_session(old, selected_version, selection, progress["gap_reviews"])
    assert [row["run_id"] for row in restored_chart["runs"]] == names
    assert {name: (moved / name).read_bytes() for name in frozen} == frozen

    altered = deepcopy(review_binding)
    altered["sha256"] = "0" * 64
    assert not same_storage_descriptor(physical["review_binding"], altered)
    assert not same_storage_descriptor(review_binding, dict(review_binding, size_bytes=float(review_binding["size_bytes"])))
    with pytest.raises(ValueError, match="changed|identity|source"):
        gaps.read_gap_review(dict(loaded, review_binding=altered))
