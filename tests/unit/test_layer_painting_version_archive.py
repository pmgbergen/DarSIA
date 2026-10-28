"""Synthetic temporary packages only: reversible version actions and drawings."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import layer_painting_version_archive as versions


def state():
    session = lp.PaintingSession(run_id="sample", image_size=(12, 8))
    session.layers = [lp.PaintedLayer(1, "Sand", "#888888", "Sand")]
    session.polygons = [lp.Polygon(1, [(0., 0.), (0., 12.), (8., 12.), (8., 0.)])]
    proposal = lp.LayerProposal(np.ones((8, 12), np.uint8), np.full((8, 12), 255, np.uint8), (12, 8), "source")
    return session, proposal


def make_version(tmp_path, kind="proposal"):
    project = SimpleNamespace(root=tmp_path)
    session, proposal = state()
    if kind == "manual":
        value = manual.ManualMap(proposal.classes.copy(), np.ones((8, 12), bool),
                                  np.zeros((8, 12), bool), (12, 8), 0.)
        record = manual.save_manual_version(project, session, value, name="First")
    elif kind == "drawing":
        record = versions.save_drawing_version(project, session, name="First")
    else:
        record = lp.save_version(project, session, proposal, [], name="First")
    return project, record


def payloads(folder):
    return {path.name: path.read_bytes() for path in Path(folder).iterdir() if path.is_file()}


@pytest.mark.parametrize("kind", ["proposal", "manual", "drawing"])
def test_archive_and_restore_preserve_every_original_byte_and_old_reader_behavior(tmp_path, kind):
    project, record = make_version(tmp_path, kind)
    before = payloads(record["folder"])
    archived = versions.archive_version(project, "sample", record)
    assert not Path(record["folder"]).exists()
    assert payloads(archived["folder"]) == before
    assert lp.list_versions(project, "sample") == []
    assert len(versions.list_version_metadata(project, "sample", deleted=True)) == 1
    events = lp.read_map_approvals(project, "sample")
    assert all(row["format"] == lp.LAYER_APPROVAL_FORMAT for row in events)
    assert all("sha256" not in row and "approved_utc" not in row for row in events)
    assert lp.map_approval(events, record["sha256"]) is None
    restored = versions.restore_deleted_version(project, "sample", archived)
    assert restored["folder"] == record["folder"] and payloads(record["folder"]) == before
    assert len(lp.list_versions(project, "sample")) == 1
    assert versions.list_version_metadata(project, "sample", deleted=True) == []


@pytest.mark.parametrize("kind", ["proposal", "manual"])
def test_actual_approval_refuses_archive_even_when_selection_claims_unapproved(tmp_path, kind):
    project, record = make_version(tmp_path, kind)
    approval = (manual.approve_manual_version(record) if kind == "manual" else lp.approve_version(record))
    before = payloads(record["folder"])
    selected = dict(record, approved=False)
    with pytest.raises(ValueError, match="approved"):
        versions.archive_version(project, "sample", selected)
    assert payloads(record["folder"]) == before
    assert lp.map_approval(lp.read_map_approvals(project, "sample"), record["sha256"]) == approval


@pytest.mark.parametrize("reference", ["placement", "preparation", "current", "history", "revision", "explicit"])
def test_placement_and_adoption_proofs_refuse_archive_without_relying_on_approvals(tmp_path, reference):
    project, record = make_version(tmp_path)
    value = dict(map_version={"sha256": record["sha256"]})
    paths = dict(placement=tmp_path / "annotations/map_placement/one/index.json",
        preparation=tmp_path / "annotations/map_adoption/one/index.json",
        current=tmp_path / "overrides/geometry_facies/current.json",
        history=tmp_path / "overrides/geometry_facies/revisions.jsonl",
        revision=tmp_path / "results/geometry/one/revision.json", explicit=tmp_path / "external-index.json")
    path = paths[reference]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf8")
    extra = [path] if reference == "explicit" else []
    with pytest.raises(ValueError, match="placed|adoption|adopted"):
        versions.archive_version(project, "sample", record, reference_files=extra)
    assert Path(record["folder"]).is_dir()


def test_rename_and_notes_preserve_content_sha_and_original_approval_bytes(tmp_path):
    project, record = make_version(tmp_path)
    lp.approve_version(record)
    before = payloads(record["folder"])
    log = lp.approvals_path(project, "sample")
    prefix = log.read_bytes()
    versions.rename_version(project, "sample", record, "New label")
    versions.set_version_note(project, "sample", record, "Needs a closer look")
    row = versions.list_version_metadata(project, "sample")[0]
    assert row["name"] == "First" and row["display_name"] == "New label"
    assert row["note"] == "Needs a closer look" and row["approved"] is True
    assert row["sha256"] == record["sha256"] and payloads(record["folder"]) == before
    assert log.read_bytes().startswith(prefix)
    assert lp.map_approval(lp.read_map_approvals(project, "sample"), record["sha256"])["name"] == "First"


def test_restore_never_overwrites_an_existing_version_folder(tmp_path):
    project, record = make_version(tmp_path)
    archived = versions.archive_version(project, "sample", record)
    destination = Path(record["folder"])
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        versions.restore_deleted_version(project, "sample", archived)
    assert sentinel.read_bytes() == b"keep" and Path(archived["folder"]).is_dir()


@pytest.mark.parametrize("change", ["sha", "file", "escape", "run"])
def test_changed_content_or_outside_folder_is_refused(tmp_path, change):
    project, record = make_version(tmp_path)
    selected = deepcopy(record)
    run = "sample"
    if change == "sha":
        selected["sha256"] = "0" * 64
    elif change == "file":
        path = Path(record["folder"]) / "session.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "escape":
        selected["folder"] = str(tmp_path)
    else:
        run = "another-run"
    with pytest.raises((ValueError, FileNotFoundError)):
        versions.archive_version(project, run, selected)
    assert Path(record["folder"]).is_dir()


def test_corrupt_approval_or_protection_metadata_fails_closed(tmp_path):
    project, record = make_version(tmp_path)
    lp.approvals_path(project, "sample").write_text("not-json\n", encoding="utf8")
    with pytest.raises(ValueError):
        versions.archive_version(project, "sample", record)
    assert Path(record["folder"]).is_dir()


def test_drawing_versions_preserve_unrounded_polygons_strokes_and_save_current_before_use(tmp_path):
    project = SimpleNamespace(root=tmp_path)
    original, _proposal = state()
    original.polygons[0].points[0] = (0.123456789012345, 0.987654321098765)
    original.strokes = [lp.Stroke(1, 1.23456789012345, [(2.3456789012345, 3.4567890123456)])]
    record = versions.save_drawing_version(project, original, name="Drawing")
    assert sorted(payloads(record["folder"])) == ["session.json", "version.json"]
    loaded = versions.load_drawing_version(project, "sample", record)
    assert loaded.polygons[0].points == original.polygons[0].points
    assert loaded.strokes[0].radius_px.hex() == original.strokes[0].radius_px.hex()
    current, _proposal = state()
    current.polygons[0].points[0] = (1., 1.)
    selected, backup = versions.use_drawing_version(project, current, record)
    assert selected.polygons[0].points == original.polygons[0].points
    assert versions.load_drawing_version(project, "sample", backup).polygons[0].points == current.polygons[0].points
    assert current.polygons[0].points[0] == (1., 1.)
    assert not lp.session_path(project, "sample").exists()


def test_unified_table_lists_old_and_new_kinds_settings_method_and_time(tmp_path):
    project, old = make_version(tmp_path)
    session, proposal = state()
    drawing = versions.save_drawing_version(project, session, name="Drawing")
    newer = versions.save_map_version(project, session, proposal, [], name="Map",
        method="explicit-method", settings={"choice": 2})
    rows = versions.list_version_metadata(project, "sample")
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {"First", "Drawing", "Map"}
    assert by_name["First"]["proposal_method"] == lp.PROPOSAL_METHOD
    assert by_name["Map"]["settings"] == {"choice": 2}
    assert by_name["Map"]["proposal_method"] == "explicit-method"
    assert by_name["Drawing"]["kind"] == versions.DRAWING_KIND
    assert all(row["created_utc"] and row["sha256"] for row in rows)
    assert {row["sha256"] for row in rows} >= {old["sha256"], drawing["sha256"], newer["sha256"]}
