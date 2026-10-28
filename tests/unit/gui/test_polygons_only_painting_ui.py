"""Polygons-only painting routes stay explicit and preserve original defaults."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QInputDialog, QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_polygons_only as polygons
from darsia.fluidflower import layer_painting_version_archive as versions
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_corrections import view, _show  # noqa: F401


@pytest.fixture(autouse=True)
def private_settings(tmp_path):
    previous = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    yield
    QSettings.setDefaultFormat(previous)


def _frame(view, monkeypatch):
    frame = RigFrame(
        corners={"top_left": (4.5, 4.5), "top_right": (4.5, 74.5),
                 "bottom_right": (54.5, 74.5), "bottom_left": (54.5, 4.5)},
        width_m=.07, height_m=.05, run_id=view.session.run_id,
        image=view.session.image_path, image_size=(80, 60),
    )
    monkeypatch.setattr(view, "current_frame", lambda: frame)
    return frame


def _gapped(view, monkeypatch):
    frame = _frame(view, monkeypatch)
    view.session.polygons = [lp.Polygon(2, [(0., 0.), (0., 20.), (59., 20.), (59., 0.)])]
    proposal = polygons.make_proposal(view.image, view.session, frame=frame)
    _show(view, proposal)
    return proposal


def _version_row(view, record):
    row = next(index for index in range(view.versions_table.rowCount())
               if view.versions_table.item(index, 0).data(Qt.ItemDataRole.UserRole) == record["folder"])
    view.versions_table.selectRow(row)
    return row


@pytest.mark.parametrize("worker_name", ["_PlacementWorker", "_AdoptionWorker"])
@pytest.mark.parametrize("new_method", [False, True])
def test_placement_and_adoption_receipts_keep_explicit_new_method_only(view, monkeypatch, tmp_path, worker_name, new_method):
    from darsia.fluidflower import map_adoption, map_placement, raw_shape_basis_reader

    session = deepcopy(view.session)
    layers = np.ones((60, 80), np.uint8)
    record = {"name": "Draft", "kind": "proposal", "sha256": "a" * 64,
              "created_utc": "2026-01-01T00:00:00Z", "folder": "saved", "run_id": session.run_id}
    if new_method:
        record["proposal_method"] = polygons.METHOD
    # This metadata-routing case already mocks drawing and the operation sink;
    # isolate the new presence preflight explicitly, without claiming RAW proof.
    index = tmp_path / "legacy-index.json"
    index.write_text(json.dumps(dict(format=map_placement.GEOMETRY_FORMAT,
        runs={session.run_id: {}})), encoding="utf-8")
    index_identity = dict(path=str(index.resolve()),
        sha256=hashlib.sha256(index.read_bytes()).hexdigest(), size_bytes=index.stat().st_size)
    preflights = []
    def preflight(index, runs):
        preflights.append((str(index), list(runs)))
    if worker_name == "_AdoptionWorker":
        monkeypatch.setattr(raw_shape_basis_reader, "require_basis_index", preflight)
    expected = [(str(index), [session.run_id, session.run_id])] if worker_name == "_AdoptionWorker" else []
    loader = polygons if new_method else slv.lpm
    def version_layers(*args, **kwargs):
        assert preflights == expected  # The preflight precedes even the stub drawing.
        return session, layers.copy()
    monkeypatch.setattr(loader, "version_layers", version_layers)
    calls, failures = [], []
    target = map_placement if worker_name == "_PlacementWorker" else map_adoption
    operation = "place_map_in_runs" if worker_name == "_PlacementWorker" else "prepare_adoption"
    def capture(*args, **kwargs):
        assert preflights == expected
        calls.append(kwargs)
        return {"prepared": True, "geometry_index": index_identity}
    monkeypatch.setattr(target, operation, capture)
    approval = {"sha256": record["sha256"]}
    worker = getattr(slv, worker_name)(record, view.current_frame(), {
        "destination": "unused", "geometry_index": index, "runs": [session.run_id],
        "photo_loader": lambda *_: None, "approval": approval})
    worker.failed.connect(failures.append)
    worker.run()
    assert failures == [] and len(calls) == 1
    assert preflights == expected
    identity = calls[0]["version"]
    assert identity["approval"] == approval
    if new_method:
        assert identity["proposal_method"] == polygons.METHOD
    else:
        assert "proposal_method" not in identity


def test_proposal_method_defaults_to_the_original_route_and_selection_is_display_only(view):
    assert view.proposal_method.currentData() == lp.PROPOSAL_METHOD
    assert view.proposal_method.findData(polygons.METHOD) >= 0
    before = deepcopy(view.session.to_dict())
    view.proposal_method.setCurrentIndex(view.proposal_method.findData(polygons.METHOD))
    assert view.proposal_method.currentData() == polygons.METHOD
    assert view.session.to_dict() == before
    view.proposal_method.setCurrentIndex(view.proposal_method.findData(lp.PROPOSAL_METHOD))
    assert view.proposal_method.currentData() == lp.PROPOSAL_METHOD
    assert view.session.to_dict() == before


def test_worker_without_method_keyword_keeps_original_classifier_and_archive(view, tmp_path, monkeypatch):
    calls = []
    expected = lp.LayerProposal(np.full((60, 80), 2, np.uint8),
                                np.full((60, 80), 255, np.uint8),
                                (80, 60), lp.proposal_digest(view.session, None))

    def classify(image, session, *, frame, progress):
        calls.append((image, session, frame, callable(progress)))
        return expected

    monkeypatch.setattr(lp, "propose_layers", classify)
    monkeypatch.setattr(polygons, "make_proposal", lambda *a, **k: pytest.fail("default selected polygons only"))
    destination = tmp_path / "default-proposal.npz"
    worker = slv._ProposalWorker(view.image, view.session, None, destination)
    finished, failed = [], []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.run()
    assert failed == [] and finished == [expected]
    assert calls == [(view.image, view.session, None, True)]
    saved = polygons.load_proposal(destination)
    assert not polygons.is_polygons_only(saved)
    assert saved.digest == expected.digest and np.array_equal(saved.classes, expected.classes)


def test_polygons_worker_dispatch_does_not_classify_and_saves_full_grid(view, tmp_path, monkeypatch):
    frame = _frame(view, monkeypatch)
    view.session.polygons = [lp.Polygon(2, [(10., 10.), (10., 30.), (30., 30.), (30., 10.)])]
    view.session.strokes = [lp.Stroke(3, 5., [(40., 50.)])]
    monkeypatch.setattr(lp, "propose_layers", lambda *a, **k: pytest.fail("polygons-only worker classified"))
    destination = tmp_path / "polygons-proposal.npz"
    worker = slv._ProposalWorker(view.image, view.session, frame, destination, method=polygons.METHOD)
    finished, failed = [], []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.run()
    assert failed == [] and len(finished) == 1
    proposal = polygons.load_proposal(destination)
    assert polygons.is_polygons_only(proposal)
    assert proposal.classes.shape == (60, 80)
    assert proposal.classes[20, 20] == 2
    assert proposal.inside[40, 50] and proposal.classes[40, 50] == lp.OUTSIDE
    assert not proposal.inside[0, 0]
    assert np.array_equal(proposal.classes, finished[0].classes)


def test_gui_propose_routes_selected_method_in_worker_with_exact_snapshot(view, qtbot, monkeypatch):
    frame = _frame(view, monkeypatch)
    precise = (10.123456789, 10.987654321)
    view.session.polygons = [lp.Polygon(2, [precise, (10., 30.), (30., 30.), (30., 10.)])]
    snapshot = deepcopy(view.session)
    seen = []
    original = polygons.make_proposal

    def observe(image, session, frame=None, progress=None):
        seen.append((threading.current_thread() is threading.main_thread(), deepcopy(session)))
        return original(image, session, frame=frame, progress=progress)

    monkeypatch.setattr(polygons, "make_proposal", observe)
    monkeypatch.setattr(lp, "propose_layers", lambda *a, **k: pytest.fail("GUI selected classifier"))
    view.proposal_method.setCurrentIndex(view.proposal_method.findData(polygons.METHOD))
    view.propose()
    qtbot.waitUntil(lambda: view._proposal_thread is None, timeout=10000)
    assert seen and seen[0][0] is False
    assert seen[0][1].polygons == snapshot.polygons
    assert view._proposal_worker is None
    assert polygons.is_polygons_only(view.proposal)
    assert view.proposal.source["polygons"][0][1][0] == list(precise)
    assert polygons.proposal_current(view.proposal, view.session, frame)
    assert polygons.is_polygons_only(polygons.load_proposal(lp.proposal_path(view.project, view.session.run_id)))


def test_fill_of_zero_gap_stays_inside_frame_and_is_one_undo_step(view, monkeypatch):
    proposal = _gapped(view, monkeypatch)
    original = proposal.classes.copy()
    view.select_layer_row(2)  # Selected layer is distinct from the drawn polygon.
    assert not view.fill_region_at(0., 0.)
    assert view.corrections == []
    assert view.fill_region_at(30., 40.)
    classes, decided, skipped = view._corrected_native_state()
    gap = proposal.inside & (original == lp.OUTSIDE)
    assert not skipped and gap.any()
    assert np.all(classes[gap] == 3) and np.all(decided[gap])
    assert np.all(classes[~proposal.inside] == lp.OUTSIDE)
    assert not decided[~proposal.inside].any()
    assert np.array_equal(classes[original == 2], original[original == 2])
    assert len(view.corrections) == 1 and view.corrections[0].replaced_layer == lp.OUTSIDE
    assert np.array_equal(proposal.classes, original)
    view.undo()
    assert view.corrections == []
    assert np.array_equal(view._corrected_native_state()[0], original)
    view.redo()
    assert len(view.corrections) == 1
    assert np.all(view._corrected_native_state()[0][gap] == 3)


def test_hidden_correction_display_does_not_change_fill_source_or_export_gap_gate(view, monkeypatch):
    proposal = _gapped(view, monkeypatch)
    assert any("unassigned" in text for text in view.export_problems_now())
    view.select_layer_row(2)
    assert view.fill_region_at(30., 40.)
    view.show_corrections.setChecked(False)
    full = view._corrected_native_state()[0]
    assert not (full[proposal.inside] == lp.OUTSIDE).any()
    # Hiding corrections only changes the overlay; there is no remaining
    # zero region in the full corrected state for Fill to select again.
    count = len(view.corrections)
    assert not view.fill_region_at(30., 40.)
    assert len(view.corrections) == count
    assert not any("unassigned" in text for text in view.export_problems_now())
    assert (proposal.classes[proposal.inside] == lp.OUTSIDE).any()


def test_export_with_gaps_refuses_before_starting_a_save_worker(view, monkeypatch):
    _gapped(view, monkeypatch)
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: messages.append(a[2]))
    monkeypatch.setattr(slv, "_SaveWorker", lambda *a, **k: pytest.fail("gapped proposal started export"))
    view.save_and_export()
    assert messages and any("unassigned" in text for text in messages)
    assert view._save_thread is None


def test_drawing_version_actions_keep_exact_marks_and_metadata_outside_content(view, monkeypatch):
    view.project.root.mkdir(parents=True, exist_ok=True)
    precise = (10.123456789, 10.987654321)
    view.session.polygons = [lp.Polygon(2, [precise, (10., 30.), (30., 30.), (30., 10.)])]
    original = deepcopy(view.session.polygons)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("First drawing", True))
    record = view.version_actions.save_drawing()
    assert record is not None and record["kind"] == versions.DRAWING_KIND
    assert view.versions_table.columnCount() == 6
    assert [view.versions_table.horizontalHeaderItem(i).text() for i in range(6)] == [
        "Name", "Kind", "Settings", "Saved UTC", "SHA-256", "State"]
    row = _version_row(view, record)
    assert view.versions_table.item(row, 1).text() == "drawing"
    assert view.versions_table.item(row, 2).text() == "drawing only"
    content = {p.name: p.read_bytes() for p in Path(record["folder"]).iterdir()}
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Renamed drawing", True))
    view.version_actions.rename()
    current = next(r for r in view.versions if r["sha256"] == record["sha256"])
    _version_row(view, current)
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *a, **k: ("A drawing note", True))
    view.version_actions.note()
    metadata = next(r for r in view.versions if r["sha256"] == record["sha256"])
    assert metadata["display_name"] == "Renamed drawing" and metadata["note"] == "A drawing note"
    assert {p.name: p.read_bytes() for p in Path(record["folder"]).iterdir()} == content
    assert lp.map_approval(view.map_approvals, record["sha256"]) is None
    view.session.polygons = [lp.Polygon(3, [(35., 40.), (35., 60.), (50., 60.), (50., 40.)])]
    changed = deepcopy(view.session.polygons)
    assert view.version_actions.use_drawing(metadata)
    assert view.session.polygons == original
    assert view.session.polygons[0].points[0] == precise
    backups = [r for r in view.versions if r["sha256"] != record["sha256"] and r["kind"] == versions.DRAWING_KIND]
    assert len(backups) == 1
    restored_backup = versions.load_drawing_version(view.project, view.session.run_id, backups[0])
    assert restored_backup.polygons == changed
    assert view.proposal is None


def test_unfinished_differences_use_exact_native_classes_without_relaxing_export(view, monkeypatch):
    proposal = _gapped(view, monkeypatch)
    worker = slv._ManualMapWorker(deepcopy(view.session), view.image.image_size,
        view.current_frame(), dict(run_id=view.session.run_id, manual_key="", compare_key="preview",
                                   proposal=proposal, corrections=[]))
    finished, failed = [], []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.run()
    assert not failed and len(finished) == 1
    assert np.array_equal(finished[0]["layers"], proposal.classes)
    assert (finished[0]["layers"][proposal.inside] == 0).any()
    with pytest.raises(ValueError, match="unassigned"):
        polygons.painted_layers(view.session, proposal, frame=view.current_frame())


def test_map_restore_keeps_exact_stale_draft_even_if_legacy_map_sha_collides(view, monkeypatch):
    frame = _frame(view, monkeypatch)
    precise = (10.123456789, 10.987654321)
    view.session.polygons = [lp.Polygon(2, [precise, (10., 30.), (30., 30.), (30., 10.)])]
    proposal = polygons.make_proposal(view.image, view.session, frame=frame)
    _show(view, proposal)
    record = view.save_version("Before edit")
    assert record is not None
    view.session.polygons[0].points[0] = (precise[0] + .000000001, precise[1])
    changed = deepcopy(view.session.polygons)
    assert not view.proposal_current()
    assert lp.map_digest(view.session, proposal, []) == record["sha256"]
    assert view.restore_version(record)
    backups = [row for row in view.versions if row.get("kind") == versions.DRAWING_KIND]
    assert len(backups) == 1
    assert versions.load_drawing_version(view.project, view.session.run_id, backups[0]).polygons == changed
    assert view.session.polygons[0].points[0] == precise
