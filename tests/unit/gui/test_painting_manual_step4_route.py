"""Selected saved manual identity, tiny private photographs and real GUI routes.

No actual series, placement, export, approval withdrawal or adoption is run.
The existing manual renderer works only on the 80x60 private fixture.
"""
from pathlib import Path
from types import SimpleNamespace
import pytest
from PySide6.QtCore import Qt

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import series_layer_view as view_module

from .test_painting_corrections import _proposal, _show, view  # noqa: F401
from .test_painting_manual_map import LEFT, RIGHT, MIDDLE, _drawn


def _saved_manual(view, qtbot, name="Selected map"):
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(polygon)
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    _drawn(view, qtbot)
    assert view.proposal is None
    record = view.save_version(name)
    assert record["uncovered_pixels"] == 0
    view.versions_table.selectRow(0)
    return record


def _select(view, record):
    for row in range(view.versions_table.rowCount()):
        if view.versions_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == record["folder"]:
            view.versions_table.selectRow(row)
            return
    raise AssertionError("The saved version was absent from the actual table")


def test_approved_manual_without_proposal_uses_saved_identity_and_round5_route(view, qtbot, monkeypatch):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    assert approval is not None
    view.versions_table.selectRow(0)
    monkeypatch.setattr(view, "_current_map_digest", lambda: (_ for _ in ()).throw(
        AssertionError("Selected manual status must not hash the proposal")))
    history_before = lp.approvals_path(view.project, view.session.run_id).read_bytes()
    view._refresh_status()
    assert "Using Selected map (approved" in view.map_status.text()
    assert record["sha256"][:12] in view.map_status.text()
    assert "not saved as a version" not in view.map_status.text()
    assert "Missing input" not in view.step_export.status.text()
    assert "Use the map in the series" in view.export_problems.text()
    assert not view.export_button.isVisibleTo(view)
    assert not view.export_button.isEnabled()
    before = {p: p.read_bytes() for p in Path(record["folder"]).iterdir()}
    view.step_use_map.set_expanded(False)
    view.export_problems.linkActivated.emit("round5")
    assert view.step_use_map.is_expanded()
    assert before == {p: p.read_bytes() for p in Path(record["folder"]).iterdir()}
    assert lp.approvals_path(view.project, view.session.run_id).read_bytes() == history_before
    assert view._placement_thread is None and view._adoption_thread is None and view._save_thread is None


def test_multiple_approved_same_content_versions_show_the_selected_literal_approval(view, qtbot, monkeypatch, tmp_path):
    first = _saved_manual(view, qtbot, "Earlier approved map")
    first_approval = manual.approve_manual_version(first, now="2026-01-01T01:00:00Z")
    second = view.save_version("Later approved map")
    assert first["sha256"] == second["sha256"] and first["folder"] != second["folder"]
    view.versions = view._read_versions(view.session.run_id)
    view._read_approvals(view.session.run_id)
    view._refresh_versions()
    _select(view, second)
    assert view._approval_for_version(second) is None
    assert "Later approved map, not approved" in view.map_status.text()
    assert view.versions_table.item(view.versions_table.currentRow(), 5).text() == "Not approved"
    assert not view.place_version_button.isEnabled() and not view.prepare_adoption_button.isEnabled()
    assert not view.place_version(second, tmp_path / "unread-index.json", runs=[view.session.run_id])
    assert not view.prepare_adoption_of(second, tmp_path / "unread-index.json", runs=[view.session.run_id])
    assert view._placement_thread is None and view._adoption_thread is None
    assert lp.read_map_approvals(view.project, view.session.run_id) == [first_approval]
    second_approval = manual.approve_manual_version(second, now="2026-01-01T02:00:00Z")
    assert first["sha256"] == second["sha256"] and first["folder"] != second["folder"]
    view.versions = view._read_versions(view.session.run_id)
    view._read_approvals(view.session.run_id)
    view._refresh_versions()
    _select(view, first)
    assert "Using Earlier approved map (approved 2026-01-01 01:00 UTC" in view.map_status.text()
    assert "this selected approval counts" in view.versions_note.text()
    _select(view, second)
    assert "Using Later approved map (approved 2026-01-01 02:00 UTC" in view.map_status.text()
    assert lp.read_map_approvals(view.project, view.session.run_id) == [first_approval, second_approval]
    assert view._approval_for_version(second) == second_approval
    assert view.place_version_button.isEnabled() and view.prepare_adoption_button.isEnabled()
    assert view.use_map_reasons[view.adopt_button].text() == ""
    # Capture the GUI handoff before either real worker is constructed/started.
    # Geometry is only a synthetic selector here; no placement proof is claimed.
    index = tmp_path / "selection-only.json"
    monkeypatch.setattr(view.run_preview, "geometry_selection", lambda: SimpleNamespace(index_path=str(index)))
    captured = []
    class HandoffCaptured(Exception):
        pass
    def capture(record, frame, options):
        captured.append(options["approval"])
        raise HandoffCaptured()
    monkeypatch.setattr(view_module, "_PlacementWorker", capture)
    monkeypatch.setattr(view_module, "_AdoptionWorker", capture)
    with pytest.raises(HandoffCaptured):
        view.place_version(second, index, runs=[view.session.run_id], photo_loader=lambda *args: None)
    with pytest.raises(HandoffCaptured):
        view.prepare_adoption_of(second, index, runs=[view.session.run_id], current_labels=lambda run: None)
    assert captured == [second_approval, second_approval]
    assert view._placement_thread is None and view._adoption_thread is None
    assert lp.read_map_approvals(view.project, view.session.run_id) == [first_approval, second_approval]


def test_unapproved_saved_manual_names_the_exact_missing_approval_and_click(view, qtbot):
    _saved_manual(view, qtbot)
    view._refresh_status()
    assert "Selected map, not approved" in view.map_status.text()
    assert "Approve the selected manual map" in view.step_export.status.text()
    assert "Open Use the map in the series" in view.export_problems.text()
    assert "not saved as a version" not in view.map_status.text()
    assert not view.export_button.isVisibleTo(view)


def test_selecting_a_proposal_after_manual_restores_the_existing_legacy_status(view, qtbot):
    manual_record = _saved_manual(view, qtbot)
    view.approve_version(manual_record)
    _show(view, _proposal())
    view.map_shown.setCurrentIndex(view.map_shown.findText("Proposal"))
    proposal = view.save_version("Proposal draft")
    _select(view, proposal)
    view.step_export.set_expanded(True)
    # N407 keeps the old API/status objects, with the obsolete section hidden.
    assert not view.step_export.isVisibleTo(view)
    assert not view.export_button.isVisibleTo(view)
    assert "round-5 route" not in view.export_problems.text()
    assert "Approved manual map" not in view.step_export.status.text()


def test_legacy_unsaved_proposal_keeps_its_existing_status(view):
    _show(view, _proposal())
    view._refresh_status()
    assert "Choose a saved map version under Versions" in view.map_status.text()
    view.step_use_map.set_expanded(True)
    view.step_export.set_expanded(True)
    # N407 keeps the old API/status objects, with the obsolete section hidden.
    assert not view.step_export.isVisibleTo(view)
    assert not view.export_button.isVisibleTo(view)
    assert view.export_problems.text() == " ".join(view.export_problems_now())


def test_unreadable_approval_history_never_claims_selected_manual_is_approved(view, qtbot):
    record = _saved_manual(view, qtbot)
    view.approve_version(record)
    lp.approvals_path(view.project, view.session.run_id).write_bytes(b"invalid history\n")
    view._read_approvals(view.session.run_id)
    view._refresh_status()
    assert "could not be read" in view.map_status.text()
    assert "cannot be read" in view.step_export.status.text()
    assert not view.export_button.isEnabled()
