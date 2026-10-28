"""Facies edits are visible and individually reversible; saved approvals stay fixed.

Only synthetic photographs and real saved manual versions in temporary projects.
The private-INI view fixture does not use a user's settings or a real series.
"""

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as lpm

from .test_painting_photo_saturation import make_view, settings, view  # noqa: F401


def _row(view, name):
    return next(row for row in range(view.facies_table.rowCount())
                if view.facies_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == name)


def _select(view, record):
    for row in range(view.versions_table.rowCount()):
        if view.versions_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == record["folder"]:
            view.versions_table.selectRow(row)
            view._version_selected()
            return
    raise AssertionError("The saved version is missing from the real version table")


@pytest.fixture
def approved(view):
    name = next(layer.facies for layer in view.session.layers if layer.layer_id == 2)
    view.session.set_facies(name, role="porous_medium", porosity=0.44, permeability=44.0)
    view.session.polygons = [lp.Polygon(2, [(0.0, 0.0), (0.0, 80.0), (60.0, 80.0), (60.0, 0.0)])]
    manual = lpm.manual_map(view.session, (80, 60), frame=view.current_frame())
    assert manual.uncovered_pixels() == 0
    record = lpm.save_manual_version(view.project, view.session, manual, name="V7")
    approval = lpm.approve_manual_version(record)
    view.versions = view._read_versions(view.session.run_id)
    view._read_approvals(view.session.run_id)
    view._refresh_facies_table()
    view._refresh_versions()
    _select(view, record)
    view._undo_stack.clear()
    view._redo_stack.clear()
    folder = Path(record["folder"])
    paths = [folder / leaf for leaf in ("session.json", "manual_map.npz", "version.json")]
    paths.append(folder.parent / "approvals.jsonl")
    immutable = {path: path.read_bytes() for path in paths}
    return name, record, approval, immutable


def _unchanged(immutable):
    assert all(path.read_bytes() == raw for path, raw in immutable.items())


def test_actual_role_control_marks_changed_row_and_step4_and_undo_keeps_saved_approval(view, approved, qtbot):
    name, record, approval, immutable = approved
    assert "match V7 (approved)" in view.facies_approval_note.text()
    assert view.facies_version_note.isHidden()
    # The re-ordered page retains both facies editing and selected-version notices.
    assert view.step_facies.content.isAncestorOf(view.facies_table)
    assert view.step_facies.content.isAncestorOf(view.unit)
    assert view.step_facies.content.isAncestorOf(view.undo_facies_button)
    assert view.step_propose.content.isAncestorOf(view.facies_version_note)
    before_marks = view._snapshot()
    role = view.facies_table.cellWidget(_row(view, name), 2)
    role.setCurrentIndex(role.findData("excluded"))
    qtbot.waitUntil(lambda: "role" in view.facies_version_note.text())
    assert view.session.facies[name].role == "excluded"
    assert view.facies_table.item(_row(view, name), 5).text() == "changed since V7 (approved)"
    assert "Current session differs from V7 (approved)" in view.facies_version_note.text()
    assert not view.facies_version_note.isHidden()
    assert "saved approval is unchanged" in view.facies_version_note.text()
    assert view.versions_table.item(view.versions_table.currentRow(), 5).text().startswith("Approved ")
    assert lp.map_approval(view.map_approvals, record["sha256"]) == approval
    assert len(view._undo_stack) == 1 and len(view._undo_stack[-1]) == 5
    view.undo_facies_button.click()
    qtbot.waitUntil(lambda: view.facies_version_note.isHidden())
    entry = view.session.facies[name]
    assert (entry.role, entry.porosity, entry.permeability) == ("porous_medium", 0.44, 44.0)
    assert view._snapshot() == before_marks
    assert view.facies_table.item(_row(view, name), 5).text() == ""
    view.redo()
    assert view.session.facies[name].role == "excluded"
    view.undo()
    assert view.session.facies[name].role == "porous_medium"
    _unchanged(immutable)


def test_actual_property_cells_and_unit_each_have_one_undo_step(view, approved, qtbot):
    name, _, _, immutable = approved
    original = view._snapshot()
    view.facies_table.item(_row(view, name), 3).setText("0.41")
    qtbot.waitUntil(lambda: view.facies_table.item(_row(view, name), 3).text() == "0.41")
    assert view.session.facies[name].porosity == 0.41
    view.facies_table.item(_row(view, name), 4).setText("21.5")
    qtbot.waitUntil(lambda: view.session.facies[name].permeability == 21.5)
    assert len(view._undo_stack) == 2
    view.unit.setCurrentIndex(view.unit.findData("mD"))
    assert len(view._undo_stack) == 3
    assert "unit" in view.facies_version_note.text()
    view._undo_facies_change()
    assert view.session.permeability_unit == "darcy" and view.unit.currentData() == "darcy"
    assert view.session.facies[name].permeability == 21.5
    view._undo_facies_change()
    assert view.session.facies[name].permeability == 44.0
    assert view.session.facies[name].porosity == 0.41
    view._undo_facies_change()
    qtbot.waitUntil(lambda: "match V7 (approved)" in view.facies_approval_note.text())
    assert view.session.facies[name].porosity == 0.44
    assert view._snapshot() == original and not view._undo_stack
    _unchanged(immutable)


def test_comparison_is_selected_version_specific_and_never_loads_map_arrays(view, approved, monkeypatch):
    name, first, _, immutable = approved
    # Make a second genuine approved version, then compare the current draft
    # with each saved session. The comparison needs no map archive decode.
    view.session.set_facies(name, porosity=0.31)
    manual = lpm.manual_map(view.session, (80, 60), frame=view.current_frame())
    second = lpm.save_manual_version(view.project, view.session, manual, name="V8")
    lpm.approve_manual_version(second)
    view.versions = view._read_versions(view.session.run_id)
    view._read_approvals(view.session.run_id)
    view._refresh_versions()
    monkeypatch.setattr(lpm, "load_manual_version", lambda *args, **kwargs: pytest.fail("Comparison decoded a map"))
    _select(view, second)
    assert "match V8 (approved)" in view.facies_approval_note.text()
    _select(view, first)
    assert "differs from V7 (approved)" in view.facies_version_note.text()
    assert "porosity" in view.facies_version_note.text()
    # The second approval is append-only, not a rewrite of the first record.
    for path, raw in immutable.items():
        if path.name == "approvals.jsonl":
            assert path.read_bytes().startswith(raw)
        else:
            assert path.read_bytes() == raw


def test_tampered_approved_session_reports_unavailable_instead_of_a_matching_baseline(view, approved):
    _, record, _, immutable = approved
    session_file = Path(record["folder"]) / "session.json"
    session_file.write_bytes(session_file.read_bytes() + b" ")
    view._refresh_facies_approval()
    assert "comparison" in view.facies_version_note.text().lower()
    assert "unavailable" in view.facies_version_note.text()
    assert "changed" in view.facies_version_note.text()
    assert view.facies_table.item(0, 5).text() == "Comparison unavailable"
    assert (session_file.parent.parent / "approvals.jsonl").read_bytes() == immutable[session_file.parent.parent / "approvals.jsonl"]


def test_no_selection_no_false_approved_baseline_and_noop_edits_make_no_undo(view, approved):
    name, record, _, immutable = approved
    view.versions_table.clearSelection()
    view._refresh_facies_approval()
    assert "Select an approved manual-map version" in view.facies_approval_note.text()
    assert view.facies_version_note.isHidden()
    view.set_facies_role(name, "porous_medium")
    view.set_facies_properties(name, porosity=0.44, permeability=44.0)
    view._unit_changed(view.unit.currentIndex())
    assert view._undo_stack == []
    assert len(view._snapshot()) == 4
    _unchanged(immutable)
    # A missing approved entry must never be described as a matching draft.
    _select(view, record)
    removed = view.session.facies.pop(name)
    view._refresh_facies_approval()
    assert f"{name} (facies removed)" in view.facies_version_note.text()
    assert "Current session differs from V7 (approved)" in view.facies_approval_note.text()
    assert "match V7" not in view.facies_approval_note.text()
    view.session.facies[name] = removed
    view._refresh_facies_approval()
    assert view.facies_version_note.isHidden()
    _unchanged(immutable)


def test_same_content_approved_folders_compare_the_literal_selected_version(view, approved):
    _, earlier, first_approval, immutable = approved
    manual = lpm.manual_map(view.session, (80, 60), frame=view.current_frame())
    later = lpm.save_manual_version(view.project, view.session, manual, name="V7 copy")
    second_approval = lpm.approve_manual_version(later, now="2026-10-21T23:59:59+00:00")
    assert later["sha256"] == earlier["sha256"]
    assert later["folder"] != earlier["folder"]
    assert first_approval["approved_utc"] != second_approval["approved_utc"]
    view.versions = view._read_versions(view.session.run_id)
    view._read_approvals(view.session.run_id)
    view._refresh_versions()
    for record, name in ((earlier, "V7"), (later, "V7 copy"), (earlier, "V7")):
        _select(view, record)
        assert view.facies_approval_note.text() == f"Current facies match {name} (approved)."
        assert "unavailable" not in view.facies_approval_note.text()
        assert view.facies_version_note.isHidden()
        assert all(view.facies_table.item(row, 5).text() == ""
                   for row in range(view.facies_table.rowCount()))
    for path, raw in immutable.items():
        if path.name == "approvals.jsonl":
            assert path.read_bytes().startswith(raw)
        else:
            assert path.read_bytes() == raw
