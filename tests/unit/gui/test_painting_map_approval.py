"""Layer painting tab: approving a saved version as the run's map.

The Versions box approves the selected version: the record names its SHA-256,
the series' reviewer setting and the time, and nothing else changes. Step 5
names the selected saved version and its literal approval. A synthetic 80 x 60
photograph and a project in a temporary folder; no real series is used.
"""

from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.reviewer_preferences import save_reviewer_name

from .test_painting_corrections import (  # noqa: F401
    COARSE,
    _proposal,
    _show,
    view,
)


def _brush(view):
    view.select_layer_row(2)  # Coarse sand
    view.add_correction(lp.Correction("brush", COARSE, [(30.0, 10.0)], radius_px=2.0))


def test_approve_records_the_version_and_step_4_names_the_approved_map(view):
    assert view.approve_version_button.text() == "Approve version…"
    assert view.versions_table.horizontalHeaderItem(5).text() == "State"
    _show(view, _proposal())
    assert not view.approve_version_button.isEnabled()
    assert "Choose a saved map version under Versions" in view.map_status.text()
    record = view.save_version("Map A")
    view.versions_table.selectRow(0)
    assert "Approve this saved version under Versions" in view.map_status.text()
    save_reviewer_name(view.project.root, "Test reviewer")
    approval = view.approve_version(record)
    assert approval["sha256"] == record["sha256"] and approval["reviewer"] == "Test reviewer"
    assert lp.read_map_approvals(view.project, "ac52") == [approval]
    state = view.versions_table.item(0, 5)
    assert state.text() == "Approved " + approval["approved_utc"][:16].replace("T", " ") + " UTC"
    assert approval["approved_utc"] in state.toolTip()
    assert approval["reviewer"] in state.toolTip()
    assert "Using Map A (approved" in view.map_status.text()
    assert "by Test reviewer" in view.map_status.text()
    assert "by Test reviewer" in view.versions_note.text()
    # Draft edits do not replace the explicitly selected immutable version.
    _brush(view)
    assert "Using Map A (approved" in view.map_status.text()
    view.restore_version(record)
    assert "Using Map A (approved" in view.map_status.text()
    # Approving changed nothing of the map.
    assert view.corrections == [] and view.proposal is not None


def test_the_button_asks_first_and_an_unreadable_file_blocks_approval(view, monkeypatch):
    _show(view, _proposal())
    record = view.save_version("Map A")
    view.versions_table.selectRow(0)
    assert view.approve_version_button.isEnabled()
    questions = []

    def answer(_parent, title, text, *args, **kwargs):
        questions.append((title, text))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", answer)
    view.approve_version_button.click()
    assert lp.read_map_approvals(view.project, "ac52") == []
    title, text = questions[0]
    assert title == "Approve version" and record["sha256"] in text
    assert "No reviewer is set" in text
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    view.approve_version_button.click()
    assert len(lp.read_map_approvals(view.project, "ac52")) == 1
    # A damaged approvals file is said, and nothing more is approved.
    path = lp.approvals_path(view.project, "ac52")
    path.write_text("not json\n", encoding="utf-8")
    view.reload()
    assert "could not be read" in view.map_status.text()
    assert len(view.versions) == 1 and view.versions[0]["sha256"] == record["sha256"]
    assert view.versions[0]["approved"] is None and not view.versions[0]["approval_known"]
    assert "UNKNOWN" in view.versions_table.item(0, 5).text()
    assert view.versions[0]["history_error"] in view.versions_table.item(0, 5).toolTip()
    view.versions_table.selectRow(0)
    assert not view.approve_version_button.isEnabled()
    assert not view.place_version_button.isEnabled()
    assert not view.prepare_adoption_button.isEnabled() and not view.adopt_button.isEnabled()
    assert not view.version_actions.buttons["archive"].isEnabled()
    assert not view.version_actions.buttons["rename"].isEnabled()
    assert not view.version_actions.buttons["note"].isEnabled()
    assert not view.version_actions.buttons["restore_deleted"].isEnabled()
    assert view.approve_version(view.versions[0]) is None
    assert path.read_text(encoding="utf-8") == "not json\n"
