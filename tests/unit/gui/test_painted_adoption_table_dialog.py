"""Automatic approved-painting facies: Check/Save and visible conflict refusal."""
import csv
import io
from pathlib import Path
import threading

import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower.map_adoption_view import AdoptionDialog

from ..test_geometry_registered_export_apply import registered_setup  # noqa: F401
from ..test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401


def test_automatic_table_needs_no_label_id_choices_and_checked_extra_cells_are_shown(
        qtbot, registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    validate = geometry.validate_geometry_config
    on_main = []
    def checked(config):
        on_main.append(threading.current_thread() is threading.main_thread())
        return validate(config)
    monkeypatch.setattr(geometry, "validate_geometry_config", checked)
    dialog = AdoptionDialog(data.project, prepared["folder"], actor="synthetic")
    qtbot.addWidget(dialog)
    assert dialog.automatic and dialog.choices == {} and dialog.facies_ids() is None
    assert dialog.table.rowCount() == 2
    assert dialog.check_button.isEnabled() and not dialog.save_button.isEnabled()
    assert dialog.check()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert dialog.candidate is not None, dialog.status.text()
    assert dialog.save_button.isEnabled() and on_main and not any(on_main)
    assert "Current frame frozen at preparation" in dialog.status.text()
    assert "no exact historical metric frame" in dialog.status.text()
    headings = [dialog.table.horizontalHeaderItem(i).text() for i in range(dialog.table.columnCount())]
    assert headings[:5] == ["id", "name", "porosity", "permeability", "swi CO2"]
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(1, headings.index("name")).text() == "Sand1"
    assert dialog.table.item(1, headings.index("swi CO2")).text() == "0.12"
    before = geometry.GeometryRevisionStore(data.project).current()["revision_id"]
    confirmations = []
    def confirm(*args, **kwargs):
        confirmations.append(args[2])
        return QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox, "question", confirm)
    dialog.save_button.click()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert dialog.revision is not None, dialog.status.text()
    assert len(confirmations) == 1
    assert "Current frame frozen at preparation" in confirmations[0]
    assert "no exact historical metric frame" in confirmations[0]
    current = geometry.GeometryRevisionStore(data.project).current()
    assert current["revision_id"] != before
    rows = list(csv.DictReader(io.StringIO(Path(current["artifacts"]["facies_properties_path"]).read_text())))
    assert len(rows) == 2 and rows[1]["swi CO2"] == "0.12"


@pytest.mark.parametrize("imported", ["conflict"], indirect=True)
def test_conflicting_source_names_are_shown_and_cannot_be_fixed_by_manual_id_choice(
        qtbot, registered_setup, tmp_path):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    before = geometry.GeometryRevisionStore(data.project).current()["revision_id"]
    dialog = AdoptionDialog(data.project, prepared["folder"], actor="synthetic")
    qtbot.addWidget(dialog)
    assert dialog.automatic and dialog.choices == {}
    assert dialog.check()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert dialog.candidate is None and not dialog.save_button.isEnabled()
    for value in ("Sand1", "swi CO2", "id 1='0.12'", "id 7='0.91'"):
        assert value in dialog.status.text()
    assert not dialog.save()
    assert geometry.GeometryRevisionStore(data.project).current()["revision_id"] == before


@pytest.mark.parametrize("failure_stage", ["candidate", "report"])
def test_unexpected_check_failure_is_shown_and_owned_thread_retires(
        qtbot, registered_setup, tmp_path, monkeypatch, failure_stage):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    before = geometry.GeometryRevisionStore(data.project).current()["revision_id"]
    dialog = AdoptionDialog(data.project, prepared["folder"], actor="synthetic")
    qtbot.addWidget(dialog)
    dialog.show()

    if failure_stage == "candidate":
        def malformed_candidate(*args, **kwargs):
            raise TypeError("malformed checked metadata")
        monkeypatch.setattr(adoption, "adoption_candidate", malformed_candidate)
        expected = "TypeError: malformed checked metadata"
    else:
        # Report processing is part of the owned worker operation too. An
        # unexpected issue schema must not leave its QThread event loop alive.
        monkeypatch.setattr(
            geometry, "validate_geometry_config", lambda _config: {"issues": [None]})
        expected = "AttributeError:"

    assert dialog.check()
    try:
        qtbot.waitUntil(
            lambda: not dialog.busy() and expected in dialog.status.text(), timeout=30000)
        assert expected in dialog.status.text()
        assert dialog._thread is None and dialog._worker is None
        assert not dialog._check_progress_timer.isActive()
        assert dialog.candidate is None and dialog.revision is None
        assert dialog.check_button.isEnabled() and not dialog.save_button.isEnabled()
        assert geometry.GeometryRevisionStore(data.project).current()["revision_id"] == before
        dialog.reject()
        assert not dialog.isVisible()
    finally:
        # Also cleanly retire the pre-fix stranded event loop if this regression
        # fails; all work here is synthetic and no worker is interrupted.
        if dialog.busy():
            thread = dialog._thread
            thread.quit()
            assert thread.wait(30000)
            dialog._thread_finished()
