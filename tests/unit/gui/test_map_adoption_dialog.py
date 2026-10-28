"""The adoption window: the facies table of a prepared map, its check and save.

The synthetic two-run corrected workspace of the registered export tests and
a painted map prepared for both runs (see test_map_adoption_store). Nothing
is chosen for the user; Check validates as the store will; Save appends a
revision only after the question. Temporary folders only.
"""

import threading

from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower.map_adoption_view import AdoptionDialog

from ..test_geometry_imported_rasters import imported  # noqa: F401
from ..test_geometry_registered_export_apply import registered_setup  # noqa: F401
from ..test_map_adoption_store import _prepare
from .test_painting_corrections import view  # noqa: F401


def _choose(dialog, value, facies_id):
    choice = dialog.choices[value]
    choice.setCurrentIndex(choice.findData(facies_id))


def test_the_facies_table_is_chosen_checked_and_saved(qtbot, registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    prepared = _prepare(data, tmp_path)
    # Every whole-revision validation runs off the GUI thread.
    on_gui_thread = []
    validate = geometry.validate_geometry_config

    def spy(config):
        on_gui_thread.append(threading.current_thread() is threading.main_thread())
        return validate(config)

    monkeypatch.setattr(geometry, "validate_geometry_config", spy)
    dialog = AdoptionDialog(data.project, prepared["folder"], actor="synthetic owner")
    qtbot.addWidget(dialog)
    assert dialog.values == [5, 6]
    # Nothing is chosen for the user.
    assert dialog.facies_ids() == {}
    assert not dialog.check_button.isEnabled() and not dialog.save_button.isEnabled()
    texts = [dialog.choices[5].itemText(index) for index in range(dialog.choices[5].count())]
    assert texts[0] == "Choose…" and texts[1].startswith("1: Sand1")
    _choose(dialog, 5, 1)
    assert not dialog.check_button.isEnabled()
    _choose(dialog, 6, 2)
    assert dialog.check_button.isEnabled()
    assert dialog.check()
    assert not dialog.check() and dialog.busy()  # One worker at a time.
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert "Ready to save: 2 runs and 2 regions" in dialog.status.text()
    assert dialog.save_button.isEnabled()
    before = geometry.GeometryRevisionStore(data.project).current()["revision_id"]
    asked = []

    def no(_parent, title, *args, **kwargs):
        asked.append(title)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", no)
    dialog.save_button.click()
    assert asked == ["Save revision"] and not dialog.busy()
    assert geometry.GeometryRevisionStore(data.project).current()["revision_id"] == before
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    dialog.save_button.click()
    qtbot.waitUntil(lambda: dialog.revision is not None and not dialog.busy(), timeout=30000)
    current = geometry.GeometryRevisionStore(data.project).current()
    assert current["revision_id"] == dialog.revision["revision_id"] != before
    assert "Saved revision" in dialog.status.text()
    # One save per window.
    assert not dialog.check_button.isEnabled() and not dialog.save_button.isEnabled()
    assert on_gui_thread and not any(on_gui_thread)


def test_a_revision_that_does_not_validate_is_said_and_not_saved(qtbot, registered_setup, tmp_path):
    data = registered_setup
    prepared = _prepare(data, tmp_path, approval="b" * 64)
    dialog = AdoptionDialog(data.project, prepared["folder"], actor="synthetic owner")
    qtbot.addWidget(dialog)
    assert not dialog.check() and "every label" in dialog.status.text()
    _choose(dialog, 5, 1)
    _choose(dialog, 6, 2)
    assert dialog.check()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert "does not validate" in dialog.status.text() and "approval" in dialog.status.text()
    assert not dialog.save_button.isEnabled() and not dialog.save()


def test_the_painting_tab_opens_the_adoption_window(view, monkeypatch, tmp_path):
    assert view.adopt_button.text() == "Adopt as the series' sand map…"
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[1]))
    assert view.open_adoption(tmp_path) is None
    assert warnings == ["Could not open the adoption"]
