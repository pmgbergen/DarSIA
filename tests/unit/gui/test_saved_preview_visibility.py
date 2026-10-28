"""Saved-map filters address saved labels without editing the live drawing."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import toml
from PySide6.QtCore import Qt

from .test_painting_background_photo import settings  # noqa: F401
from .test_painting_corrections import view  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual


def test_saved_map_preview_accepts_relocated_photo_but_rejects_another(settings, view, qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower import painting_run_preview_ui as ui
    from darsia.fluidflower.storage_paths import refresh_storage_paths

    record = _saved_manual(view, qtbot, 'Before photo move')
    qtbot.waitUntil(lambda: not view.run_preview.busy())
    frozen = {p.name: p.read_bytes() for p in Path(record['folder']).iterdir()}
    old = Path(view.session.image_path)
    moved = tmp_path / 'relocated-photos' / old.name
    moved.parent.mkdir()
    old.rename(moved)
    registry = tmp_path / 'private-registry.toml'
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases={str(old): str(moved)}))), encoding='utf-8')
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(registry))
    refresh_storage_paths()
    view.run_preview.show_working_map()
    view.session.image_path = str(moved)
    view.run_preview.show_saved_version(record)
    qtbot.waitUntil(lambda: not view.run_preview.busy())
    assert view.run_preview._reference_version_key is not None, view.run_preview.note.text()
    assert view.run_preview.last_result is not None
    assert not view.run_preview.canvas.proposal_item.pixmap().isNull()
    assert {p.name: p.read_bytes() for p in Path(record['folder']).iterdir()} == frozen

    another = moved.with_name('another-photo' + moved.suffix)
    another.write_bytes(moved.read_bytes())
    different = deepcopy(view.session)
    different.image_path = str(another)
    worker = ui._SavedVersionWorker(view.project, different, record, view.current_frame(),
                                    view.run_preview.last_result.photo)
    worker.run()
    assert worker.result is None
    assert 'another reference photograph' in worker.error


def test_saved_visibility_uses_snapshot_labels_and_restores_working_tools(settings, view, qtbot):
    record = _saved_manual(view, qtbot, 'Saved map')
    qtbot.waitUntil(lambda: not view.run_preview.busy())
    archive = {p.name: p.read_bytes() for p in Path(record['folder']).iterdir()}
    saved = deepcopy(view.session.to_dict())
    view.run_preview.show_working_map()
    # This saved layer no longer exists in the live drawing. A filter based
    # on the live session would leave its old saved pixels visible.
    view.session.remove_layer(2)
    view.session.add_layer('Only in working map')
    working = deepcopy(view.session.to_dict())
    view.run_preview.show_saved_version(record)
    qtbot.waitUntil(lambda: not view.run_preview.busy()
        and view.run_preview._reference_version_key is not None)
    preview = view.run_preview
    snapshot = view._version_snapshot_session
    assert snapshot is not view.session
    assert snapshot.to_dict() == saved
    assert [view.layer_table.item(row, 1).text() for row in range(view.layer_table.rowCount())] == [
        layer.name for layer in snapshot.layers]
    assert all(view.facies_table.item(row, 0).text() in snapshot.facies
        for row in range(view.facies_table.rowCount()))
    assert not view.tool_buttons['polygon'].isEnabled()
    assert view.tool_buttons['pan'].isEnabled() and view.map_fit.isEnabled()
    labels = preview.last_result.layers
    layer_row = next(i for i, layer in enumerate(snapshot.layers) if layer.layer_id == 2)
    row, col = map(int, np.argwhere(labels == 2)[0])
    pixel = lambda: preview.canvas.proposal_item.pixmap().toImage().pixelColor(col, row)
    view.layer_all_off_button.click()
    assert view.layer_all_off_button.isChecked() and pixel().alpha() == 0
    view.layer_table.item(layer_row, 4).setCheckState(Qt.CheckState.Checked)
    assert pixel().alpha() > 0
    view.facies_all_off_button.click()
    assert view.facies_all_off_button.isChecked() and pixel().alpha() == 0
    facies = snapshot.layer(2).facies
    facies_row = next(i for i in range(view.facies_table.rowCount())
        if view.facies_table.item(i, 0).text() == facies)
    view.facies_table.item(facies_row, 0).setCheckState(Qt.CheckState.Checked)
    assert pixel().alpha() > 0 and view._color_mode == 'facies'
    assert view.session.to_dict() == working and snapshot.to_dict() == saved
    preview.edit_current_map.click()
    assert view.tool_buttons['polygon'].isEnabled()
    assert [view.layer_table.item(row, 1).text() for row in range(view.layer_table.rowCount())] == [
        layer.name for layer in view.session.layers]
    assert view.session.to_dict() == working
    assert {p.name: p.read_bytes() for p in Path(record['folder']).iterdir()} == archive
