"""Store editor feedback on private metadata with explicit I/O doubles."""
import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from darsia.fluidflower import store_map_ui as ui
from darsia.fluidflower.store_map import parse_store_map


RAW = b'[machine]\nid="local"\n[stores.sample]\nseries="/data/series"\nresults="/products"\n'


@pytest.mark.parametrize("invalid", [
    '[machine]\n[stores.sample]\nseries="/data/series"\nresults="/products"\n',
    '[machine\nid="local"\n',
], ids=["missing-required-field", "malformed-toml"])
def test_invalid_edit_explains_save_and_valid_recovery_preserves_loaded_context(qtbot, tmp_path, invalid):
    path = tmp_path / "stores.toml"
    path.write_bytes(RAW)
    reads, writes = [], []

    def reader(selected, **kwargs):
        reads.append((selected, kwargs))
        return parse_store_map(path.read_bytes(), source={"path": str(path)})

    def writer(*args, **kwargs):
        writes.append((args, kwargs))
        raise AssertionError("Editing and validation must not save settings")

    dialog = ui.StoreSettingsDialog(tmp_path / "series", path=path, reader=reader, writer=writer)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.load()
    qtbot.waitUntil(lambda: dialog._thread is None)
    loaded_status = dialog.status.text()
    loaded_text = dialog.editor.toPlainText()
    assert dialog.save_button.isEnabled() and not dialog.validation_label.isVisible()

    with pytest.raises(ValueError) as failure:
        parse_store_map(invalid.encode("utf-8"))
    dialog.editor.setPlainText(invalid)
    assert not dialog.save_button.isEnabled()
    assert dialog.validation_label.isVisible()
    assert str(failure.value) in dialog.validation_label.text()
    assert str(failure.value) in dialog.save_button.toolTip()
    assert dialog.status.text() == loaded_status
    assert dialog._loaded and dialog._expected == RAW and dialog._displayed_text == loaded_text
    assert dialog.path == path and dialog.series_root == tmp_path / "series"

    dialog.editor.setPlainText(loaded_text)
    assert dialog.save_button.isEnabled()
    assert not dialog.validation_label.isVisible() and not dialog.validation_label.text()
    assert "Cannot save" not in dialog.save_button.toolTip()
    assert dialog.status.text() == loaded_status
    assert len(reads) == 1 and not writes and path.read_bytes() == RAW


def test_validation_feedback_keeps_background_status_and_close_guard(qtbot, tmp_path):
    path = tmp_path / "stores.toml"
    path.write_bytes(RAW)
    entered, release = threading.Event(), threading.Event()

    def reader(selected, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("The metadata fixture was not released")
        return parse_store_map(path.read_bytes(), source={"path": str(path)})

    dialog = ui.StoreSettingsDialog(path=path, reader=reader)
    qtbot.addWidget(dialog)
    dialog.show()
    assert not dialog.save_button.isEnabled() and "Load existing" in dialog.save_button.toolTip()
    dialog.new()
    invalid_reason = dialog.validation_label.text()
    assert invalid_reason and dialog.validation_label.isVisible()
    try:
        dialog.load()
        qtbot.waitUntil(entered.is_set)
        assert "Reading local settings" in dialog.status.text()
        assert dialog.validation_label.text() == invalid_reason
        assert dialog.editor.isReadOnly() and not dialog.save_button.isEnabled()
        assert "current local metadata operation" in dialog.save_button.toolTip()
        dialog.reject()
        assert dialog.isVisible() and dialog._thread is not None
        assert "Wait for the current" in dialog.status.text()
        assert dialog.validation_label.text() == invalid_reason
    finally:
        release.set()
        qtbot.waitUntil(lambda: dialog._thread is None)
    assert dialog.save_button.isEnabled() and not dialog.editor.isReadOnly()
    assert not dialog.validation_label.isVisible() and not dialog.validation_label.text()
    assert "Loaded" in dialog.status.text() and path.read_bytes() == RAW
