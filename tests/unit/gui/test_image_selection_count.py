"""Image-count proposals use the ordinary draft and explicit save path."""

from dataclasses import replace
import tomllib

import pytest

from darsia.fluidflower import image_selection_widget as ui
from darsia.fluidflower import series_image_selection as selection

from .test_series_image_selection import app, metadata, root  # noqa: F401


@pytest.fixture(autouse=True)
def discard_unsaved_test_drafts(monkeypatch):
    monkeypatch.setattr(ui.QMessageBox, "question", lambda *_args: ui.QMessageBox.StandardButton.Yes)


def test_count_preview_apply_and_save_preserve_existing_images(root, metadata, app):
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: metadata)
    assert not dialog.count_controls.button.isEnabled()
    dialog.load_run()
    kept = [metadata.frames[i]["source_path"] for i in (0, 4, 8, 17, 80, 200, 417)]
    dialog._set_draft_paths(kept)
    original = (root / "series.toml").read_bytes()
    result = dialog.count_controls.propose()
    assert len(result["selected_paths"]) == 80
    assert set(kept) <= set(result["selected_paths"])
    assert dialog.selected_paths() == kept
    assert (root / "series.toml").read_bytes() == original
    assert dialog.proposal_table.rowCount() == 80
    assert "80 without a saved quality class" in dialog.proposal_status.text()
    assert dialog.apply_button.isEnabled()
    dialog.apply_button.click()
    assert dialog.selected_paths() == result["selected_paths"]
    assert (root / "series.toml").read_bytes() == original
    dialog.save_run()
    saved = tomllib.loads((root / "series.toml").read_text())
    assert len(saved["image_selection"]["runs"]["ac17"]["selected"]) == 80
    assert saved["image_selection"]["runs"]["ac17"]["rule"].startswith("80 images")
    assert saved["method"]["phase_model"]["model_version"] == "existing-model"
    dialog.close()


def test_count_or_draft_change_requires_fresh_preview(root, metadata, app):
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: metadata)
    dialog.load_run()
    dialog._set_draft_paths([metadata.frames[i]["source_path"] for i in (0, 417)])
    dialog.count_controls.propose()
    dialog.count_controls.count.setValue(81)
    assert not dialog.apply_button.isEnabled()
    assert dialog.count_controls.proposed is None
    dialog.count_controls.propose()
    dialog.table.item(4, 0).setCheckState(ui.Qt.CheckState.Checked)
    assert dialog.count_controls.proposed is None
    assert not dialog.apply_button.isEnabled()
    dialog.close()


def test_batch_count_only_changes_runs_with_saved_choices(root, metadata, app):
    initial = (root / "series.toml").read_bytes()
    kept = [metadata.frames[i]["source_path"] for i in (0, 417)]
    selection.save_run_image_selection(root, metadata, kept, expected_bytes=initial)
    dialog = ui.SeriesImageSelectionDialog(
        root, metadata_reader=lambda _root, run: replace(metadata, run_id=run)
    )
    lines = dialog.count_controls.save_all()
    assert lines == ["AC17: saved 80 images"]
    saved = tomllib.loads((root / "series.toml").read_text())["image_selection"]["runs"]
    assert list(saved) == ["ac17"]
    assert len(saved["ac17"]["selected"]) == 80
    dialog.close()


def test_insufficient_timeline_leaves_saved_choice_and_draft_unchanged(root, metadata, app):
    short = replace(metadata, frames=metadata.frames[:20])
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: short)
    dialog.load_run()
    kept = [short.frames[i]["source_path"] for i in (0, 19)]
    dialog._set_draft_paths(kept)
    before = (root / "series.toml").read_bytes()
    with pytest.raises(ValueError, match="80 images are required"):
        dialog.count_controls.propose()
    assert dialog.selected_paths() == kept
    assert dialog.count_controls.proposed is None
    assert (root / "series.toml").read_bytes() == before
    dialog.close()
