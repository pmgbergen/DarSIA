import tomllib

from darsia.fluidflower.image_selection_widget import SeriesImageSelectionDialog
from ..test_production_image_grid import grid_data  # noqa: F401


def test_production_proposal_runs_in_background_and_saves_separate_purpose(qtbot, grid_data):
    root, values = grid_data
    dialog = SeriesImageSelectionDialog(root, metadata_reader=lambda root, run: values[run])
    qtbot.addWidget(dialog)
    controls = dialog.production_controls
    controls.group.setChecked(True)
    before = (root / "series.toml").read_bytes()
    assert controls.count.value() == 80 and controls.through.value() == 48.
    assert controls.late.value() == 10 and not controls.save_button.isEnabled()
    controls.propose()
    qtbot.waitUntil(lambda: not controls.busy)
    assert controls.table.rowCount() == 2
    assert controls.table.item(0, 4).text() == "91"
    assert (root / "series.toml").read_bytes() == before
    assert controls.save_button.isEnabled()
    controls.save()
    document = tomllib.loads((root / "series.toml").read_text())
    assert "production_image_selection" in document and "image_selection" not in document
    assert dialog.expected_bytes == (root / "series.toml").read_bytes()
    controls.count.setValue(90)
    assert controls.proposed is None and not controls.save_button.isEnabled()
    dialog.close()
