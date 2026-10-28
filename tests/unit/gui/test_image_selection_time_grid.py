"""Image choices from the series' time grids, for one run or all runs.

The grids are the [[data_time]] tables of config/common.toml. A proposal uses
the existing camera-grid selector; Apply and Save stay explicit, and "Propose
and save for all runs" does both for every run after one question. Local
metadata and offscreen Qt only; no image is decoded.
"""

from dataclasses import replace
import tomllib

import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import image_selection_grid as grid
from darsia.fluidflower import image_selection_widget as ui

from .test_series_image_selection import app, metadata, root  # noqa: F401

GRIDS = """
[[data_time]]
name = "comparison_grid_v1_injection"
times = ["00:07:30", "00:15:00", "00:30:00"]
tol = "00:00:15"

[[data_time]]
name = "comparison_grid_v1_rest"
times = ["02:30:00", "24:00:00", "200:00:00"]
tol = "00:02:30"

[[data_time]]
name = "mass_operational_boundaries"
times = ["04:06:00"]
tol = "00:02:30"
"""


@pytest.fixture
def grids_root(root):  # noqa: F811
    (root / "config").mkdir()
    (root / "config" / "common.toml").write_text(GRIDS, encoding="utf-8")
    return root


def _saved(root):
    return tomllib.loads((root / "series.toml").read_text(encoding="utf-8"))[
        "image_selection"
    ]["runs"]


def test_clock_hours_reads_signed_times_with_seconds():
    assert grid.clock_hours("02:30:00") == 2.5
    assert grid.clock_hours("-00:15:00") == -0.25
    assert grid.clock_hours("100:00:01.8") == pytest.approx(100.0005)
    with pytest.raises(ValueError, match="HH:MM:SS"):
        grid.clock_hours("2:30")


def test_grids_are_read_from_common_toml_and_grouped_in_families(grids_root, tmp_path):
    grids = grid.read_time_grids(grids_root)
    assert list(grids) == [
        "comparison_grid_v1_injection",
        "comparison_grid_v1_rest",
        "mass_operational_boundaries",
    ]
    assert grids["comparison_grid_v1_injection"]["times_h"] == [0.125, 0.25, 0.5]
    assert grids["comparison_grid_v1_rest"]["tolerance_s"] == pytest.approx(150.0)
    assert grid.grid_families(grids) == {
        "comparison_grid_v1": ["comparison_grid_v1_injection", "comparison_grid_v1_rest"],
        "mass_operational_boundaries": ["mass_operational_boundaries"],
    }
    assert grid.grid_target_hours(grids, ["comparison_grid_v1_rest", "comparison_grid_v1_injection"]) == [
        0.125, 0.25, 0.5, 2.5, 24.0, 200.0
    ]
    with pytest.raises(ValueError, match="No time grid named"):
        grid.grid_target_hours(grids, ["grid_v9"])
    with pytest.raises(ValueError, match="at least one"):
        grid.grid_target_hours(grids, [])
    assert grid.read_time_grids(tmp_path) == {}


def test_a_proposal_matches_each_time_and_reports_the_missing_ones(grids_root, metadata):  # noqa: F811
    grids = grid.read_time_grids(grids_root)
    names = grid.grid_families(grids)["comparison_grid_v1"]
    result = grid.propose_from_grid(metadata, grids, names)
    assert result["missing_hours"] == [200.0]  # After the run's last image.
    chosen = [path.rsplit("image_", 1)[1] for path in result["selected_paths"]]
    # The last image before injection comes first, as every saved choice needs.
    assert chosen == ["0000.png", "0003.png", "0005.png", "0009.png", "0023.png", "0109.png"]
    assert result["baseline_path"] == result["selected_paths"][0]
    assert result["unrecorded_quality"] == 6  # No saved quality receipt.
    summary = grid.proposal_summary(result)
    assert summary.startswith("5 images for 6 times of comparison_grid_v1_injection")
    assert "and the image before injection" in summary
    assert "1 time(s) without an image" in summary and "200 h" in summary
    # A saved class is used where the receipt has the image.
    key = result["selected_paths"][1].replace("\\", "/").casefold()
    rated = grid.propose_from_grid(
        metadata, grids, names, quality={"classes": {key: {"class": "minor"}}}
    )
    assert rated["unrecorded_quality"] == 5
    assert rated["target_matches"][0]["frame_quality_class"] == "minor"


def test_the_dialog_proposes_applies_and_saves_one_run(grids_root, metadata, app):  # noqa: F811
    dialog = ui.SeriesImageSelectionDialog(grids_root, metadata_reader=lambda *_: metadata)
    assert dialog.grid_combo.itemText(0) == "comparison_grid_v1 (6 times)"
    assert dialog.grid_combo.count() == 2
    assert not dialog.grid_button.isEnabled() and dialog.grid_all_button.isEnabled()
    dialog.load_run()
    assert dialog.grid_button.isEnabled()
    result = dialog.propose_from_time_grid()
    assert dialog.proposal_table.rowCount() == 6
    assert dialog.proposal_table.item(5, 2).text() == "Not selected"
    assert "5 images for 6 times" in dialog.proposal_status.text()
    assert dialog.apply_button.isEnabled()
    assert dialog.apply_proposal() == result["selected_paths"]
    dialog.save_run()
    saved = _saved(grids_root)["ac17"]
    assert saved["rule"] == "time grid: comparison_grid_v1_injection, comparison_grid_v1_rest"
    assert saved["requested_target_hours"] == [0.125, 0.25, 0.5, 2.5, 24.0]
    # A manual change after Apply is saved as a manual choice.
    dialog.propose_from_time_grid()
    dialog.apply_proposal()
    dialog.table.item(1, 0).setCheckState(ui.Qt.CheckState.Checked)  # Not chosen before.
    dialog.save_run()
    assert _saved(grids_root)["ac17"]["rule"] == "manual_or_edited_targets"


def test_propose_and_save_for_all_runs(grids_root, metadata, app, monkeypatch):  # noqa: F811
    dialog = ui.SeriesImageSelectionDialog(
        grids_root, metadata_reader=lambda _root, run: replace(metadata, run_id=run)
    )
    lines = dialog.propose_and_save_all()
    assert [line.split(":")[0] for line in lines] == ["AC17", "AC18"]
    saved = _saved(grids_root)
    assert sorted(saved) == ["ac17", "ac18"]
    assert all(row["rule"].startswith("time grid:") for row in saved.values())
    assert dialog.run_combo.currentIndex() == 0
    # The button asks once first.
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.StandardButton.No
    )
    dialog.grid_all_button.click()
    assert "For each of the 2 runs" in asked[0]
    # Unsaved choices are never overwritten.
    dialog.load_run()
    dialog.table.item(1, 0).setCheckState(ui.Qt.CheckState.Checked)  # Not chosen before.
    with pytest.raises(ValueError, match="unsaved image choices first: AC17"):
        dialog.propose_and_save_all()


def test_all_runs_stops_at_the_first_run_that_cannot_be_saved(grids_root, metadata, app):  # noqa: F811
    def reader(_root, run):
        if run == "ac18":
            raise ValueError("protocol missing")
        return replace(metadata, run_id=run)

    dialog = ui.SeriesImageSelectionDialog(grids_root, metadata_reader=reader)
    with pytest.raises(RuntimeError) as failure:
        dialog.propose_and_save_all()
    text = str(failure.value)
    assert text.startswith("AC17: saved;") and "AC18: not saved: ValueError: protocol missing" in text
    assert sorted(_saved(grids_root)) == ["ac17"]
