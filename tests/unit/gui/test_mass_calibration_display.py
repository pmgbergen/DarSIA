"""Synthetic saved metadata and actual disabled/read-only Qt display tests."""

import builtins
import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView

from darsia.fluidflower import mass_calibration_display as display


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value).encode("utf-8"))
    return path


def values(value):
    return {name: saved for name, saved, _source in value.rows}


def test_phase_split_results_distinguish_measured_estimated_and_no_reference(tmp_path):
    path = write_json(tmp_path / "new-fit.json", dict(parameters=dict(gain=1., beta=0.),
        overflow_phase_split_policy="own-measurement-or-pressure-matched-phase-split-v1",
        overflow_phase_split_status=[
            dict(run="one", kind="own_measurement", image_count=3, donor_runs=[]),
            dict(run="two", kind="pressure_matched_estimate", image_count=2,
                 donor_runs=["one"], pressure_difference_bar_range=[.001, .004]),
            dict(run="two", kind="model_phase_split", image_count=1, donor_runs=[])]))
    value = display.read_mass_calibration_display("phase_model", str(path), series_root=tmp_path)
    shown = [row for row in value.rows if row[0].startswith("Overflow phase split")]
    assert any("Measured" in row[1] and "3 image" in row[1] for row in shown)
    assert any("ESTIMATED" in row[1] and "one" in row[1] and "ΔP" in row[1] for row in shown)
    assert any("no overflow reference" in row[1] for row in shown)


@pytest.fixture
def phase(tmp_path):
    path = write_json(
        tmp_path / "frozen.json",
        {
            "contract": "ffac-common-mass-round3-frozen-v1",
            "frozen": True,
            "parameters": {
                "candidate": "N1",
                "gain": 23.859055832362177,
                "beta": 0.6513006394912089,
            },
            "calibration_files": {"global": {"path": str(tmp_path / "global.json")}},
        },
    )
    (tmp_path / "global.json").write_bytes(b"not to be read")
    return path


@pytest.fixture
def colour(tmp_path):
    folder = tmp_path / "colour-calibration"
    signal = folder / "signal_model"
    signal.mkdir(parents=True)
    (signal / "signal_model_3.csv").write_bytes(
        b"supports,values\n0.0,0.125\n0.5,0.8\n1.0,1.25\n"
    )
    write_json(
        folder / "flash" / "flash.json",
        {
            "min_value_aq": 0.1,
            "max_value_aq": 0.8,
            "min_value_g": 1.0,
            "max_value_g": 1.5,
        },
    )
    return folder


def test_phase_only_reports_saved_gain_beta_and_no_fake_metric_or_anchor(
    phase, monkeypatch
):
    original = Path.open
    opened = []

    def bounded_only(path, *args, **kwargs):
        opened.append(path)
        assert path == phase
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", bounded_only)
    result = display.read_mass_calibration_display("phase_model", phase)
    shown = values(result)
    assert shown["Gain"] == "23.859055832362177"
    assert shown["Beta"] == "0.6513006394912089"
    assert shown["Dilution factor"] == display.NOT_IN_CALIBRATION
    assert shown["Anchor"] == display.NOT_IN_CALIBRATION
    assert shown["Saved mass-balance error"] == display.NOT_IN_CALIBRATION
    assert opened == [phase]
    assert "per-image dilution factor and per-run anchor" in " ".join(result.notes)


def test_bound_frame_values_are_shown_verbatim_without_recalculation(phase):
    frame = {
        "anchor": {"elapsed_h": 2.25, "affected_pore_volume_m3": 0.00045},
        "scalar_result": {"dilution_factor": 0.625},
        "total_mass_g": 999,
        "injected_mass_g": 1,
    }
    result = display.read_mass_calibration_display(
        "phase_model", phase, matching_result=frame
    )
    shown = values(result)
    assert shown["Dilution factor"] == "0.625"
    assert json.loads(shown["Anchor"]) == frame["anchor"]
    assert shown["Saved mass-balance error"] == display.NOT_IN_CALIBRATION


def test_no_first_frame_default_and_explicit_selected_frame_works(phase):
    frame = {"dilution_factor": 0.75, "anchor": {"elapsed_h": 2.5}}
    result = display.read_mass_calibration_display(
        "phase_model", phase, matching_result={"frames": [frame]}
    )
    assert values(result)["Dilution factor"] == display.NOT_IN_CALIBRATION
    selected = display.read_mass_calibration_display(
        "phase_model", phase, matching_result={"frames": [frame], "frame": frame}
    )
    assert values(selected)["Dilution factor"] == "0.75"


def test_declared_wrong_calibration_sha_does_not_supply_frame_values(phase):
    result = display.read_mass_calibration_display(
        "phase_model",
        {"path": str(phase), "sha256": "a" * 64},
        matching_result={
            "bindings": {"calibration_sha256": "b" * 64},
            "dilution_factor": 0.5,
        },
    )
    assert values(result)["Dilution factor"] == display.NOT_IN_CALIBRATION
    assert "another calibration" in " ".join(result.notes)


def test_saved_metric_is_copied_with_its_name_no_percent_recalculation(phase):
    result = display.read_mass_calibration_display(
        "phase_model",
        phase,
        matching_result={
            "calibration": {"metrics": {"mean_run_nmae": 0.0123, "nmae": 0.456}},
        },
    )
    assert (
        values(result)["Saved mass-balance error"]
        == "mean_run_nmae = 0.0123 (saved fraction)"
    )


def test_colour_csv_and_flash_are_current_saved_values_only(
    colour, tmp_path, monkeypatch
):
    history = write_json(
        tmp_path / "mass_calibration_optuna" / "newest" / "summary.json",
        {"mean_error_percent_best": 99},
    )
    original = Path.open
    opened = []

    def current_only(path, *args, **kwargs):
        opened.append(path)
        assert path != history
        assert path.is_relative_to(colour)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", current_only)
    result = display.read_mass_calibration_display("colour_path", colour)
    shown = values(result)
    assert json.loads(shown["Signal supports (facies 3)"]) == ["0.0", "0.5", "1.0"]
    assert json.loads(shown["Signal values (facies 3)"]) == ["0.125", "0.8", "1.25"]
    for key, expected in (
        ("min_value_aq", "0.1"),
        ("max_value_aq", "0.8"),
        ("min_value_g", "1.0"),
        ("max_value_g", "1.5"),
    ):
        assert shown["Flash " + key] == expected
    assert shown["Saved mass-balance error"] == display.NOT_REPORTED
    assert len(opened) == 2
    assert "history folders are not searched" in " ".join(result.notes)


def test_explicit_optuna_error_is_labelled_as_report_not_current_manual_error(
    colour, tmp_path
):
    report = write_json(
        tmp_path / "explicit-summary.json",
        {
            "calibration_folder": str(colour),
            "configuration": ["run.toml"],
            "finished": "2026-10-15T12:00:00",
            "saved": False,
            "mean_error_percent_best": 2.5,
            "values_best": {"3": [0, 0.7, 1.0]},
        },
    )
    result = display.read_mass_calibration_display(
        "colour_path", colour, optuna_summary_reference=report
    )
    shown = values(result)
    assert shown["Saved mass-balance error (Optuna report)"] == "2.5 %"
    assert json.loads(shown["Signal values (facies 3)"]) == ["0.125", "0.8", "1.25"]
    assert json.loads(shown["Report best signal values (facies 3)"]) == [0, 0.7, 1.0]
    assert "does not say its best values were adopted" in " ".join(result.notes)
    assert "not a new measurement" in " ".join(result.notes)


def test_report_for_other_colour_calibration_does_not_supply_error(colour, tmp_path):
    report = write_json(
        tmp_path / "other-summary.json",
        {
            "calibration_folder": str(tmp_path / "other"),
            "saved": True,
            "mean_error_percent_best": 0.0,
        },
    )
    result = display.read_mass_calibration_display(
        "colour_path", colour, optuna_summary_reference=report
    )
    assert values(result)["Saved mass-balance error"] == display.NOT_REPORTED
    assert "not bound" in " ".join(result.notes)


def test_large_json_is_bounded_and_never_follows_global_path(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_bytes(b" " * (display.MAX_JSON_BYTES + 100))
    result = display.read_mass_calibration_display("phase_model", path)
    assert values(result)["Gain"] == display.NOT_REPORTED
    assert "display limit" in " ".join(result.notes)


def test_signal_rows_are_bounded_without_changing_flash(colour):
    path = colour / "signal_model" / "signal_model_3.csv"
    path.write_bytes(
        b"supports,values\n" + b"0.5,1.0\n" * (display.MAX_SIGNAL_ROWS + 1)
    )
    result = display.read_mass_calibration_display("colour_path", colour)
    assert values(result)["Signal values by facies"] == display.NOT_REPORTED
    assert values(result)["Flash min_value_aq"] == "0.1"
    assert "row limit" in " ".join(result.notes)


def test_relative_references_use_selected_series_and_no_missing_paths_created(
    tmp_path, phase
):
    before = set(tmp_path.iterdir())
    result = display.read_mass_calibration_display(
        "phase_model", phase.name, series_root=phase.parent
    )
    assert values(result)["Gain"] != display.NOT_IN_CALIBRATION
    missing = display.read_mass_calibration_display("colour_path", tmp_path / "missing")
    assert values(missing)["Saved mass-balance error"] == display.NOT_REPORTED
    assert set(tmp_path.iterdir()) == before
    rejected = display.read_mass_calibration_display(
        "phase_model", "some-relative.json"
    )
    assert "relative reference" in " ".join(rejected.notes)


def test_widget_both_modes_disabled_all_table_cells_readonly_and_render_no_io(qtbot):
    panel = display.MassCalibrationDisplay()
    qtbot.addWidget(panel)
    assert not panel.manual_button.isEnabled()
    assert not panel.automatic_button.isEnabled()
    assert not panel.manual_button.isChecked()
    assert not panel.automatic_button.isChecked()
    assert "Jobs" in panel.mode_notice.text()
    assert (
        panel.parameters.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    )
    content = display.SavedMassCalibrationData(
        (("Gain", "23.5", "saved.json"),), "Explicit reference", ("No fitting.",)
    )
    panel.set_display(content)
    assert panel.source.text() == "Explicit reference"
    assert panel.parameters.item(0, 1).text() == "23.5"
    assert not panel.parameters.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    assert (
        not panel.manual_button.isEnabled() and not panel.automatic_button.isEnabled()
    )


def test_unknown_method_does_not_choose_or_open_files(monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("No method choice may read any file")

    monkeypatch.setattr(Path, "open", fail)
    value = display.read_mass_calibration_display("", "ignored.json")
    assert value.source_label == "Method: not chosen"
