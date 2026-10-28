"""Pressure and temperature source per run: synthetic series, real readers, no writes.

The rows come from DarSIA's own [protocols] reader and pressure-temperature
protocol reader. The Mass calibration page and Calibration review show them
read only.
"""

import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from darsia.fluidflower import pressure_source as source
from darsia.fluidflower.mass_calibration_display import MassCalibrationDisplay
from darsia.fluidflower.series_method import MethodSelection, save_series_method

from .test_native_series_method import method_window  # noqa: F401

HEADER = "datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,temperature_gradient_celsius\n"
RUNS = ("a1", "a2", "a3", "a4", "a5", "a6")


def _files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _series(root, runs=RUNS, series_id="series-p"):
    """A local series whose runs cover every source case; a6 has no run file."""
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "series.toml", f'schema_version = 1\nseries_id = "{series_id}"\nname = "{series_id}"\n'
           "[runs]\nids = [" + ", ".join(f'"{run}"' for run in runs) + "]\n")
    protocols = root / "protocols"
    _write(protocols / "a1" / "pressure_temperature_protocol.csv", HEADER
           + "2022-10-04 16:00:00,0.98,20.0,0.0,0.0\n"
           + "2022-10-04 17:00:00,0.99,21.0,0.0,0.0\n"
           + "2022-10-04 18:00:00,0.985,20.5,0.0,0.0\n")
    _write(protocols / "a2" / "pressure_temperature_protocol.csv", HEADER
           + "2022-11-01 10:00:00,0.998798,23.0,0.0,0.0\n"
           + "2023-01-01 10:00:00,0.998798,23.0,0.0,0.0\n")
    _write(protocols / "a5" / "pressure_temperature_protocol.csv",
           "datetime,pressure_bar,temperature_celsius\n2022-10-04 16:00:00,1.0,20.0\n")
    merged = root / "config" / "merged"
    _write(merged / "a1.toml", '[protocols]\npressure_temperature = "protocols/a1/pressure_temperature_protocol.csv"\n'
           'pressure_temperature_mode = "Load from CSV"\n')
    # No mode: DarSIA's default "constant" with 1.013 bar and 20 °C.
    _write(merged / "a2.toml", '[protocols]\npressure_temperature = "protocols/a2/pressure_temperature_protocol.csv"\n')
    _write(merged / "a3.toml", '[protocols]\npressure_temperature = "protocols/a3/pressure_temperature_protocol.csv"\n'
           'pressure_temperature_mode = "constant"\npressure_bar = 1.013\ntemperature_celsius = 20.0\n')
    _write(merged / "a4.toml", '[protocols]\nimaging_mode = "exif"\n')
    _write(merged / "a5.toml", '[protocols]\npressure_temperature = "protocols/a5/pressure_temperature_protocol.csv"\n'
           'pressure_temperature_mode = "Load from CSV"\n')
    return root


@pytest.fixture
def series(tmp_path):
    return _series(tmp_path / "series-p")


def _rows(sources):
    return {row.run: row for row in sources.rows}


def test_every_run_shows_its_configured_source_and_a_short_status(series):
    before = _files(series)
    sources = source.series_pressure_sources(series, method="phase_model", calibration=None)
    rows = _rows(sources)
    assert [row.run for row in sources.rows] == list(RUNS)
    a1 = rows["a1"]
    assert a1.source == "a1/pressure_temperature_protocol.csv"
    assert a1.values == "0.98–0.99 bar · 20–21 °C · gradients 0"
    assert a1.span == "3 rows · 2022-10-04 16:00:00 to 2022-10-04 18:00:00"
    assert a1.status == "Ready" and a1.setup == ""
    assert "Load from CSV" in a1.detail
    # The default "constant" mode would write 1.013 bar and 20 °C; the file says otherwise.
    a2 = rows["a2"]
    assert a2.values == "0.998798 bar · 23 °C · gradients 0"
    assert a2.status == "Ready" and a2.setup == "1.013 bar · 20 °C"
    assert rows["a3"].source == "Constant (file not written)"
    assert rows["a3"].values == "1.013 bar · 20 °C" and rows["a3"].status == "Not written yet"
    assert rows["a4"].source == "Missing" and rows["a4"].status == "No file configured"
    assert rows["a5"].status == "Not readable"
    assert "Pressure_Gradient_Bar" in rows["a5"].detail  # DarSIA's own reader message.
    assert rows["a6"].source == "No run configuration"
    assert rows["a6"].status == source.NOT_RECORDED and "does not exist" in rows["a6"].detail
    assert {row.fit for row in sources.rows} == {"No calibration referenced"}
    notes = " ".join(sources.notes)
    assert source.PHASE_MAP_NOTE in sources.notes and source.DIRECT_COLOUR_NOTE in sources.notes
    assert "1 of 6 runs: pressure_temperature_mode is 'constant'" in notes
    assert _files(series) == before


def test_constant_file_that_matches_the_setup_constant_has_no_setup_note(tmp_path):
    root = _series(tmp_path / "s", runs=("a2",))
    _write(root / "protocols" / "a2" / "pressure_temperature_protocol.csv",
           HEADER + "2022-11-01 10:00:00,1.013,20.0,0.0,0.0\n")
    row = source.series_pressure_sources(root).rows[0]
    assert row.status == "Ready" and row.setup == ""
    assert row.values == "1.013 bar · 20 °C · gradients 0"


def test_rows_out_of_time_order_and_gradients_are_reported(tmp_path):
    root = _series(tmp_path / "s", runs=("a1",))
    _write(root / "protocols" / "a1" / "pressure_temperature_protocol.csv", HEADER
           + "2022-10-04 18:00:00,0.98,20.0,0.1,0.0\n"
           + "2022-10-04 16:00:00,0.99,21.0,0.2,0.0\n")
    row = source.series_pressure_sources(root).rows[0]
    assert row.status == "Rows not in time order"
    assert row.values.endswith("gradients not 0")
    assert "pressure_gradient_bar 0.1–0.2" in row.detail


def test_registered_merged_file_is_the_one_read(tmp_path):
    root = _series(tmp_path / "s", runs=("a1",))
    _write(root / "series.toml", 'schema_version = 1\nseries_id = "s"\n[runs]\nids = ["a1"]\n'
           '[runs.merged_files]\na1 = "config/other/a1.toml"\n')
    _write(root / "config" / "other" / "a1.toml", "[protocols]\n")
    row = source.series_pressure_sources(root).rows[0]
    assert row.source == "Missing" and "config" in row.detail and "other" in row.detail


def test_iterator_lists_every_run_first_then_reads_one_run_per_step(series):
    snapshots = list(source.iter_series_pressure_sources(series))
    assert len(snapshots) == len(RUNS) + 1
    assert [row.status for row in snapshots[0].rows] == [source.READING] * len(RUNS)
    assert not snapshots[0].complete and not snapshots[-2].complete and snapshots[-1].complete
    assert sum(row.status != source.READING for row in snapshots[3].rows) == 3
    assert snapshots[-1] == source.series_pressure_sources(series)
    empty = series.parent / "empty"
    empty.mkdir()
    (empty / "series.toml").write_text("schema_version = 1\n[runs]\nids = []\n", encoding="utf-8")
    only = list(source.iter_series_pressure_sources(empty))
    assert len(only) == 1 and only[0].complete and only[0].rows == ()
    assert only[0].notes[0] == "No runs are registered in series.toml."


@pytest.mark.parametrize(
    "document, expected",
    [
        ({"parameters": {"gain": 23.8}, "calibration_files": {"g": {"path": "x/pressure_temperature.csv"}}},
         source.NOT_RECORDED),
        ({"chemistry": {"pressure_bar": 1.0}}, "chemistry.pressure_bar = 1.0"),
        ({"runs": [{"atmosphere_series": {"rows": []}}]}, "Recorded under runs[0].atmosphere_series"),
    ],
)
def test_the_fit_column_reports_only_what_the_calibration_records(tmp_path, document, expected):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert source.calibration_pressure("phase_model", "calibration.json", tmp_path).text == expected


def test_the_fit_column_without_a_readable_reference(tmp_path):
    assert source.calibration_pressure("", "x.json", tmp_path).text == "No method chosen"
    assert source.calibration_pressure("phase_model", "", tmp_path).text == "No calibration referenced"
    assert source.calibration_pressure("phase_model", "absent.json", tmp_path).text == "Calibration not readable"
    colour = tmp_path / "colour"
    (colour / "signal_model").mkdir(parents=True)
    assert source.calibration_pressure("colour_path", "colour", tmp_path).text == source.NOT_RECORDED
    (colour / "flash").mkdir()
    (colour / "flash" / "flash.json").write_text('{"min_value_aq": 0.1}', encoding="utf-8")
    fit = source.calibration_pressure("colour_path", {"path": str(colour)})
    assert fit.text == source.NOT_RECORDED and "flash.json" in fit.detail
    assert source.recorded_pressure_field({"a": [{"b": {"Temperature_C": 20}}]}) == ("a[0].b.Temperature_C", 20)


def test_one_line_for_a_run_page():
    row = source.PressureSourceRow("ac28", "ac28/pt.csv", "1 bar · 20 °C · gradients 0", "2 rows · t0 to t1",
                                   "Ready", "Not recorded", "")
    line = source.pressure_line(row, "ac28")
    assert line.startswith("Pressure and temperature for AC28: ac28/pt.csv · 1 bar")
    assert "— Ready. In the fit: Not recorded." in line and "Phase-map jobs" in line
    assert source.pressure_line(None) == "Pressure and temperature: choose a run."
    assert "Not recorded (not a run of this series)" in source.pressure_line(None, "zz")


def test_display_shows_rows_read_only_and_hides_without_them(qtbot, series, monkeypatch):
    panel = MassCalibrationDisplay()
    qtbot.addWidget(panel)
    assert panel.pressure_box.isHidden()
    sources = source.series_pressure_sources(series)

    def no_io(*_args, **_kwargs):
        pytest.fail("Rendering rows reads no file")

    monkeypatch.setattr(Path, "open", no_io)
    panel.set_pressure_sources(sources)
    monkeypatch.undo()
    assert not panel.pressure_box.isHidden()
    table = panel.pressure_table
    assert table.rowCount() == len(RUNS)
    assert [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())] == list(
        ("Run", "Source", "Pressure · temperature", "Time span", "Status", "In the fit"))
    assert table.item(0, 0).text() == "A1" and table.item(0, 4).text() == "Ready"
    assert not table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    assert "Run configuration:" in table.item(0, 1).toolTip()
    assert "Phase-map method: jobs do not read these files." in panel.pressure_notes.text()
    assert "whole series (6 runs)" in panel.pressure_title.text()
    panel.set_pressure_sources(None)
    assert panel.pressure_box.isHidden()


def test_display_follows_the_reader_one_run_per_tick_and_a_new_call_replaces_it(qtbot, series):
    panel = MassCalibrationDisplay()
    qtbot.addWidget(panel)
    panel.follow_pressure_sources(source.iter_series_pressure_sources(series))
    assert not panel.pressure_complete
    assert panel.pressure_table.item(5, 4).text() == source.READING
    qtbot.waitUntil(lambda: panel.pressure_complete, timeout=10000)
    assert panel.pressure_table.item(5, 4).text() == source.NOT_RECORDED
    single = _series(series.parent / "single", runs=("a1",))
    panel.follow_pressure_sources(source.iter_series_pressure_sources(series))
    panel.follow_pressure_sources(source.iter_series_pressure_sources(single))
    qtbot.waitUntil(lambda: panel.pressure_complete, timeout=10000)
    assert panel.pressure_table.rowCount() == 1
    panel.follow_pressure_sources(None)
    assert panel.pressure_box.isHidden() and panel.pressure_complete


def test_mass_calibration_page_shows_the_whole_series_table(qtbot, method_window):
    window = method_window
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    root = window.phase_model_workspace.series_roots["series-a"]
    _series(root, series_id="series-a")
    (root / "frozen.json").write_text('{"parameters": {"gain": 23.859, "beta": 0.6513}}', encoding="utf-8")
    save_series_method(root, MethodSelection(active="phase_model", model_version="v1",
                                             mass_calibration="frozen.json"))
    window._refresh_method_state(force=True)
    before = _files(root)
    window._select_row(("phase_model", "mass_from_phase"))
    panel = window.mass_calibration_display
    assert window.content_stack.currentWidget() is panel
    qtbot.waitUntil(lambda: panel.pressure_complete, timeout=10000)
    assert panel.pressure_table.rowCount() == len(RUNS)
    assert panel.pressure_table.item(0, 5).text() == source.NOT_RECORDED
    assert panel.pressure_table.item(1, 1).text() == "a2/pressure_temperature_protocol.csv"
    window._select_row(("calibration", "mass"))
    qtbot.waitUntil(lambda: panel.pressure_complete, timeout=10000)
    assert panel.pressure_table.item(0, 5).text() == "No calibration referenced"
    assert _files(root) == before


def test_calibration_review_shows_the_viewed_runs_source_in_one_line(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module
    from .calibration_review_fixture import add_saved_results

    root = _series(tmp_path / "series-p")
    monkeypatch.setattr(module, "configured_series", lambda: (("series-p", "series-p", root),))
    before = _files(root)
    page = module.MassAnalysisWorkspace(series="series-p", project_root=root)
    qtbot.addWidget(page)
    try:
        experiments = page.saved_results.view_experiment
        add_saved_results(page, root, ("a1", "a4"))
        before = _files(root)
        page.show_view("review")
        experiments.setCurrentIndex(0)
        assert not page.pressure_line.isHidden()
        text = page.pressure_line.text()
        assert text.startswith("Pressure and temperature for A1: a1/pressure_temperature_protocol.csv")
        assert "— Ready." in text and "In the fit: No calibration referenced." in text
        experiments.setCurrentIndex(1)
        assert "for A4: Missing — No file configured." in page.pressure_line.text()
        page.show_view("results")
        assert page.pressure_line.isHidden()
        assert _files(root) == before
    finally:
        page._stop_lookups()
