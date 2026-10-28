"""The read-only running pressure view: its readers, its window and its Protocols button.

Offscreen Qt on synthetic files in tmp_path. Nothing is fitted, adopted,
started or written; settings go to an INI file in tmp_path.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QMessageBox, QScrollArea

from darsia.fluidflower import running_pressure_view as rpv

from .test_native_sidebar import native_window  # noqa: F401

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
FILLED = "provisional_fill_for_test"


def _stamp(minute):
    return START + timedelta(minutes=minute)


def _row(minute, quality=rpv.OBSERVED):
    pressure = 1.0 + minute / 10000
    return {
        "utc": _stamp(minute).isoformat(),
        "time_hours": minute / 60,
        "pressure_bar": pressure,
        "temperature_c": 22.0,
        "station_pressure_hpa": pressure * 1000,
        "quality": quality,
    }


def _gap(first, last, missing, filled):
    return {
        "first_utc": _stamp(first).isoformat(),
        "last_utc": _stamp(last).isoformat(),
        "missing_stamp_count": missing,
        "filled_count": filled,
        "resolved_provisionally": filled == missing,
    }


def _descriptor(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return _descriptor(path)


def _series_document(series, run, rows, gaps, *, complete):
    return {
        "format": rpv.SERIES_FORMAT,
        "series": series,
        "run": run,
        "imaging_first_utc": _stamp(5).isoformat(),
        "imaging_last_utc": _stamp(95).isoformat(),
        "rows": rows,
        "gaps": gaps,
        "complete": complete,
        "adopted": False,
    }


def _index(root):
    """Two runs: one gap filled provisionally (a row not observed), one gap open."""

    filled = [_gap(50, 50, 1, 1)]
    first = _write(
        root / "SERIES_A" / "run1" / "atmosphere_series.json",
        _series_document(
            "SERIES_A",
            "run1",
            [_row(minute, FILLED if minute == 50 else rpv.OBSERVED) for minute in range(0, 101, 10)],
            filled,
            complete=True,
        ),
    )
    open_gap = [_gap(50, 70, 3, 0)]
    second = _write(
        root / "SERIES_B" / "run2" / "atmosphere_series.json",
        _series_document(
            "SERIES_B",
            "run2",
            [_row(minute) for minute in (0, 10, 20, 30, 40, 80, 90, 100)],
            open_gap,
            complete=False,
        ),
    )
    index = {
        "format": rpv.INDEX_FORMAT,
        "runs": {
            "SERIES_A": {
                "run1": {
                    "run": "run1",
                    "complete": True,
                    "row_count": 11,
                    "provisional_gap_default": True,
                    "gaps": filled,
                    "atmosphere_series": first,
                    "adopted": False,
                }
            },
            "SERIES_B": {
                "run2": {
                    "run": "run2",
                    "complete": False,
                    "row_count": 8,
                    "provisional_gap_default": False,
                    "gaps": open_gap,
                    "atmosphere_series": second,
                    "adopted": False,
                }
            },
        },
        "adopted": False,
        "complete": False,
    }
    path = root / "index.json"
    path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def _series(root):
    """Both synthetic runs, read through the index."""

    return [
        rpv.read_running_pressure_series(row.atmosphere_series)
        for row in rpv.read_running_pressure_index(_index(root))
    ]


def _files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _settings(monkeypatch, tmp_path):
    """The view's settings in an INI file in tmp_path."""

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(rpv, "QSettings", lambda *args: settings)
    return settings


@pytest.fixture
def index(tmp_path):
    return _index(tmp_path / "data")


@pytest.mark.parametrize("format_id", [rpv.INDEX_FORMAT, rpv.DATASET_INDEX_FORMAT])
def test_the_index_reader_refuses_another_format(tmp_path, index, format_id):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": "another-format-v1", "runs": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a running pressure index"):
        rpv.read_running_pressure_index(other)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON file"):
        rpv.read_running_pressure_index(broken)
    document = json.loads(index.read_text(encoding="utf-8"))
    document["format"] = format_id
    del document["runs"]["SERIES_A"]["run1"]["complete"]
    lacking = tmp_path / "lacking.json"
    lacking.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="'complete' must be true or false"):
        rpv.read_running_pressure_index(lacking)


def test_the_index_reader_returns_one_row_per_run_in_index_order(index):
    rows = rpv.read_running_pressure_index(index)
    assert [(row.series, row.run) for row in rows] == [("SERIES_A", "run1"), ("SERIES_B", "run2")]
    first, second = rows
    assert (first.complete, first.row_count, first.gap_count) == (True, 11, 1)
    assert (first.provisional_gap_default, first.adopted) == (True, False)
    assert first.gaps == (rpv.PressureGap(_stamp(50), _stamp(50), 1, 1, True),)
    assert (second.complete, second.row_count, second.gap_count) == (False, 8, 1)
    assert second.gaps[0].last_utc == _stamp(70) and not second.gaps[0].resolved_provisionally
    assert set(second.atmosphere_series) == {"path", "sha256", "size_bytes"}


@pytest.mark.parametrize("format_id", [rpv.SERIES_FORMAT, rpv.DATASET_SERIES_FORMAT])
def test_the_series_reader_checks_size_and_sha_before_parsing(tmp_path, index, format_id):
    descriptor = rpv.read_running_pressure_index(index)[0].atmosphere_series
    path = tmp_path / "data" / "SERIES_A" / "run1" / "atmosphere_series.json"
    if format_id != rpv.SERIES_FORMAT:
        document = json.loads(path.read_bytes())
        descriptor = _write(path, dict(document, format=format_id))
    original = path.read_bytes()
    assert len(rpv.read_running_pressure_series(descriptor).rows) == 11
    # Same size, not JSON: refused for its SHA-256, so it was never parsed.
    path.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="SHA-256") as refused:
        rpv.read_running_pressure_series(descriptor)
    assert "JSON" not in str(refused.value)
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="bytes, the index records"):
        rpv.read_running_pressure_series(descriptor)
    path.unlink()
    with pytest.raises(OSError):
        rpv.read_running_pressure_series(descriptor)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="not bound by path, SHA-256 and size"):
        rpv.read_running_pressure_series({"path": str(path), "size_bytes": len(original)})
    with pytest.raises(ValueError, match="belongs to SERIES_A run1"):
        rpv.read_running_pressure_series(descriptor, series="SERIES_A", run="another")
    # A verified file must still be a running pressure file in time order.
    document = json.loads(original)
    wrong = tmp_path / "wrong.json"
    with pytest.raises(ValueError, match="not a running pressure file"):
        rpv.read_running_pressure_series(_write(wrong, dict(document, format="another-format-v1")))
    with pytest.raises(ValueError, match="not in increasing UTC time"):
        rpv.read_running_pressure_series(_write(wrong, dict(document, rows=document["rows"][::-1])))
    naive = dict(document, rows=[dict(document["rows"][0], utc="2024-01-01T00:00:00")])
    with pytest.raises(ValueError, match="not a time with an offset"):
        rpv.read_running_pressure_series(_write(wrong, naive))
    # Values beyond a float or a date are refused as values, never as crashes.
    early = dict(document, rows=[dict(document["rows"][0], utc="0001-01-01T00:00:00+01:00")])
    with pytest.raises(ValueError, match="not a time with an offset"):
        rpv.read_running_pressure_series(_write(wrong, early))
    giant = json.dumps(dict(document, rows=[document["rows"][0]])).replace(
        '"pressure_bar": 1.0,', '"pressure_bar": 1' + "0" * 400 + ",", 1
    )
    wrong.write_text(giant, encoding="utf-8")
    with pytest.raises(ValueError, match="'pressure_bar' must be a finite number"):
        rpv.read_running_pressure_series(_descriptor(wrong))


def test_the_series_reader_returns_rows_gaps_quality_and_imaging_window(index):
    first = rpv.read_running_pressure_index(index)[0]
    series = rpv.read_running_pressure_series(first.atmosphere_series, series="SERIES_A", run="run1")
    assert (series.series, series.run, series.complete, series.reason) == ("SERIES_A", "run1", True, "")
    assert series.sha256 == first.atmosphere_series["sha256"]
    assert series.size_bytes == first.atmosphere_series["size_bytes"]
    assert [row.quality for row in series.rows].count(FILLED) == 1
    filled = series.rows[5]
    assert (filled.utc, filled.quality, filled.temperature_c) == (_stamp(50), FILLED, 22.0)
    assert filled.pressure_bar == pytest.approx(1.005) and filled.time_hours == pytest.approx(50 / 60)
    assert filled.station_pressure_hpa == pytest.approx(1005.0)
    assert series.gaps == first.gaps
    assert (series.imaging_first_utc, series.imaging_last_utc) == (_stamp(5), _stamp(95))


def test_a_file_without_rows_keeps_its_reason(tmp_path):
    descriptor = _write(
        tmp_path / "empty.json",
        {
            "format": rpv.SERIES_FORMAT,
            "series": "SERIES_C",
            "run": "run3",
            "complete": False,
            "rows": [],
            "reason": "No source export yet",
        },
    )
    series = rpv.read_running_pressure_series(descriptor)
    assert series.rows == () and series.gaps == () and series.imaging_first_utc is None
    assert series.reason == "No source export yet" and not series.complete


def test_plot_elements_mark_the_imaging_window_gaps_and_rows_not_observed(tmp_path):
    first, second = _series(tmp_path)
    elements = rpv.plot_elements(first)
    assert elements.times[0] == _stamp(0) and elements.pressures[-1] == pytest.approx(1.01)
    # A filled gap keeps the line; an open gap stops it.
    assert elements.segments == ((0, 11),)
    assert rpv.plot_elements(second).segments == ((0, 5), (5, 8))
    assert elements.imaging == (_stamp(5), _stamp(95))
    assert elements.gaps == ((_stamp(50), _stamp(50)),)
    assert elements.not_observed == {FILLED: ((_stamp(50), pytest.approx(1.005)),)}
    assert rpv.plot_elements(second).not_observed == {}


def test_imaging_window_pressure_and_the_details_line(tmp_path):
    first = _series(tmp_path)[0]
    # Minutes 10 to 90 lie inside 00:05-01:35.
    assert rpv.imaging_window_pressure(first) == (pytest.approx(1.001), pytest.approx(1.009), 9)
    details = rpv.series_details(first, 11)
    assert f"File: {first.path}" in details and first.sha256 in details
    assert "11 rows, 1 not observed" in details and "index records" not in details
    assert "2024-01-01 00:05:00 to 2024-01-01 01:35:00 UTC" in details
    assert "min 1.00100 bar, max 1.00900 bar (9 rows)" in details
    assert "(the index records 12)" in rpv.series_details(first, 12)
    unwindowed = replace(first, imaging_first_utc=None)
    assert rpv.imaging_window_pressure(unwindowed) is None
    assert "No imaging window is recorded." in rpv.series_details(unwindowed)
    later = replace(first, imaging_first_utc=_stamp(200), imaging_last_utc=_stamp(300))
    assert rpv.imaging_window_pressure(later) is None
    assert "no row inside it" in rpv.series_details(later)
    empty = replace(first, rows=(), reason="No source export yet")
    assert "0 rows" in rpv.series_details(empty)
    assert "No rows: No source export yet" in rpv.series_details(empty)


def test_render_pressure_plot_draws_rgba_pixels_of_the_asked_size(tmp_path):
    first, second = _series(tmp_path)
    pixels = rpv.render_pressure_plot(first, size_px=(640, 300))
    assert pixels.shape == (300, 640, 4) and pixels.dtype == np.uint8
    assert pixels[..., :3].std() > 0
    assert rpv.render_pressure_plot(second, size_px=(640, 300), scale=2.0).shape == (600, 1280, 4)
    # Never smaller than the smallest plot.
    assert rpv.render_pressure_plot(first, size_px=(10, 10)).shape[:2] == rpv.MIN_PLOT_PX[::-1]
    empty = replace(first, rows=(), gaps=(), reason="No source export yet")
    assert rpv.render_pressure_plot(empty, size_px=(640, 300)).shape == (300, 640, 4)


def test_the_window_lists_the_runs_and_plots_the_selected_one(qtbot, tmp_path, index):
    before = _files(tmp_path)
    dialog = rpv.RunningPressureDialog(index)
    qtbot.addWidget(dialog)
    dialog.show()
    table = dialog.table
    cells = [[table.item(row, column).text() for column in range(table.columnCount())] for row in range(2)]
    # The first run is read and plotted when the window opens.
    assert cells == [
        ["SERIES_A", "run1", "yes", "11", "1", "yes", "no", "verified"],
        ["SERIES_B", "run2", "no", "8", "1", "no", "no", ""],
    ]
    assert dialog.series.run == "run1" and not dialog.plot.pixmap().isNull()
    assert dialog.series.sha256 in dialog.details.text()
    assert "2 runs in 2 series, 1 complete." in dialog.note.text() and str(index) in dialog.note.text()
    table.selectRow(1)
    assert dialog.current == 1 and dialog.series.run == "run2"
    assert dialog.file_state(1) == "verified" and "8 rows" in dialog.details.text()
    assert not dialog.plot.pixmap().isNull()
    # One window without a scroll area of its own, and a modest minimum size.
    assert dialog.findChildren(QScrollArea) == []
    minimum = dialog.minimumSizeHint()
    assert minimum.width() <= 900 and minimum.height() <= 700
    assert dialog.width() <= rpv.WINDOW_PX[0] and dialog.height() <= rpv.WINDOW_PX[1]
    assert _files(tmp_path) == before


@pytest.mark.parametrize("index_format", [rpv.INDEX_FORMAT, rpv.DATASET_INDEX_FORMAT])
@pytest.mark.parametrize("series_format", [rpv.SERIES_FORMAT, rpv.DATASET_SERIES_FORMAT])
def test_generic_and_historical_datasets_share_the_picker_and_verified_reader(
    qtbot, tmp_path, monkeypatch, index_format, series_format
):
    root = tmp_path / "data"
    source_index = _index(root)
    document = json.loads(source_index.read_bytes())
    document["format"] = index_format
    for supplier, entries in zip(("GFI", "Pressure logger"), document["runs"].values()):
        for entry in entries.values():
            path = Path(entry["atmosphere_series"]["path"])
            pressure = json.loads(path.read_bytes())
            pressure.update(format=series_format, supplier=supplier)
            # Station-specific raw values are optional. The pressure_bar field
            # and explicit UTC stamps are the same contract for every supplier.
            for row in pressure["rows"]:
                row.pop("station_pressure_hpa")
            entry["atmosphere_series"] = _write(path, pressure)
    chosen = root / "pressure_samples.json"
    _write(chosen, document)
    before = _files(root)
    _settings(monkeypatch, tmp_path)
    picker = []

    def choose(parent, title, start, filters):
        picker.append((title, filters))
        return str(chosen), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose)
    dialog = rpv.open_running_pressure(None)
    qtbot.addWidget(dialog)
    assert picker == [("Pressure dataset index", "Pressure dataset index (*.json)")]
    assert dialog.windowTitle() == "Pressure datasets"
    assert dialog.choose_button.text() == "Choose dataset…"
    assert "GFI" not in dialog.note.text()
    first = dialog.series
    dialog.table.selectRow(1)
    second = dialog.series
    assert dialog.file_state(0) == dialog.file_state(1) == "verified"
    assert (first.series, first.run) == ("SERIES_A", "run1")
    assert (second.series, second.run) == ("SERIES_B", "run2")
    assert first.rows[1].pressure_bar == second.rows[1].pressure_bar == pytest.approx(1.001)
    assert first.rows[1].utc == second.rows[1].utc == _stamp(10)
    assert all(row.station_pressure_hpa is None for row in (*first.rows, *second.rows))
    assert first.rows[5].quality == FILLED and second.gaps[0].filled_count == 0
    assert _files(root) == before


def test_a_run_whose_file_differs_from_the_index_shows_as_unreadable(qtbot, tmp_path, index):
    path = tmp_path / "data" / "SERIES_B" / "run2" / "atmosphere_series.json"
    path.write_bytes(path.read_bytes() + b"\n")
    dialog = rpv.RunningPressureDialog(index)
    qtbot.addWidget(dialog)
    assert dialog.file_state(0) == "verified"
    dialog.table.selectRow(1)
    assert dialog.current == 1 and not dialog.show_run(1)
    assert dialog.file_state(1) == "unreadable" and dialog.series is None
    assert "bytes, the index records" in dialog.table.item(1, 7).toolTip()
    assert "unreadable" in dialog.plot.text() and dialog.plot.pixmap().isNull()
    assert dialog.details.text().startswith(f"File: {path}\nUnreadable: ")
    # The other runs stay available.
    dialog.table.selectRow(0)
    assert dialog.series.run == "run1" and dialog.file_state(0) == "verified"


def test_the_plot_is_drawn_again_at_its_new_size(qtbot, index):
    dialog = rpv.RunningPressureDialog(index)
    qtbot.addWidget(dialog)
    dialog.resize(900, 640)
    dialog.show()

    def fitted():
        size = dialog.plot.pixmap().deviceIndependentSize()
        return abs(size.width() - dialog.plot.width()) <= 1 and abs(size.height() - dialog.plot.height()) <= 1

    qtbot.waitUntil(fitted, timeout=5000)
    dialog.resize(1100, 760)
    qtbot.waitUntil(fitted, timeout=5000)
    assert dialog.plot.redraw()
    dialog.plot.show_message("Nothing to show")
    assert not dialog.plot.redraw() and dialog.plot.text() == "Nothing to show"


def test_choosing_an_index_remembers_it_in_the_settings(qtbot, monkeypatch, tmp_path, index):
    settings = _settings(monkeypatch, tmp_path)
    starts, answers = [], [str(index)]

    def ask(parent, title, start, filters):
        starts.append(start)
        return (answers.pop(0) if answers else "", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", ask)
    dialog = rpv.open_running_pressure(None)
    qtbot.addWidget(dialog)
    assert isinstance(dialog, rpv.RunningPressureDialog) and dialog.isVisible()
    assert settings.value(rpv.INDEX_KEY) == str(index) and starts == [""]
    # A cancelled choice opens nothing and keeps the remembered index.
    assert rpv.open_running_pressure(None) is None
    assert starts[-1] == str(index) and settings.value(rpv.INDEX_KEY) == str(index)
    # An index that cannot be read is said, in a message or in the window.
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": "another-format-v1"}), encoding="utf-8")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    answers.append(str(other))
    assert rpv.open_running_pressure(None) is None
    assert warnings and warnings[0][0] == "Could not read the pressure dataset index"
    answers.append(str(other))
    assert not dialog.choose_index()
    assert "could not be read" in dialog.details.text() and dialog.table.rowCount() == 2
    answers.append(str(index))
    assert dialog.choose_index() and dialog.index_path == index


def test_the_pressure_page_has_the_button_and_it_opens_the_window(
    native_window, qtbot, monkeypatch, tmp_path, index  # noqa: F811
):
    window = native_window
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    entry = window.running_pressure
    window._select_row(("preprocessing", "protocols"))
    qtbot.waitUntil(lambda: window.protocol_sections.navigation is not None, timeout=10000)
    sections = window.protocol_sections
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    button = sections.pressure_datasets_button
    assert button.text() == "Show pressure datasets…"
    assert sections.pages["pressure_temperature"].isAncestorOf(button)
    _settings(monkeypatch, tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(index), ""))
    button.click()
    dialog = entry.dialog
    assert isinstance(dialog, rpv.RunningPressureDialog) and dialog.isVisible()
    assert dialog.parent() is window and dialog.table.rowCount() == 2
    dialog.close()
