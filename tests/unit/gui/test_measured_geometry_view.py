"""The read-only view of each run's measured geometry.

A synthetic geometry index in the builder's form: two runs with frame models
(metric rig millimetres to a 80 x 60 camera grid, y up), one with a per-frame
model and a pixel-area map, and a third run listed but not yet built. The
painting tab opens the view from step 1. Temporary folders only.
"""

import hashlib
import json

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

from darsia.fluidflower import map_placement as mp
from darsia.fluidflower import measured_geometry_view as mgv
from darsia.fluidflower import series_layer_view as view_module

from .test_painting_corrections import view  # noqa: F401

SHAPE = [60, 80]
# 0.08 px per mm; the bottom of the rig on row 58, the left wall on column 2.
FIRST_H = [[0.08, 0.0, 2.0], [0.0, -0.08, 58.0], [0.0, 0.0, 1.0]]
SECOND_H = [[0.08, 0.0, 5.0], [0.0, -0.08, 58.0], [0.0, 0.0, 1.0]]


def _index(folder):
    folder.mkdir(parents=True)
    runs = {}
    for run, matrix in (("run1", FIRST_H), ("run2", SECOND_H)):
        (folder / run).mkdir()
        model = {
            "format": mp.GEOMETRY_FORMAT,
            "run": run,
            "run_id": run,
            "reference_image": {"path": f"C:/photos/{run}.arw"},
            "grid": "camera_u",
            "shape_hw": SHAPE,
            "H_metric_mm_to_camera_u": matrix,
            "metric_frame": {"unit": "mm", "measured_width_mm": 899.0},
            "optical_offset_mm": 14.0,
            "area_factor": 1.0343865,
        }
        path = folder / run / "reference_model.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        runs[run] = {
            "run": run,
            "reference_model": {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            },
        }
    (folder / "run1" / "frames").mkdir()
    (folder / "run1" / "frames" / "case1.json").write_text("{}", encoding="utf-8")
    (folder / "run1" / "areas").mkdir()
    area = np.linspace(0.9, 1.1, 60 * 80, dtype=np.float32).reshape(60, 80)
    np.save(folder / "run1" / "areas" / "case1.npy", area)
    path = folder / "index.json"
    path.write_text(
        json.dumps(
            {
                "format": mp.GEOMETRY_FORMAT,
                "state": "PARTIAL_TEST",
                "run_order": ["run1", "run2", "run3"],
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_frame_data_bound_by_the_index_is_found_in_its_own_folder(tmp_path):
    index = _index(tmp_path / "geometry")
    frames = tmp_path / "frame_data"
    (frames / "run2" / "frames").mkdir(parents=True)
    (frames / "run2" / "areas").mkdir()
    (frames / "run2" / "frames" / "f1.json").write_text("{}", encoding="utf-8")
    (frames / "run2" / "frames" / "f2.json").write_text("{}", encoding="utf-8")
    np.save(frames / "run2" / "areas" / "f2.npy", np.ones((6, 8), dtype=np.float32))
    frame_index = frames / "index.json"
    frame_index.write_text(json.dumps({"format": "frame data", "run": "run2"}), encoding="utf-8")
    data = json.loads(index.read_text(encoding="utf-8"))
    data["runs"]["run2"]["frame_data"] = {
        "path": str(frame_index),
        "sha256": hashlib.sha256(frame_index.read_bytes()).hexdigest(),
        "size_bytes": frame_index.stat().st_size,
    }
    index.write_text(json.dumps(data), encoding="utf-8")
    geometry = mp.read_run_geometry(index)
    files = mgv.model_files(geometry["run2"])
    assert [path.name for path in files["frames"]] == ["f1.json", "f2.json"]
    assert [path.name for path in files["areas"]] == ["f2.npy"]
    # Without a binding, the folder beside the reference model.
    assert [path.name for path in mgv.model_files(geometry["run1"])["frames"]] == ["case1.json"]
    frame_index.write_text("{}", encoding="utf-8")
    try:
        mgv.model_files(geometry["run2"])
    except ValueError as error:
        assert "frame data index changed" in str(error)
    else:
        raise AssertionError("a changed frame data index was read")
    # The table says so instead of failing.
    assert mgv._cells(geometry["run2"])[-2:] == ("unreadable", "unreadable")


def _pixel(dialog, x, y):
    color = dialog.view.item.pixmap().toImage().pixelColor(x, y)
    return color.red(), color.green(), color.blue()


def test_the_frame_model_is_drawn_on_the_runs_grid(tmp_path):
    geometry = mp.read_run_geometry(_index(tmp_path / "geometry"))["run1"]
    assert mgv.measured_width_mm(geometry) == 899.0
    files = mgv.model_files(geometry)
    assert [path.name for path in files["frames"]] == ["case1.json"]
    assert [path.name for path in files["areas"]] == ["case1.npy"]
    lines = mgv.frame_lines(geometry)
    assert np.allclose(lines["frame"][0], [[2.0, 58.0], [73.92, 58.0]])
    assert len(lines["grid"]) == 16  # Every 100 mm across and up, inside 899 mm.
    drawing, factor = mgv.draw_frame_model(geometry, (60, 80))
    assert factor == 1 and drawing.shape == (60, 80, 3)
    assert tuple(drawing[58, 30]) == mgv.FRAME_COLOR
    assert tuple(drawing[42, 30]) == mgv.GRID_COLOR  # 200 mm up.
    assert tuple(drawing[46, 14]) == (40, 40, 40)
    photo = np.full((60, 80, 3), 90, dtype=np.uint8)
    drawing, _factor = mgv.draw_frame_model(geometry, (60, 80), photo)
    assert tuple(drawing[46, 14]) == (90, 90, 90) and photo[58, 30, 0] == 90
    # A full camera grid is drawn reduced.
    drawing, factor = mgv.draw_frame_model(geometry, (3080, 5472))
    assert factor == 3 and drawing.shape == (1026, 1824, 3)
    image, stats = mgv.area_map_image(files["areas"][0])
    assert image.shape == (60, 80, 3) and image.dtype == np.uint8
    assert abs(stats["min"] - 0.9) < 1e-6 and abs(stats["max"] - 1.1) < 1e-6


def test_the_view_lists_each_runs_frame_model_and_shows_its_maps(qtbot, tmp_path):
    loaded = []

    def photo(run, geometry):
        loaded.append(run)
        return np.full(tuple(geometry.shape_hw) + (3,), 90, dtype=np.uint8)

    dialog = mgv.MeasuredGeometryDialog(_index(tmp_path / "geometry"), photo_loader=photo)
    qtbot.addWidget(dialog)
    dialog.show()
    cells = [
        [dialog.table.item(row, column).text() for column in range(dialog.table.columnCount())]
        for row in range(dialog.table.rowCount())
    ]
    assert cells == [
        ["RUN1", "run1.arw", "80 x 60", "899", "14", "1.0344", "12.5000 / 12.5000", "1", "1"],
        ["RUN2", "run2.arw", "80 x 60", "899", "14", "1.0344", "12.5000 / 12.5000", "0", "0"],
    ]
    note = dialog.findChildren(type(dialog.status))[0].text()
    assert "Not yet in the index: RUN3" in note and "PARTIAL_TEST" in note
    assert dialog.legend.text().count("■") == 2
    # The first run is shown on its camera grid; then on its photograph.
    assert dialog.current == "run1" and dialog.view.item.pixmap().width() == 80
    assert _pixel(dialog, 14, 46) == (40, 40, 40)
    assert dialog.show_photograph() and loaded == ["run1"]
    assert _pixel(dialog, 14, 46) == (90, 90, 90) and _pixel(dialog, 30, 58) == mgv.FRAME_COLOR
    assert dialog.area_choice.count() == 1 and dialog.area_button.isEnabled()
    assert dialog.show_area_map() and "min 0.9" in dialog.status.text()
    dialog.table.selectRow(1)
    assert dialog.current == "run2" and not dialog.area_button.isEnabled()
    # A photograph that cannot be read is said; nothing else changes.
    dialog.photo_loader = lambda run, geometry: (_ for _ in ()).throw(OSError("offline"))
    assert not dialog.show_photograph() and "could not be read: offline" in dialog.status.text()


def test_step_1_opens_the_view_from_a_remembered_index(view, monkeypatch, tmp_path):
    index = _index(tmp_path / "geometry")
    settings = view_module.QSettings(
        str(tmp_path / "settings.ini"), view_module.QSettings.Format.IniFormat
    )
    monkeypatch.setattr(view_module, "QSettings", lambda *args: settings)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(index), ""))
    # The button stays usable where the rig panel is disabled (a corrected workspace).
    view.rig_panel.setEnabled(False)
    assert view.measured_geometry_button.isEnabled()
    view.measured_geometry_button.click()
    dialog = view._geometry_dialog
    assert isinstance(dialog, mgv.MeasuredGeometryDialog) and dialog.isVisible()
    assert dialog.runs == ["run1", "run2"]
    assert settings.value("placement/geometry_index") == str(index)
    dialog.close()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert view.show_measured_geometry(bad) is None
    assert warnings and warnings[0][0] == "Could not read the geometry index"
