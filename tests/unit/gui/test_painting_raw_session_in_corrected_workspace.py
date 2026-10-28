"""Layer painting tab: a raw-photo session in a corrected-frame workspace.

When the project's geometry is in the saved corrected rig frame, the painting
tab shows the saved corrected maps read-only. A session whose marks were
painted on the raw reference photo must still open on that raw photo, with
painting, the proposal and the background photo working, and must not be
exported into the corrected project. A synthetic 80 x 60 photograph, an
80 x 50 corrected grid and a project in a temporary folder; no real series,
rig model or NAS input is used.
"""

import hashlib
import json
import threading

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMenu

from darsia.fluidflower import geometry_workspace_frame as workspace_frame
from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import photograph_io
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.native import raw_photograph
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerPainterView

RAW_SIZE = (80, 60)
CORRECTED_SHAPE = (50, 80)
# Rows beyond the corrected grid: these marks only fit the raw photo.
RAW_POLYGON = [(5.0, 5.0), (5.0, 35.0), (58.0, 35.0), (58.0, 5.0)]
CORRECTED_REFUSAL = (
    "Corrected-frame production export and Build run model are not qualified yet."
)


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


def _field(path, array):
    np.savez(path, array=array)
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "shape": list(array.shape),
    }


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Keep remembered choices in a private INI file, not the user's registry."""

    path = str(tmp_path / "painting.ini")
    monkeypatch.setattr(
        slv, "QSettings", lambda *args: QSettings(path, QSettings.Format.IniFormat)
    )


@pytest.fixture
def paths(tmp_path):
    folder = tmp_path / "series"
    folder.mkdir()
    baseline = folder / "raw.png"
    Image.new("RGB", RAW_SIZE, (30, 40, 50)).save(baseline)
    before = folder / "before_indicator.png"
    Image.new("RGB", RAW_SIZE, (200, 160, 40)).save(before)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    return {"baseline": baseline, "before": before, "config": config}


@pytest.fixture
def project(tmp_path):
    """A project whose current geometry revision is in the corrected frame."""

    root = tmp_path / "project"
    rig = tmp_path / "rig"
    rig.mkdir()
    labels = np.zeros(CORRECTED_SHAPE, dtype=np.uint8)
    labels[25:, :] = 1
    fields = {
        "shape_corrected_baseline": _field(
            rig / "shape_corrected_baseline.npz",
            np.full((*CORRECTED_SHAPE, 3), 0.5, dtype=np.float32),
        ),
        "labels": _field(rig / "labels.npz", labels),
        "facies": _field(rig / "facies.npz", labels.copy()),
        "depth": _field(
            rig / "depth.npz", np.full(CORRECTED_SHAPE, 0.02, dtype=np.float64)
        ),
    }
    correction = _field(rig / "shape_correction.npz", np.zeros(2, dtype=np.float32))
    config = {
        "frame": "rig-corrected",
        "width_m": 0.9,
        "height_m": 0.49,
        "frame_binding": {
            "format": workspace_frame.FRAME_BINDING_FORMAT,
            "source_reference_run": "historical",
            "display_reference_run": "ac52",
            "reference_shape_corrections": [correction],
            "runs": {
                "ac52": {
                    "fields": fields,
                    "configured_depth_map": {"path": str(rig / "depth_map.npz")},
                }
            },
        },
    }
    current = root / "overrides" / "geometry_facies" / "current.json"
    current.parent.mkdir(parents=True)
    current.write_text(
        json.dumps({"frame": "rig-corrected", "configuration": config}),
        encoding="utf-8",
    )
    return _Project(root)


def _save_session(project, image_path, size, polygons):
    session = lp.PaintingSession.default("ac52", str(image_path), size)
    session.polygons = [lp.Polygon(2, list(points)) for points in polygons]
    lp.save_painting_session(project, session, actor="test")
    return lp.session_path(project, "ac52")


@pytest.fixture
def make_view(qtbot, monkeypatch, settings, paths, project):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    reference = SeriesRunInputs(
        "ac52", paths["config"], baseline_image=paths["baseline"], width_m=0.9, height_m=0.49
    )

    def make():
        widget = LayerPainterView(project, lambda: reference)
        qtbot.addWidget(widget)
        widget.resize(1400, 900)
        widget.show()
        qtbot.waitExposed(widget)
        return widget

    return make


def _corrected_path(project):
    current = json.loads(
        (project.root / "overrides" / "geometry_facies" / "current.json").read_text(
            encoding="utf-8"
        )
    )
    fields = current["configuration"]["frame_binding"]["runs"]["ac52"]["fields"]
    return fields["shape_corrected_baseline"]["path"]


def test_without_a_raw_session_the_corrected_view_is_unchanged(make_view, project):
    view = make_view()
    assert view.image is not None
    assert view.image.image_size == (CORRECTED_SHAPE[1], CORRECTED_SHAPE[0])
    assert str(view.image.path) == _corrected_path(project)
    assert not view.canvas.paint_enabled
    assert not view.background_available()
    assert view.export_problems_now() == [CORRECTED_REFUSAL]


def test_a_raw_photo_session_opens_on_the_raw_photo(make_view, project, paths):
    # The session file as an earlier load in the corrected frame left it: the
    # corrected baseline as its path, the raw photo's size and raw marks.
    saved = _save_session(project, _corrected_path(project), RAW_SIZE, [RAW_POLYGON])
    marks_before = json.loads(saved.read_text(encoding="utf-8"))["polygons"]
    view = make_view()
    assert view.image is not None
    assert view.image.image_size == RAW_SIZE
    assert str(view.image.path) == str(paths["baseline"])
    assert view.reference().baseline_image == paths["baseline"]
    assert view.canvas.paint_enabled
    assert view.propose_button.isEnabled()
    assert [polygon.to_list() for polygon in view.session.polygons] == marks_before
    assert view.propose_status.text() == slv._RAW_SESSION_NOTE
    # The background photo works as on a raw project.
    assert view.background_available()
    assert view.set_background_photo(paths["before"])
    assert view.background_shown
    view.toggle_background()
    assert not view.background_shown


def test_saving_records_the_raw_photo_again_and_keeps_the_marks(make_view, project, paths):
    saved = _save_session(project, _corrected_path(project), RAW_SIZE, [RAW_POLYGON])
    marks_before = json.loads(saved.read_text(encoding="utf-8"))["polygons"]
    view = make_view()
    view.add_polygon(lp.Polygon(3, [(35.0, 40.0), (35.0, 70.0), (55.0, 70.0)]))
    view.save_session()
    after = json.loads(saved.read_text(encoding="utf-8"))
    assert after["image"]["path"] == str(paths["baseline"])
    assert after["image"]["size"] == list(RAW_SIZE)
    assert after["polygons"][: len(marks_before)] == marks_before
    assert len(after["polygons"]) == len(marks_before) + 1


def test_a_raw_photo_session_is_not_exported_into_the_corrected_project(
    make_view, project
):
    _save_session(project, _corrected_path(project), RAW_SIZE, [RAW_POLYGON])
    view = make_view()
    assert view.export_problems_now() == [slv._RAW_SESSION_EXPORT_NOTE]
    assert not view.export_button.isEnabled()
    view.save_and_export()
    assert view.export_status.text() == slv._RAW_SESSION_EXPORT_NOTE


def test_marks_on_the_corrected_grid_keep_the_corrected_view(make_view, project):
    size = (CORRECTED_SHAPE[1], CORRECTED_SHAPE[0])
    _save_session(project, _corrected_path(project), size, [RAW_POLYGON[:3]])
    view = make_view()
    assert view.image is not None and view.image.image_size == size
    assert not view.canvas.paint_enabled
    assert view.export_problems_now() == [CORRECTED_REFUSAL]


def test_a_session_without_marks_keeps_the_corrected_view(make_view, project):
    _save_session(project, _corrected_path(project), RAW_SIZE, [])
    view = make_view()
    assert view.image is not None
    assert view.image.image_size == (CORRECTED_SHAPE[1], CORRECTED_SHAPE[0])
    assert not view.canvas.paint_enabled


def test_a_wrapped_proposal_ready_still_runs_in_the_gui_thread(make_view, project, qtbot):
    # The coloured-map import replaces view._proposal_ready with a plain
    # function, and Qt calls a plain function in the emitting thread. The
    # proposal worker must therefore reach the view through a real slot.
    # (19 Oct: DarSIA closed after Propose; the facies table was refreshed
    # from the worker thread and Qt6Core raised an access violation.)
    session = lp.PaintingSession.default("ac52", _corrected_path(project), RAW_SIZE)
    session.polygons = [
        lp.Polygon(2, list(RAW_POLYGON)),
        lp.Polygon(3, [(35.0, 45.0), (35.0, 75.0), (55.0, 75.0), (55.0, 45.0)]),
    ]
    lp.save_painting_session(project, session, actor="test")
    view = make_view()
    assert view.propose_button.isEnabled()
    gui = threading.get_ident()
    seen = []
    original = view._proposal_ready

    def wrapped(proposal):
        seen.append(threading.get_ident())
        if seen[-1] == gui:  # Never touch widgets from another thread.
            original(proposal)

    view._proposal_ready = wrapped
    view.propose()
    # busy() follows the worker thread, which can stop before the GUI thread
    # has run the queued finished slot; wait for the slot itself first.
    qtbot.waitUntil(lambda: bool(seen), timeout=60000)
    qtbot.waitUntil(lambda: not view.busy(), timeout=60000)
    assert seen == [gui]
    assert view.proposal is not None


def test_a_raw_session_on_a_camera_raw_opens_on_u_from_the_linked_series(
    qtbot, monkeypatch, settings, tmp_path, paths, project
):
    # FF_AC's layout: the project is the geometry package, its manifest links
    # the series root, and that series profiles the run's camera RAW on grid U.
    # The marks were drawn on U, so the tab must find the profile through the
    # link and open the RAW on U instead of clearing the canvas.
    series = paths["baseline"].parent
    document = {
        "registration": {"common": {"profiles": {"early": {}},
                                    "sensor": {"camera_wh": list(RAW_SIZE)}}},
        "source": {"profiles": {"AC52": "early"}},
    }
    raw = json.dumps(document).encode("utf-8")
    (series / "native.json").write_bytes(raw)
    (series / "series.toml").write_text(
        f'[native]\nconfiguration = "native.json"\nsha256 = "{hashlib.sha256(raw).hexdigest()}"\n',
        encoding="utf-8",
    )
    project.manifest["tools"] = {"native_series": {"series_root": str(series)}}
    camera_raw = series / "DSC00441.ARW"
    camera_raw.write_bytes(b"synthetic camera RAW placeholder")
    rendered = (90, 180, 120)
    monkeypatch.setattr(
        raw_photograph, "load_camera_u_rgb8",
        lambda path, **kwargs: np.full((RAW_SIZE[1], RAW_SIZE[0], 3), rendered, dtype=np.uint8),
    )
    photograph_io._CAMERA_U_CACHE.clear()
    saved = _save_session(project, _corrected_path(project), RAW_SIZE, [RAW_POLYGON])
    marks_before = json.loads(saved.read_text(encoding="utf-8"))["polygons"]
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    reference = SeriesRunInputs(
        "ac52", paths["config"], baseline_image=camera_raw, width_m=0.9, height_m=0.49
    )
    view = LayerPainterView(project, lambda: reference)
    qtbot.addWidget(view)
    view.resize(1400, 900)
    view.show()
    qtbot.waitExposed(view)
    assert view.image is not None and view.image.image_size == RAW_SIZE
    assert view.image.camera_u is not None and str(view.image.path) == str(camera_raw)
    assert view.canvas.paint_enabled
    assert [polygon.to_list() for polygon in view.session.polygons] == marks_before
    assert view.propose_status.text() == slv._RAW_SESSION_NOTE
    assert view.background_available()
