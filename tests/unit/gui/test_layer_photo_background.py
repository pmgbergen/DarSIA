"""Layer painting tab: a large photograph is decoded off the GUI thread and kept.

The reference photograph, a background photo (remembered or chosen) and
another photograph of a run are decoded in the background; the page says so
meanwhile and shows them when they are ready. Synthetic 80 x 60 photographs and
a project in a temporary folder; no real series, rig model or NAS input is used.
"""

import threading
from types import SimpleNamespace as NS

import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel, QMenu, QWidget

from darsia.fluidflower import series_layer_view as view_module
from darsia.fluidflower.series_geometry import SeriesRunInputs

View = view_module.LayerPainterView
REFERENCE_RGB = (30, 40, 50)
BACKGROUND_RGB = (200, 160, 40)


def _host(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.reloads, host.cleared = [], []
    host.reload = lambda: host.reloads.append(True)
    host.canvas = NS(clear_photo=lambda: host.cleared.append(True))
    host.propose_status = QLabel()
    for name in ("_decoded_photo", "_keep_decoded", "_start_photo_decode", "_photo_decoded"):
        setattr(host, name, getattr(View, name).__get__(host))
    return host


def _photo(tmp_path, name, size):
    path = tmp_path / name
    path.write_bytes(b"x" * size)
    return path


def test_a_large_photo_is_decoded_in_the_background_and_kept(qtbot, monkeypatch, tmp_path):
    host, calls = _host(qtbot), []

    def decode(path, camera_u):
        calls.append(threading.current_thread() is threading.main_thread())
        return "image", "detail"

    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 16)
    monkeypatch.setattr(view_module, "_decode_reference_photo", decode)
    photo = _photo(tmp_path, "reference.arw", 64)
    assert host._decoded_photo(photo, None) is None  # The page says it is loading.
    qtbot.waitUntil(lambda: bool(host.reloads), timeout=10000)
    assert host._decoded_photo(photo, None) == ("image", "detail")  # Kept: no second decode.
    assert calls == [False]


def test_a_small_photo_decodes_at_once(qtbot, monkeypatch, tmp_path):
    host = _host(qtbot)
    monkeypatch.setattr(view_module, "_decode_reference_photo", lambda path, camera_u: ("image", None))
    assert host._decoded_photo(_photo(tmp_path, "fixture.png", 4), None) == ("image", None)
    assert not host.reloads


def test_a_photo_that_cannot_be_decoded_says_so(qtbot, monkeypatch, tmp_path):
    host = _host(qtbot)

    def decode(path, camera_u):
        raise ValueError("not a camera RAW")

    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 16)
    monkeypatch.setattr(view_module, "_decode_reference_photo", decode)
    assert host._decoded_photo(_photo(tmp_path, "broken.arw", 64), None) is None
    qtbot.waitUntil(lambda: bool(host.cleared), timeout=10000)
    assert host.propose_status.text() == "Could not read the baseline photograph: ValueError: not a camera RAW"
    assert not host.reloads


def test_a_photo_no_longer_wanted_is_kept_but_not_shown(qtbot, monkeypatch, tmp_path):
    host, release, calls = _host(qtbot), threading.Event(), []

    def decode(path, camera_u):
        calls.append(path.name)
        if path.name == "first.arw":
            assert release.wait(10)
        return "image of " + path.name, None

    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 16)
    monkeypatch.setattr(view_module, "_decode_reference_photo", decode)
    first = _photo(tmp_path, "first.arw", 64)
    assert host._decoded_photo(first, None) is None
    # Another run with a small photograph is opened meanwhile.
    assert host._decoded_photo(_photo(tmp_path, "other.png", 4), None) == ("image of other.png", None)
    release.set()
    qtbot.waitUntil(lambda: not host._photo_decoder.busy, timeout=10000)
    qtbot.wait(50)
    assert not host.reloads
    assert host._decoded_photo(first, None) == ("image of first.arw", None)
    assert calls == ["first.arw", "other.png"]


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Keep the remembered choices in a private INI file, not the user's registry."""

    path = str(tmp_path / "painting.ini")
    monkeypatch.setattr(view_module, "QSettings", lambda *args: QSettings(path, QSettings.Format.IniFormat))


@pytest.fixture
def paths(tmp_path):
    folder = tmp_path / "series"
    folder.mkdir()
    baseline = folder / "raw.png"
    Image.new("RGB", (80, 60), REFERENCE_RGB).save(baseline)
    before = folder / "before_indicator.png"
    Image.new("RGB", (80, 60), BACKGROUND_RGB).save(before)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    return {"baseline": baseline, "before": before, "config": config}


def _view(qtbot, monkeypatch, tmp_path, paths):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    reference = SeriesRunInputs("ac52", paths["config"], baseline_image=paths["baseline"], width_m=0.9, height_m=0.49)
    widget = View(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    return widget


def _held(monkeypatch, name):
    """Hold the module's decode function ``name`` until the returned event is set."""

    release, original = threading.Event(), getattr(view_module, name)

    def held(*args):
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(view_module, name, held)
    return release


def test_the_page_opens_while_the_reference_photo_decodes(qtbot, monkeypatch, tmp_path, settings, paths):
    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 1)
    release = _held(monkeypatch, "_decode_reference_photo")
    view = _view(qtbot, monkeypatch, tmp_path, paths)
    assert view.image is None
    assert view.propose_status.text() == "Loading the reference photograph…"
    release.set()
    qtbot.waitUntil(lambda: view.image is not None, timeout=10000)
    assert view.image.image_size == (80, 60)
    assert view.canvas.preview["path"] == str(paths["baseline"])
    assert view.session is not None


def test_the_remembered_background_photo_is_restored_when_decoded(qtbot, monkeypatch, tmp_path, settings, paths):
    view = _view(qtbot, monkeypatch, tmp_path, paths)
    assert view.set_background_photo(paths["before"])
    view.show_background(False)
    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 1)
    view._decoded_backgrounds.clear()
    release = _held(monkeypatch, "_decode_background_photo")
    view.reload()
    # The kept reference photo is shown at once; the background follows.
    assert view.image is not None and view.background_path is None
    release.set()
    qtbot.waitUntil(lambda: view.background_path == paths["before"], timeout=10000)
    assert not view.background_shown
    assert view.canvas.preview["path"] == str(paths["baseline"])


def test_a_chosen_large_background_photo_is_shown_when_decoded(qtbot, monkeypatch, tmp_path, settings, paths):
    view = _view(qtbot, monkeypatch, tmp_path, paths)
    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 1)
    release = _held(monkeypatch, "_decode_background_photo")
    assert not view.set_background_photo(paths["before"])
    assert view.photo_note.text() == "Loading the background photo before_indicator.png…"
    assert not view.photo_note_row.isHidden()
    assert view.canvas.preview["path"] == str(paths["baseline"])  # The reference stays meanwhile.
    release.set()
    qtbot.waitUntil(lambda: view.background_shown, timeout=10000)
    assert view.background_path == paths["before"]
    assert view.canvas.preview["path"] == str(paths["before"])
    assert view.photo_note_row.isHidden()


def test_a_background_choice_made_while_one_decodes_wins(qtbot, monkeypatch, tmp_path, settings, paths):
    view = _view(qtbot, monkeypatch, tmp_path, paths)
    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 1)
    release = _held(monkeypatch, "_decode_background_photo")
    assert not view.set_background_photo(paths["before"])
    assert view.set_background_photo(None)  # Back to the reference photo meanwhile.
    release.set()
    qtbot.waitUntil(lambda: not view._background_decoder.busy, timeout=10000)
    qtbot.wait(50)
    assert view.background_path is None and not view.background_shown
    assert len(view._decoded_backgrounds) == 1  # Kept for a later choice.


def _other_photo(tmp_path):
    path = tmp_path / "series" / "other_frame.png"
    Image.new("RGB", (80, 60), (90, 200, 90)).save(path)
    return path


def test_a_large_other_photo_is_shown_when_decoded_and_kept(qtbot, monkeypatch, tmp_path, settings, paths):
    view = _view(qtbot, monkeypatch, tmp_path, paths)
    other, calls, release = _other_photo(tmp_path), [], threading.Event()
    original = view_module._decode_frame_photo

    def decode(path, camera_u):
        calls.append(threading.current_thread() is threading.main_thread())
        assert release.wait(10)
        return original(path, camera_u)

    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 1)
    monkeypatch.setattr(view_module, "_decode_frame_photo", decode)
    assert view.show_frame_photo(other)
    assert view.photo_note.text() == "Loading other_frame.png…"
    assert view._other_photo == other and not view.canvas.paint_enabled
    release.set()
    qtbot.waitUntil(lambda: (view.canvas.preview or {}).get("path") == str(other), timeout=10000)
    assert view.photo_note.text() == "Showing other_frame.png. Painting works on the reference photo."
    # The saturation and contrast controls show it again without decoding it.
    view.saturation.setValue(30)
    view.apply_saturation()
    view.enhance.toggle()
    assert calls == [False]
    assert view.canvas.preview["path"] == str(other)


def test_going_back_to_the_reference_while_another_photo_decodes(qtbot, monkeypatch, tmp_path, settings, paths):
    view = _view(qtbot, monkeypatch, tmp_path, paths)
    other = _other_photo(tmp_path)
    monkeypatch.setattr(view_module, "_BACKGROUND_DECODE_BYTES", 1)
    release = _held(monkeypatch, "_decode_frame_photo")
    assert view.show_frame_photo(other)
    view.show_reference_photo()
    release.set()
    qtbot.waitUntil(lambda: not view._frame_decoder.busy, timeout=10000)
    qtbot.wait(50)
    assert view._other_photo is None
    assert view.canvas.preview["path"] == str(paths["baseline"])
    assert len(view._decoded_frames) == 1
