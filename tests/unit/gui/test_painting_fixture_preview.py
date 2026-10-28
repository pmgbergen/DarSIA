"""Actual private painting view: opt-in underlay, guards and immutable inputs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import toml
from pathlib import Path

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_corrections import view  # noqa: F401


def _fixture(view, *, complete=False):
    view.rig_panel.frame = RigFrame(
        corners={"top_left": (0., 0.), "top_right": (0., 79.),
                 "bottom_right": (59., 79.), "bottom_left": (59., 0.)},
        width_m=.079, height_m=.059, run_id=view.session.run_id,
        image=view.session.image_path, image_size=(80, 60))
    view.session.snap_mm = 0.
    view.session.edge_fill_mm = 0.
    view.set_facies_role(view.session.layers[3].facies, "colorchecker")
    view.session.polygons = [
        lp.Polygon(2, [(0, 0), (0, 80 if complete else 29),
                       (30, 80 if complete else 29), (30, 0)]),
        lp.Polygon(3, [(30, 0 if complete else 51), (30, 80),
                       (60, 80), (60, 0 if complete else 51)]),
        lp.Polygon(4, [(20, 30), (20, 50), (40, 50), (40, 30)]),
        lp.Polygon(2, [(22, 32), (22, 36), (26, 36), (26, 32)]),
    ]
    if complete:
        view.session.layers[3] = replace(view.session.layers[3], always_on_top=True)
    view._sync_settings()
    view._refresh_status()
    return view.fixture_preview


def test_role_is_not_guessed_from_facies_name_and_preview_is_default_off(view):
    controller = view.fixture_preview
    assert not controller.show_underlay.isChecked()
    assert not controller.busy()
    assert "colorchecker" in lp.LAYER_ROLES
    assert view.session.facies_of(4).role == "excluded"
    before = deepcopy(view.session.to_dict())
    controller.refresh()
    assert not controller.show_underlay.isEnabled()
    assert not controller.item.isVisible()
    assert "explicitly" in controller.note.text()
    assert view.session.to_dict() == before


def test_moved_reference_photo_retains_its_frame_underlay(view, tmp_path, monkeypatch):
    controller = _fixture(view)
    old = Path(view.session.image_path)
    new = tmp_path / "moved" / old.name
    new.parent.mkdir()
    old.rename(new)
    registry = tmp_path / "private-registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases={str(old): str(new)}))), encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    view.session.image_path = str(new)
    view.image = replace(view.image, path=new)
    context, problem = controller._snapshot()
    assert context is not None and problem == ''
    view.rig_panel.frame.image = str(new.with_name('different.png'))
    assert 'another reference photograph' in controller._snapshot()[1]


def test_real_preview_counts_drawn_continued_and_preserves_saved_session(view, qtbot):
    controller = _fixture(view)
    view.save_session()
    session_file = lp.session_path(view.project, view.session.run_id)
    before_file = session_file.read_bytes()
    before = deepcopy(view.session.to_dict())
    assert controller.show_underlay.isEnabled()
    controller.show_underlay.setChecked(True)
    qtbot.waitUntil(lambda: not controller.busy() and controller.result is not None,
                    timeout=10000)
    assert not controller.error
    assert controller.result["counts"]["drawn"] > 0
    assert controller.result["counts"]["continued"] > 0
    assert controller.item.isVisible()
    assert "(solid)" in controller.note.text() and "(hatched)" in controller.note.text()
    assert view.session.to_dict() == before
    assert session_file.read_bytes() == before_file
    controller.show_underlay.setChecked(False)
    assert not controller.item.isVisible()
    assert view.session.to_dict() == before


def test_other_photo_and_native_grid_mismatch_hide_existing_preview(view, qtbot):
    controller = _fixture(view)
    controller.show_underlay.setChecked(True)
    qtbot.waitUntil(lambda: not controller.busy() and controller.result is not None,
                    timeout=10000)
    original = view.session.image_size
    view.session.image_size = (79, 60)
    controller.refresh()
    assert not controller.item.isVisible()
    assert "grid" in controller.note.text()
    view.session.image_size = original
    view._other_photo = {"path": "another-private-photo"}
    controller.refresh()
    assert not controller.item.isVisible() and not controller.show_underlay.isEnabled()
    assert "reference photograph" in controller.note.text()
    view._other_photo = None
    controller.show_underlay.setChecked(False)


def test_stale_worker_never_publishes_after_drawing_changes_or_preview_closes(view, qtbot, monkeypatch):
    import threading
    from PySide6.QtCore import QTimer
    from darsia.fluidflower import painting_fixture_preview as module
    controller = _fixture(view)
    entered, release = threading.Event(), threading.Event()
    original = module.preview_underlay
    def blocked(*args):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("The private test did not release its finite worker")
        return original(*args)
    monkeypatch.setattr(module, "preview_underlay", blocked)
    ticks = []
    timer = QTimer(view)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    try:
        controller.show_underlay.setChecked(True)
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: len(ticks) >= 3)
        assert controller.busy() and not controller.can_close()
        view.session.polygons.append(lp.Polygon(3, [(28, 32), (28, 35), (30, 35), (30, 32)]))
        view.changed.emit()
        assert not controller.item.isVisible()
        controller.show_underlay.setChecked(False)
    finally:
        timer.stop()
        release.set()
        controller.wait_for_workers()
    assert not controller.busy() and controller.can_close()
    assert controller.result is None and controller._loaded_key is None
    assert not controller.item.isVisible()


def test_adoption_worker_replays_real_saved_fixture_version_before_sink(view, qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import map_adoption, map_placement, raw_shape_basis_reader
    from darsia.fluidflower import series_layer_view as module
    controller = _fixture(view, complete=True)
    # Use the original immutable manual saver/reader, without performing adoption.
    from darsia.fluidflower import layer_painting_manual as manual
    frame = view.current_frame()
    saved = manual.manual_map(view.session, (80, 60), frame=frame)
    record = manual.save_manual_version(view.project, view.session, saved, name="Private map")
    approval = manual.approve_manual_version(record)
    # Keep this as real fixture replay before its already mocked sink, with
    # no claim that the synthetic index has scientific RAW-basis authority.
    seen, preflights = {}, []
    index = tmp_path / "legacy-index.json"
    index.write_text(json.dumps(dict(format=map_placement.GEOMETRY_FORMAT,
        runs={view.session.run_id: {}})), encoding="utf-8")
    index_identity = dict(path=str(index.resolve()),
        sha256=hashlib.sha256(index.read_bytes()).hexdigest(), size_bytes=index.stat().st_size)
    def preflight(path, runs):
        preflights.append((Path(path), list(runs)))
    def capture(destination, layers, **kwargs):
        assert preflights == [(index, [view.session.run_id, view.session.run_id])]
        seen.update(layers=layers, **kwargs)
        return {"private": True, "geometry_index": index_identity}
    monkeypatch.setattr(raw_shape_basis_reader, "require_basis_index", preflight)
    monkeypatch.setattr(map_adoption, "prepare_adoption", capture)
    worker = module._AdoptionWorker(record, frame, dict(destination=tmp_path / "unused",
        geometry_index=index, runs=[view.session.run_id], approval=approval))
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures
    assert seen["roles"][4] == "colorchecker"
    assert seen["fixture_frame"].to_dict() == frame.to_dict()
    assert seen["fixture_underlay"].provenance["version"]["sha256"] == record["sha256"]
    assert seen["fixture_underlay"].analysis_layers[23, 34] == 4
    assert seen["fixture_underlay"].drawn_layers[23, 34] == 2
    assert not (tmp_path / "unused").exists()
    assert not controller.busy()
