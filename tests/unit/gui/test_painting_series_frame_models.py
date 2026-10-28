"""Series display models are automatic, read only, and below USER overrides."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import threading

from PIL import Image
import pytest
from PySide6.QtCore import QSettings

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import painting_frame_models as frame_models
from darsia.fluidflower import painting_run_preview_ui as preview_ui
from darsia.fluidflower import series_layer_view as layer_ui
from darsia.fluidflower.map_placement import GEOMETRY_FORMAT
from darsia.fluidflower.series_geometry import SeriesRunInputs


def _descriptor(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _files(*roots):
    return {str(p): p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}


class _Project:
    def __init__(self, root, series):
        self.root, self.project_id = root, "private-series-project"
        self.manifest = {"rig": {"type": "medium"}, "tools": {"native_series": {
            "series_root": str(series), "series_id": "private_series"}}}

    def update_metadata(self, *_args, **_kwargs):
        pass

    def _audit(self, *_args, **_kwargs):
        pass


@pytest.fixture
def context(qtbot, tmp_path, monkeypatch, request):
    mode = getattr(request, "param", "series")
    gui_thread = threading.get_ident()
    settings_path = tmp_path / "private-user.ini"
    calls = []

    def private_settings(*args):
        assert tuple(args) == preview_ui._SETTINGS == layer_ui._BACKGROUND_SETTINGS
        assert threading.get_ident() == gui_thread, "Metadata worker accessed QSettings"
        value = QSettings(str(settings_path), QSettings.Format.IniFormat)
        value.setFallbacksEnabled(False)
        assert value.format() == QSettings.Format.IniFormat
        assert value.scope() == QSettings.Scope.UserScope
        assert Path(value.fileName()) == settings_path and not value.fallbacksEnabled()
        calls.append(tuple(args))
        return value

    monkeypatch.setattr(layer_ui, "QSettings", private_settings)
    monkeypatch.setattr(preview_ui, "QSettings", private_settings)
    def forbidden_picker(*_args, **_kwargs):
        pytest.fail("Automatic display models must not open a file picker")
    monkeypatch.setattr(preview_ui.QFileDialog, "getOpenFileName", forbidden_picker)
    registry = tmp_path / "private-registry.toml"
    registry.write_text("schema_version = 1\n", encoding="utf8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    series = tmp_path / "series"
    (series / "config").mkdir(parents=True)
    (series / "series.toml").write_text('schema_version = 1\nseries_id = "private_series"\n'
        '[paths]\ncommon_config = "config/common.toml"\n[runs]\nids = ["run_a", "run_b"]\n'
        '[runs.config_files]\nrun_a = "config/first.toml"\nrun_b = "config/second.toml"\n', encoding="utf8")
    (series / "config/common.toml").write_text(
        '[rig]\nwidth = 0.159\nheight = 0.119\n'
        '[corrections.curvature.crop]\n'
        'top_left = [0.0, 0.0]\nbottom_left = [119.0, 0.0]\n'
        'bottom_right = [119.0, 159.0]\ntop_right = [0.0, 159.0]\n', encoding="utf8")
    project = _Project(tmp_path / "project", series)
    folder = project.root / "display-models"
    folder.mkdir(parents=True)
    entries, references, photos = {}, {}, {}
    corners = dict(top_left=[0., 0.], bottom_left=[119., 0.],
                   bottom_right=[119., 159.], top_right=[0., 159.])
    for number, run in enumerate(("run_a", "run_b")):
        photo = series / (run + ".png")
        Image.new("RGB", (160, 120), (40 + 70*number, 80, 120)).save(photo)
        config = series / "config" / ("first.toml" if number == 0 else "second.toml")
        config.write_text(f'[data]\nfolder = "{series.as_posix()}"\nbaseline = "{run}.png"\n', encoding="utf8")
        references[run] = SeriesRunInputs(run, config, baseline_image=photo,
            width_m=.159, height_m=.119, crop=deepcopy(corners))
        photos[run] = photo
        model = folder / (run + ".json")
        model.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, run_id=run,
            reference_image=_descriptor(photo), shape_hw=[120, 160],
            H_metric_mm_to_camera_u=[[1., 0., 5.*number], [0., -1., 119.], [0., 0., 1.]],
            metric_frame=dict(unit="mm", x_axis="right", y_axis="up", measured_width_mm=159.))), encoding="utf8")
        entries[run] = dict(reference_model=_descriptor(model))
    index = folder / "index.json"
    index.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, runs=entries)), encoding="utf8")
    binding = dict(format=frame_models.FORMAT, scope=frame_models.SCOPE, adopted=False,
        geometry_index=dict(_descriptor(index), path="display-models/index.json"))
    if mode == "malformed":
        binding["adopted"] = True
    if mode == "wrong_sha":
        binding["geometry_index"]["sha256"] = "0" * 64
    if mode != "missing":
        project.manifest["tools"]["native_series"]["display_frame_models"] = binding
    settings = private_settings(*preview_ui._SETTINGS)
    settings.setValue("fixture/preserved", "unchanged")
    override = tmp_path / "user-override-index.json"
    override.write_bytes(index.read_bytes())
    if mode == "override":
        settings.setValue(preview_ui._INDEX, str(override))
    settings.sync()
    verification_calls = []
    original_verify = frame_models.verify_frame_model_selection
    def observed_verify(selection):
        verification_calls.append((threading.get_ident(), selection))
        assert threading.get_ident() != gui_thread, "Series SHA must be checked on a metadata worker"
        return original_verify(selection)
    monkeypatch.setattr(frame_models, "verify_frame_model_selection", observed_verify)
    if getattr(preview_ui, "verify_frame_model_selection", None) is original_verify:
        monkeypatch.setattr(preview_ui, "verify_frame_model_selection", observed_verify)
    # New placement policy freezes the real private project/config authority;
    # no in-memory-only frame or real USER/project file is substituted.
    (project.root / "series.json").write_text(json.dumps(project.manifest), encoding="utf8")
    view = layer_ui.LayerPainterView(project, lambda: references["run_a"],
                                    run_inputs=lambda: list(references.values()))
    qtbot.addWidget(view)
    view.resize(1500, 1000)
    view.show()
    qtbot.waitExposed(view)
    qtbot.waitUntil(lambda: not view.busy() and not view.run_preview.busy(), timeout=15000)
    # Save only this private drawing before capturing preservation evidence.
    view.add_polygon(lp.Polygon(2, [(10., 10.), (10., 70.), (70., 70.), (70., 10.)]))
    view.save_session()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False
    assert calls
    yield SimpleNamespace(view=view, project=project, series=series, index=index,
        override=override, settings=settings, settings_path=settings_path,
        references=references, photos=photos, verify_calls=verification_calls, gui_thread=gui_thread)
    view.wait_for_workers()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False


def _preview_other(context, qtbot):
    preview = context.view.run_preview
    preview.runs.setCurrentIndex(preview.runs.findData("run_b"))
    qtbot.waitUntil(lambda: not context.view.busy() and not preview.busy(), timeout=15000)
    return preview


def _before(context):
    context.settings.sync()
    return (_files(context.series, context.project.root), deepcopy(context.view.session.to_dict()),
            deepcopy(context.project.manifest), context.settings_path.read_bytes())


def _unchanged(context, before):
    assert _files(context.series, context.project.root) == before[0]
    assert context.view.session.to_dict() == before[1]
    assert context.project.manifest == before[2]
    context.settings.sync()
    assert context.settings_path.read_bytes() == before[3]


def test_series_models_are_automatic_without_a_user_selection_or_picker(context, qtbot):
    preview = context.view.run_preview
    assert not context.settings.contains(preview_ui._INDEX)
    selected = preview.geometry_selection()
    assert selected.source == "series display binding"
    assert Path(selected.index_path) == context.index and selected.sha256 == _descriptor(context.index)["sha256"]
    before = _before(context)
    preview = _preview_other(context, qtbot)
    assert preview.last_result is not None, preview.note.text()
    assert preview.last_result.run_id == "run_b"
    assert Path(preview.last_result.photo_path) == context.photos["run_b"]
    assert context.verify_calls and all(t != context.gui_thread for t, _s in context.verify_calls)
    assert not context.settings.contains(preview_ui._INDEX), "A series choice became a USER override"
    _unchanged(context, before)


@pytest.mark.parametrize("context", ["override"], indirect=True)
def test_explicit_private_user_override_wins_and_is_preserved(context, qtbot):
    preview = context.view.run_preview
    selected = preview.geometry_selection()
    assert selected.source == "user override" and not selected.sha256
    assert Path(selected.index_path) == context.override
    before = _before(context)
    preview.ensure_saved_geometry_index()
    assert Path(preview.geometry_selection().index_path) == context.override
    preview = _preview_other(context, qtbot)
    assert preview.last_result is not None, preview.note.text()
    assert context.settings.value(preview_ui._INDEX) == str(context.override)
    _unchanged(context, before)


@pytest.mark.parametrize("context", ["missing", "malformed"], indirect=True)
def test_absent_or_malformed_series_binding_has_a_clear_reason_without_picker(context, qtbot):
    before = _before(context)
    preview = _preview_other(context, qtbot)
    assert preview.last_result is None
    reason = preview.note.text().casefold()
    assert "frame" in reason and any(word in reason for word in ("invalid", "no measured", "missing", "override"))
    assert not context.settings.contains(preview_ui._INDEX)
    _unchanged(context, before)


@pytest.mark.parametrize("context", ["wrong_sha"], indirect=True)
def test_changed_series_index_sha_is_refused_by_metadata_worker(context, qtbot):
    before = _before(context)
    preview = _preview_other(context, qtbot)
    assert preview.last_result is None
    assert "frame-model index changed" in preview.note.text().casefold()
    assert context.verify_calls and all(t != context.gui_thread for t, _s in context.verify_calls)
    assert any(selection.sha256 == "0" * 64 for _t, selection in context.verify_calls)
    _unchanged(context, before)


def test_another_private_user_also_gets_the_series_default(context, qtbot, monkeypatch):
    # A second actual host uses the same project and saved drawing, but a
    # different empty USER INI. The series choice must survive that boundary.
    other_path = context.settings_path.with_name("another-private-user.ini")
    def other_settings(*args):
        assert tuple(args) == preview_ui._SETTINGS == layer_ui._BACKGROUND_SETTINGS
        assert threading.get_ident() == context.gui_thread
        value = QSettings(str(other_path), QSettings.Format.IniFormat)
        value.setFallbacksEnabled(False)
        assert value.format() == QSettings.Format.IniFormat
        assert value.scope() == QSettings.Scope.UserScope
        assert Path(value.fileName()) == other_path and not value.fallbacksEnabled()
        return value
    monkeypatch.setattr(layer_ui, "QSettings", other_settings)
    monkeypatch.setattr(preview_ui, "QSettings", other_settings)
    other = other_settings(*preview_ui._SETTINGS)
    other.setValue("fixture/another-user", "preserved")
    other.sync()
    assert not other.contains(preview_ui._INDEX)
    before = _before(context)
    second = layer_ui.LayerPainterView(context.project, lambda: context.references["run_a"],
        run_inputs=lambda: list(context.references.values()))
    qtbot.addWidget(second)
    second.resize(1500, 1000)
    second.show()
    try:
        qtbot.waitExposed(second)
        qtbot.waitUntil(lambda: not second.busy() and not second.run_preview.busy(), timeout=15000)
        selection = second.run_preview.geometry_selection()
        assert selection.source == "series display binding" and Path(selection.index_path) == context.index
        second_user_bytes = other_path.read_bytes()
        second.run_preview.runs.setCurrentIndex(second.run_preview.runs.findData("run_b"))
        qtbot.waitUntil(lambda: not second.busy() and not second.run_preview.busy(), timeout=15000)
        assert second.run_preview.last_result is not None, second.run_preview.note.text()
        assert not other.contains(preview_ui._INDEX)
        other.sync()
        assert other_path.read_bytes() == second_user_bytes
        _unchanged(context, before)
    finally:
        second.wait_for_workers()
        for timer in (second._save_timer, second._manual_timer, second._saturation_timer):
            timer.stop()
        second._dirty = False


def test_place_saved_version_uses_the_same_series_models_without_picker(context, monkeypatch):
    # Save and approve only a complete private fixture map, then select it in
    # the actual table. Placement itself is captured, so no placement worker runs.
    view = context.view
    # This fixture isolates frame-model selection, not per-run point approval.
    monkeypatch.setattr(view, '_reference_point_review_blocker', lambda: None)
    saved_session = deepcopy(view.session)
    saved_session.polygons = [lp.Polygon(2, [(0., 0.), (0., 159.), (119., 159.), (119., 0.)])]
    mapped = manual.manual_map(saved_session, (160, 120), frame=view.current_frame())
    record = manual.save_manual_version(context.project, saved_session, mapped,
                                       name="Private complete map")
    manual.approve_manual_version(record, reviewer="Private fixture reviewer")
    view.versions = view._read_versions(view.session.run_id)
    view._read_approvals(view.session.run_id)
    view._refresh_versions()
    view.versions_table.selectRow(0)
    assert view._selected_version()["sha256"] == record["sha256"]
    assert view._placeable(view._selected_version())
    # Existing accepted blank review order is not a new default/override.
    context.settings.setValue(layer_ui._FIRST_RUNS_KEY, "")
    context.settings.sync()
    monkeypatch.setattr(layer_ui.QInputDialog, "getText", lambda *_a, **_k: ("", True))
    captured = []
    def capture(record, geometry_index, **options):
        captured.append((record, geometry_index, options))
        return True
    monkeypatch.setattr(view, "place_version", capture)
    selection = view.run_preview.geometry_selection()
    before = _before(context)
    view._place_version_clicked()
    assert len(captured) == 1
    selected_record, index_path, options = captured[0]
    assert selected_record["sha256"] == record["sha256"]
    assert Path(index_path) == context.index == Path(selection.index_path)
    assert options["runs"] == ["run_a", "run_b"] and options["first_runs"] == []
    assert view.run_preview.geometry_selection() == selection
    assert selection.source == "series display binding"
    assert not context.settings.contains(preview_ui._INDEX)
    _unchanged(context, before)


def test_open_series_default_view_observes_a_later_private_user_override(context):
    preview = context.view.run_preview
    assert preview.geometry_selection().source == "series display binding"
    assert not context.settings.contains(preview_ui._INDEX)
    # An explicit choice saved by another GUI consumer takes priority over
    # this view's already-cached automatic series choice on its next refresh.
    context.settings.setValue(preview_ui._INDEX, str(context.override))
    context.settings.sync()
    before = _before(context)
    preview.ensure_saved_geometry_index()
    preview.refresh()
    selected = preview.geometry_selection()
    assert selected.source == "user override" and Path(selected.index_path) == context.override
    assert not selected.sha256
    _unchanged(context, before)
