"""Synthetic routing and saved-basis contracts; no real rig build or RAW decode."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.fluidflower import rig_raw_input as raw
from darsia.fluidflower.native import raw_positioning
from darsia.presets.workflows.rig import Rig


@pytest.fixture
def bound_input(tmp_path, monkeypatch):
    source = tmp_path / "baseline.ARW"
    source.write_bytes(b"synthetic acquisition")
    profile = tmp_path / "camera.json"
    parameters = dict(common=dict(profiles={"profile": {}},
                                  sensor=dict(camera_wh=[3, 2])))
    profile.write_text(json.dumps(dict(registration=parameters,
        source=dict(profiles=dict(sample="profile")))), encoding="utf-8")
    binding = dict(configuration=str(profile),
                   sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                   profile_id="profile", run_id="sample")
    sensor = SimpleNamespace(metadata=dict(white_level=1100, black_level=100))
    linear = np.arange(18, dtype=np.float32).reshape(3, 2, 3) * 10
    calls = []

    def read(path, **kwargs):
        calls.append((path, kwargs))
        return sensor

    monkeypatch.setattr(raw_positioning, "read_sensor", read)
    monkeypatch.setattr(raw_positioning, "camera_u", lambda *a, **k: linear.copy())
    from darsia.fluidflower import photograph_io
    monkeypatch.setattr(photograph_io, "raw_capture_time", lambda path: None)
    return SimpleNamespace(source=source, profile=profile, binding=binding,
                           parameters=parameters, linear=linear, calls=calls)


def test_raw_setup_uses_bound_camera_u_linear_values(bound_input):
    f = bound_input
    before = f.source.read_bytes(), f.profile.read_bytes()
    image = raw.read_raw_image(f.source, f.binding, dtype=np.float32)
    np.testing.assert_array_equal(image.img, np.moveaxis(f.linear, 0, -1) / np.float32(1000))
    assert image.img.dtype == np.float32 and image.series is False
    assert f.calls == [(f.source, dict(
        expected_sha256=hashlib.sha256(before[0]).hexdigest(),
        parameters=f.parameters, profile_id="profile"))]
    assert (f.source.read_bytes(), f.profile.read_bytes()) == before


def test_changed_camera_profile_refuses_before_sensor_decode(bound_input):
    f = bound_input
    f.profile.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Source artifact changed"):
        raw.read_raw_image(f.source, f.binding)
    assert not f.calls


def test_raw_reader_uses_declared_moved_source_for_hash_sensor_and_time(bound_input, tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths, photograph_io
    f = bound_input
    moved = tmp_path / "moved" / f.source.name
    moved.parent.mkdir()
    f.source.rename(moved)
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(f.source).replace("\\", "/").casefold(), str(moved).replace("\\", "/"))])
    dates = []
    monkeypatch.setattr(photograph_io, "raw_capture_time", lambda path: dates.append(path))
    image = raw.read_raw_image(f.source, f.binding, dtype=np.float32)
    np.testing.assert_array_equal(image.img, np.moveaxis(f.linear, 0, -1) / np.float32(1000))
    assert f.calls[0][0] == moved and dates == [moved] and not f.source.exists()
    assert f.calls[0][1]["expected_sha256"] == hashlib.sha256(moved.read_bytes()).hexdigest()


def test_raw_rig_rejects_a_companion_or_cached_optical_input(bound_input):
    f = bound_input
    rig = Rig()
    rig.raw_camera = f.binding
    for suffix in (".JPG", ".npz"):
        with pytest.raises(ValueError, match="requires RAW optical inputs"):
            rig._read_optical_image(f.source.with_suffix(suffix))
    assert not f.calls


@pytest.mark.parametrize("update", [dict(sha256="bad"), dict(profile_id=""),
                                    dict(extra=True), dict(configuration="relative.json")])
def test_raw_profile_binding_is_explicit(bound_input, update):
    with pytest.raises(ValueError, match="raw_camera"):
        raw.camera_binding(dict(bound_input.binding, **update))


def test_raw_rebuild_refuses_a_previous_relative_color_fit():
    config = SimpleNamespace(relative_color=SimpleNamespace(path=Path("previous.npz")))
    with pytest.raises(ValueError, match="must rebuild relative color"):
        raw.require_raw_rebuild(config)


@pytest.mark.parametrize("linear", [False, True])
def test_saved_drift_basis_controls_the_estimation_view(tmp_path, linear):
    values = np.linspace(0.01, 0.8, 18, dtype=np.float32).reshape(2, 3, 3)
    correction = darsia.DriftCorrection(values, config=dict(linear_input=linear))
    archive = tmp_path / "drift.npz"
    correction.save(archive)
    restored = darsia.DriftCorrection()
    restored.load(archive)
    assert restored.linear_input is linear
    assert ("linear_input" in restored.return_config()) is linear
    observed = []

    def estimate(source, target, *args):
        observed.append(source.copy())
        return np.array([[1., 0., 0.], [0., 1., 0.]]), True

    restored.translation_estimator = SimpleNamespace(find_effective_translation=estimate)
    image = darsia.OpticalImage(values.copy(), color_space="RGB")
    raw_positioning._apply_drift(image, restored)
    np.testing.assert_array_equal(observed[0], values if linear else raw_positioning.drift_view(values))
    np.testing.assert_array_equal(image.img, values)


def test_raw_setup_passes_its_owned_context_and_keeps_an_existing_npz_cache(bound_input, tmp_path, monkeypatch):
    f = bound_input
    setup = importlib.import_module("darsia.presets.workflows.setup.setup_rig")
    cache = tmp_path / "raw_cache"
    cache.mkdir()
    old = cache / "baseline.npz"
    old.write_bytes(b"previous input basis")
    config = SimpleNamespace(
        data=SimpleNamespace(data=[], baseline=f.source, use_cache=False,
                             raw_cache=cache, folders=[tmp_path], raw_camera=f.binding),
        depth=SimpleNamespace(depth_map="depth"),
        labeling=SimpleNamespace(labels="labels"),
        facies=SimpleNamespace(path="facies", props="props"),
        protocols=SimpleNamespace(imaging="imaging", injection="injection",
                                  pressure_temperature="pt", blacklist=None),
        rig=SimpleNamespace(path=tmp_path / "rebuilt"), corrections=None,
        image_porosity=None, check=lambda *args: None,
    )
    monkeypatch.setattr(setup, "FluidFlowerConfig", lambda *a, **k: config)
    monkeypatch.setattr(setup.darsia, "ProtocolledExperiment", lambda **k: "experiment")
    from darsia.fluidflower import rig_raw_build
    from darsia.fluidflower import rig_raw_basis
    from darsia.presets.workflows.config import utils
    build = SimpleNamespace(raw_source=dict(sha256="a" * 64), operation_id="fresh", publish=lambda rig: None)
    monkeypatch.setattr(rig_raw_build, "begin_raw_rig_build", lambda *a, **k: build)
    monkeypatch.setattr(rig_raw_build, "read_build_request", lambda *a, **k: ({}, ()))
    monkeypatch.setattr(rig_raw_basis, "require_protocol_sources", lambda *a: None)
    monkeypatch.setattr(utils, "_get_section_from_toml", lambda *a: dict(raw_camera=f.binding, use_cache=False))
    calls = []

    class RecordingRig:
        def setup(self, **kwargs):
            calls.append(kwargs)

        def save(self, folder):
            calls.append(folder)

    config_path = tmp_path / "run.toml"
    config_path.write_text("# explicit operation input\n", encoding="utf-8")
    setup.setup_rig(RecordingRig, config_path, raw_build_request=dict(synthetic=True))
    assert calls[0]["baseline_path"] == f.source
    assert calls[0]["raw_camera"] == f.binding
    assert calls[0]["raw_build"] is build
    assert calls[1] == config.rig.path
    assert old.read_bytes() == b"previous input basis"


def test_reloaded_rig_keeps_its_raw_optical_basis(bound_input, tmp_path, monkeypatch):
    f = bound_input
    rig = Rig()
    rig.baseline_path = f.source
    rig.raw_camera = f.binding
    rig.baseline = darsia.OpticalImage(np.zeros((2, 3, 3)), color_space="RGB")
    rig.shape_corrections = rig.color_corrections = []
    for name in ("depth", "labels", "facies", "porosity", "permeability", "image_porosity"):
        setattr(rig, name, darsia.ScalarImage(np.ones((2, 3))))
    destination = tmp_path / "rig"
    # A preserved synthetic saved fixture exercises reading only. New RAW saves
    # below require a reserved context; this is not a certificate retrofit.
    destination.mkdir()
    rig.baseline.save(destination / "baseline.npz")
    (destination / "meta_data.json").write_text(json.dumps(dict(
        baseline_path=str(f.source), raw_camera=f.binding)), encoding="utf-8")
    for method in ("load_corrections", "setup_depth", "setup_labels", "setup_inner_labels",
                   "setup_facies", "setup_facies_props", "setup_geometry",
                   "setup_image_porosity", "setup_boolean_image_porosity"):
        monkeypatch.setattr(Rig, method, lambda *a, **k: None)
    restored = Rig.load(destination)
    assert restored.raw_camera == f.binding
    assert restored.baseline_path == f.source
    restored._read_optical_image(f.source)
    assert len(f.calls) == 1


def test_raw_save_without_exclusive_operation_refuses_before_any_writer(bound_input, tmp_path):
    rig = Rig()
    rig.raw_camera = bound_input.binding
    destination = tmp_path / "unowned"
    with pytest.raises(ValueError, match="exclusive build context"):
        rig.save(destination)
    assert not destination.exists()


@pytest.mark.parametrize("raw_input", [False, True])
def test_raw_post_shape_disk_diagnostics_are_suppressed_without_changing_setup_or_display(
        bound_input, tmp_path, monkeypatch, raw_input):
    rig, calls = Rig(), []
    log = tmp_path / "figures"
    baseline, labels, porosity = object(), object(), object()
    def reading(*args, **kwargs):
        calls.append(("reading", kwargs.get("log"), kwargs.get("show_plot")))
        rig.baseline = baseline
    def stage(name):
        def run(*args, **kwargs):
            calls.append((name, kwargs.get("log"), kwargs.get("show_plot")))
            if name == "labels": rig.labels = labels
            if name == "image_porosity": rig.image_porosity = porosity
            if name == "color": rig.color_corrections = [object()]
        return run
    monkeypatch.setattr(rig, "setup_reading", reading)
    for method, name in (("setup_depth", "depth"), ("setup_labels", "labels"),
            ("setup_inner_labels", "inner"), ("setup_facies", "facies"),
            ("setup_facies_props", "properties"), ("setup_geometry", "geometry"),
            ("setup_image_porosity", "image_porosity"),
            ("setup_boolean_image_porosity", "boolean_porosity"), ("setup_color_corrections", "color")):
        monkeypatch.setattr(rig, method, stage(name))
    class FinishedScientificSetup(Exception): pass
    def averaging(*args):
        assert args == (labels, porosity, baseline)
        raise FinishedScientificSetup
    monkeypatch.setattr(darsia, "porosity_based_averaging", averaging)
    camera = bound_input.binding if raw_input else None
    context = SimpleNamespace(camera=camera, raw_source=dict(sha256="a" * 64),
        operation_id="operation", begin_setup=lambda *args: None)
    with pytest.raises(FinishedScientificSetup):
        rig.setup(SimpleNamespace(experiment_start=None), bound_input.source,
            Path("depth"), Path("labels"), facies_path=Path("facies"), log=log,
            show_plot=True, raw_camera=camera, raw_baseline_sha256="a" * 64,
            raw_build_id="operation", raw_build=context if raw_input else None)
    assert calls[0] == ("reading", log, True)
    expected_log = None if raw_input else log
    for name, actual_log, display in calls[1:]:
        if name not in {"properties", "geometry"}:
            assert actual_log == expected_log
        if name in {"facies", "image_porosity", "boolean_porosity", "color"}:
            assert display is True
    assert [name for name, _, _ in calls].count("image_porosity") == 2
    assert [name for name, _, _ in calls].count("boolean_porosity") == 2
    assert rig.baseline is baseline and rig.labels is labels and rig.image_porosity is porosity
