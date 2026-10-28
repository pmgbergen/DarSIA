"""Ordering tests with real full-photo grids and independent crop arithmetic."""
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower.native import full_photometry as subject
from .test_build_native_bridge import inputs as bridge_inputs


def test_full_photo_mode_requires_saved_choice_in_the_actual_source_binding(bridge_inputs):
    from darsia.fluidflower.native.raw_rig_source import validate_binding
    from darsia.fluidflower.mass.n1_pipeline import DataReader, asset
    value = deepcopy(bridge_inputs.binding)
    parameters = DataReader().json(bridge_inputs.context.provenance["native_configuration"])
    value["correction_order"] = subject.ORDER
    with pytest.raises(ValueError, match="actual saved"):
        validate_binding(value, parameters)
    value["normalisation"] = dict(choice=dict(method="off"), record=asset(bridge_inputs.binding_path))
    validate_binding(value, parameters)
    assert bridge_inputs.calls.corrections == []


def test_crop_interpolates_corrected_linear_signal_not_log_and_keeps_unknowns():
    linear = np.broadcast_to(np.array([[1., 9., 25.], [1., 9., 25.]], np.float32), (3, 2, 3)).copy()
    h = np.array([[1., 0., .5], [0., 1., 0.], [0., 0., 1.]])
    actual = subject.crop_corrected(np.log(linear), h, (2, 2), coverage_tolerance=1e-6)
    assert np.allclose(np.exp(actual), np.broadcast_to([[5., 17.], [5., 17.]], actual.shape), rtol=2e-7)
    assert not np.isclose(np.exp(actual[0, 0, 0]), np.sqrt(9.))
    log = np.log(linear)
    log[:, 0, 0] = np.nan
    actual = subject.crop_corrected(log, h, (2, 2), coverage_tolerance=1e-6)
    assert np.isnan(actual[:, 0, 0]).all() and np.isfinite(actual[:, 1, 0]).all()


def test_complete_photo_supplies_chart_outside_crop_before_correction(tmp_path, monkeypatch):
    full = np.full((3, 6, 8), 4., np.float32)
    full[:, 2, 1] = 100.  # Checker excluded by the later common crop.
    h = np.array([[1., 0., 4.], [0., 1., 1.], [0., 0., 1.]])
    baseline = SimpleNamespace(source=dict(sha256="baseline"))
    sensor = SimpleNamespace(source=dict(sha256="current"))
    choice = dict(method="match_reference_lighting", reference_run="chosen-reference")
    parameters = dict(registration=dict(common=dict(sensor=dict(camera_wh=[8, 6], content_wh=[3, 3]))),
        photometry=dict(minimum_sensor_signal=0., native_shape_chw=[3, 6, 8],
            area_grid_shape_chw=[3, 3, 4], native_area_log_int_8=2),
        support=dict(coverage_tolerance=1e-6), source=dict(symbols=dict(reference_to_frame_h_key="reference_h")))
    events = []
    def render(actual, *, camera_u):
        assert camera_u is full
        events.append("geometry")
        return None, dict(native_to_camera_u_h=h.tolist())
    owner = SimpleNamespace(parameters=parameters, processor=SimpleNamespace(fit=SimpleNamespace(CB=np.zeros((3, 24))),
        reference=object(), grid=object(), old_field=None), baseline_sensor=baseline,
        baseline_camera_u=np.ones_like(full), baseline_renderer=dict(native_to_camera_u_h=np.eye(3).tolist()),
        binding=dict(contract="actual-raw", native_shape_hw=[3, 3], normalisation=dict(choice=choice)),
        identity={}, _render=render)
    monkeypatch.setattr(subject.raw_positioning, "camera_u", lambda *a, **k: full)
    monkeypatch.setattr(subject.source, "chart_baseline_coordinates", lambda *a, **k: np.zeros((24, 2)))
    def register(**kwargs):
        events.append("chart")
        assert kwargs["parameters"]["common"]["sensor"]["content_wh"] == [8, 6]
        assert np.array_equal(kwargs["frame_native_to_u_h"], np.eye(3))
        assert kwargs["frame_native_linear_rgb"][:, 2, 1].tolist() == [100., 100., 100.]
        return np.ones((24, 2), np.float32), np.ones(24, bool), {}
    monkeypatch.setattr(subject.raw_chart, "register", register)
    monkeypatch.setattr(subject.photometry, "chart_xy_to_reference", lambda xy, *a, **k: xy)
    def light(area, *args, **kwargs):
        events.append("light")
        assert np.isclose(np.exp(area[0, 1, 0]), 28.)  # Mean includes the out-of-crop checker.
        return np.full((3, 4), np.log(2.), np.float32), dict(fitted_on_full_photo=True)
    monkeypatch.setattr(subject.photometry, "frame_light_from_native_grid", light)
    def correction(log_full, *args, **kwargs):
        events.append("colour-full")
        assert log_full.shape == full.shape
        assert np.isclose(np.exp(log_full[0, 2, 1]), 100.)
        # Nonlinear transform makes correction-before-interpolation observable.
        return (2 * log_full + np.float32(np.log(2.))).astype(np.float32)
    monkeypatch.setattr(subject.photometry, "apply_native", correction)
    before = deepcopy(parameters)
    actual, audit = subject.correct(owner, sensor, np.linalg.inv(h), np.eye(3), np.eye(3))
    assert events == ["geometry", "chart", "light", "colour-full"]
    assert np.allclose(np.exp(actual), 32., rtol=2e-7)
    assert audit["correction_order"] == subject.ORDER
    assert audit["chart"]["samples_before_common_crop"]
    assert parameters == before
    owner.binding["normalisation"]["choice"] = dict(method="off")
    events.clear()
    actual, audit = subject.correct(owner, sensor, np.linalg.inv(h), np.eye(3), np.eye(3))
    assert events == ["geometry"]
    assert np.allclose(np.exp(actual), 4., rtol=2e-7)
    assert audit["photometry"]["per_frame_light"]["mode"] == "off"


def test_reference_cell_means_use_full_photo_and_explicit_measured_transform():
    source = np.broadcast_to(np.arange(1, 49, dtype=np.float32).reshape(6, 8), (3, 6, 8)).copy()
    parameters = dict(native_shape_chw=[3, 4, 4], area_grid_shape_chw=[3, 2, 2],
                      native_area_log_int_8=2, minimum_sensor_signal=0.)
    h = np.array([[1., 0., 2.], [0., 1., 1.], [0., 0., 1.]])
    actual = subject.reference_area_log(source, h, parameters=parameters)
    expected = source[:, 1:5, 2:6].reshape(3, 2, 2, 2, 2).mean(axis=(2, 4), dtype=np.float64)
    assert np.allclose(actual, np.log(expected).astype(np.float32), rtol=0, atol=0)


def test_current_baseline_collection_uses_raw_and_build_geometry_without_old_models(bridge_inputs, monkeypatch):
    value=bridge_inputs
    calls=[]
    sensor=SimpleNamespace(metadata=dict(white_level=110,black_level=10))
    def read(path,**kwargs):
        calls.append((path,kwargs))
        return sensor
    monkeypatch.setattr(subject.raw_positioning,'read_sensor',read)
    monkeypatch.setattr(subject.photometry,'load_baseline',lambda *a,**k:pytest.fail('old native model was opened'))
    actual=subject.collect_baseline(value.context,require_chart=False)
    assert len(calls)==1 and calls[0][1]['expected_sha256']==value.baseline['sha256']
    assert actual['rgb_full'].shape==(3,2,3) and actual['rgb_native'].shape==(2,3,3)
    np.testing.assert_array_equal(actual['native_to_camera_u_h'],value.matrices['hb'])
    assert actual['source_pins']['raw_acquisition']==value.baseline
    assert actual['chart_rgb'] is None and actual['chart_roi_rc'] is None
