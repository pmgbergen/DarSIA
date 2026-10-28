"""Synthetic fields and local files cover the remaining public native operations."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.native import extension, photometry as photo


@pytest.fixture
def parameters():
    fixture = Path(photo.__file__).resolve().parents[4] / "tests/fluidflower/fixtures/native_photometry_parameters.json"
    data = json.loads(fixture.read_bytes())
    # A small explicit test geometry; method thresholds and arithmetic stay original.
    data.update(grid_shape_hw=[48, 64], native_shape_hw=[64, 80],
        polynomial_basis_default_h=48, polynomial_basis_default_w=64,
        remap_to_reference_default_destination_shape=[48, 64], area_grid_shape_chw=[3, 48, 64],
        estimate_frame_light_grid_int_93=12, estimate_frame_light_grid_int_171=16)
    return data


@pytest.fixture
def baseline(parameters):
    shape = tuple(parameters["grid_shape_hw"])
    xy = np.column_stack([np.tile(np.arange(4) + 2., 6), np.repeat(np.arange(6) + 2., 4)]).astype(np.float32)
    chart = np.zeros(shape, bool)
    chart[2:8, 2:6] = True
    return photo.Baseline(run="fixture", M=np.eye(3), Minv=np.eye(3), P=np.zeros(14),
        a={1: 0.}, n={1: int(np.prod(shape))}, fac=np.ones(shape, np.int16),
        sand=~chart, chart=chart, cpl=np.zeros(3),
        Ug=np.zeros(shape, np.float32), Vg=np.zeros(shape, np.float32), taper=np.ones(shape, np.float32),
        logB=np.zeros((3, *shape), np.float32), gb0=np.zeros(shape, np.float32), xy=xy,
        CB=np.zeros((3, 24), np.float32), F=None, residual_exclusion_fraction=0.)


def test_fit_baseline_grid_fits_neutral_field_and_rejects_wrong_lattice(parameters, baseline):
    kwargs = dict(run="fixture", logB=baseline.logB, facies=baseline.fac,
        chart_xy=baseline.xy, chart_log_rgb=baseline.CB, ref_chart_log_rgb=baseline.CB,
        chart_usable=np.ones(24, bool), matrix=np.eye(3), rig_consensus=baseline.logB)
    fit = photo.fit_baseline_grid(parameters=parameters, **kwargs)
    assert fit.P.shape == (14,) and fit.P.dtype == np.float64
    assert np.array_equal(fit.P, np.zeros(14)) and fit.a == {1: 0.}
    assert fit.n[1] >= 1000 and fit.logB.dtype == np.float32
    assert not fit.sand[2, 2] and fit.sand[-1, -1]
    with pytest.raises(ValueError, match="Facies, chart or matrix shape mismatch"):
        photo.fit_baseline_grid(parameters=parameters, **dict(kwargs, chart_xy=np.zeros((23, 2))))


def test_baseline_field_returns_supported_zero_and_rejects_disjoint_facies(parameters, baseline):
    field, sand = photo.baseline_field(baseline, baseline, parameters=parameters)
    assert field.dtype == np.float32 and np.array_equal(field, np.zeros(baseline.fac.shape, np.float32))
    assert sand.dtype == bool and np.array_equal(sand, ~baseline.chart)
    with pytest.raises(ValueError, match="No common facies"):
        photo.baseline_field(baseline, replace(baseline, a={2: 0.}), parameters=parameters)


def test_registered_and_reference_corrections_preserve_identity_and_reject_unbound_inputs(parameters, baseline):
    original = baseline.logB.copy()
    registered = photo.apply_registered(original, baseline, baseline, tile_rows=7, parameters=parameters)
    referenced = photo.apply_reference_geometry(original, baseline, baseline, np.eye(3), tile_rows=7, parameters=parameters)
    assert registered.dtype == referenced.dtype == np.float32
    assert np.array_equal(registered, original) and np.array_equal(referenced, original)
    assert np.array_equal(baseline.logB, original)
    native = np.zeros((3, *parameters["native_shape_hw"]), np.float32)
    with pytest.raises(ValueError, match="source-bound native facies"):
        photo.apply_registered(native, baseline, baseline, parameters=parameters)
    with pytest.raises(ValueError, match="bound optical-support mask"):
        photo.apply_reference_geometry(native, baseline, baseline, np.eye(3), parameters=parameters)
    with pytest.raises(ValueError, match="frozen reference grid"):
        photo.apply_reference_geometry(original, baseline, baseline, np.eye(3),
            light_field_ref8=np.zeros((1, 1), np.float32), parameters=parameters)


def test_response_loading_rendering_and_invalid_encoding(tmp_path, parameters):
    response = dict(M=np.eye(3).tolist(), luts=[[[-1., 1.], [0., 255.]]] * 3)
    path = tmp_path / "response.json"
    path.write_text(json.dumps({"response": response}), encoding="utf-8")
    loaded = photo.load_response(path, parameters=parameters)
    assert loaded == response
    signal = np.zeros((3, 2, 3), np.float32)
    signal[:, 1, 2] = np.nan
    rendered = photo.render_response(signal, loaded, parameters=parameters)
    assert rendered.dtype == np.float32 and rendered.shape == (2, 3, 3)
    assert np.array_equal(rendered[0, 0], np.full(3, .5, np.float32))
    assert np.array_equal(rendered[1, 2], np.ones(3, np.float32))
    assert np.isnan(signal[:, 1, 2]).all()
    with pytest.raises(ValueError, match="Invalid frozen response"):
        photo.render_response(signal, dict(response, M=[[1.]]), parameters=parameters)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="response"):
        photo.load_response(path, parameters=parameters)


def test_grid_baseline_persists_exact_arrays_and_source_binding(tmp_path, parameters, baseline):
    provenance = dict(source_arw_sha256="1" * 64, source_review_sha256="2" * 64,
                      source_manifest_sha256="3" * 64, facies_sha256="4" * 64)
    path = tmp_path / "fits" / "baseline.npz"
    receipt = photo.save_grid_baseline(path, baseline, provenance, parameters=parameters)
    assert receipt["npz_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert receipt["source"] == provenance
    assert json.loads(path.with_suffix(".json").read_text()) == receipt
    with np.load(path, allow_pickle=False) as arrays:
        assert np.array_equal(arrays["logB"], baseline.logB)
        assert arrays["fac"].dtype == np.int16 and arrays["sand"].dtype == bool
        assert arrays["has_F"].item() is False and arrays["F"].size == 0
    with pytest.raises(ValueError, match="must be a .npz"):
        photo.save_grid_baseline(tmp_path / "wrong.npy", baseline, provenance, parameters=parameters)
    with pytest.raises(ValueError, match="Missing baseline source binding"):
        photo.save_grid_baseline(tmp_path / "unbound.npz", baseline, {}, parameters=parameters)
    assert not (tmp_path / "unbound.npz").exists()


def test_green_donor_receipt_does_not_promote_ineligible_support(parameters):
    receipt = dict(photo_sand_qa_scope="green_donor_mask_only",
        green_donor_only_optical_fit_eligible=True, photo_sand_qa_pass=True)
    assert photo.require_green_donor_only_receipt(receipt, parameters=parameters) is None
    with pytest.raises(ValueError, match="not fit-eligible"):
        photo.require_green_donor_only_receipt(dict(receipt, green_donor_only_optical_fit_eligible=False), parameters=parameters)


def test_reference_frame_light_is_zero_for_unchanged_chart_and_signal(parameters, baseline):
    field, audit = photo.estimate_frame_light_reference_geometry(baseline, baseline.logB, np.eye(3),
        frame_chart_xy_run8=baseline.xy, frame_chart_usable=np.ones(24, bool), parameters=parameters)
    assert field.shape == baseline.fac.shape and field.dtype == np.float32
    assert np.array_equal(field, np.zeros_like(field))
    assert audit["n_chart"] == 24 and audit["chart_log_drift"] == 0.
    assert audit["frame_chart_source"] == "exact-run-frame-registered-then-projected"
    with pytest.raises(ValueError, match="Exact frame chart centres"):
        photo.estimate_frame_light_reference_geometry(baseline, baseline.logB, np.eye(3),
            frame_chart_xy_run8=baseline.xy[:23], frame_chart_usable=np.ones(24, bool), parameters=parameters)


def test_native_grid_frame_light_preserves_frozen_centre_and_checks_dtypes(parameters, baseline):
    grid = photo.ReferenceGrid(baseline.fac.shape, (0, 0), baseline.fac.shape, parameters=parameters)
    field, audit = photo.frame_light_from_native_grid(baseline.logB, baseline, grid, np.eye(3),
        baseline.xy, np.ones(24, bool), parameters=parameters)
    assert field.dtype == np.float32 and np.array_equal(field, np.zeros_like(field))
    assert audit["n_chart"] == 24 and audit["central_frozen_v5_parity"] is True
    with pytest.raises(TypeError, match="CHW float32 area-log grid"):
        photo.frame_light_from_native_grid(baseline.logB.astype(np.float64), baseline, grid,
            np.eye(3), baseline.xy, np.ones(24, bool), parameters=parameters)
    with pytest.raises(TypeError, match="usability must be boolean"):
        photo.frame_light_from_native_grid(baseline.logB, baseline, grid, np.eye(3),
            baseline.xy, np.ones(24, np.uint8), parameters=parameters)


def test_sensor_window_samples_declared_coordinates_and_rejects_wrong_sensor():
    lens = dict(knots=[0., 1.], distortion_denominator=16384., ca_denominator=2**21,
                raw_offset_xy=[0, 0])
    parameters = dict(extension=dict(homography_min_determinant=1e-12, tile_rows=2),
        registration=dict(common=dict(sensor=dict(raw_wh=[7, 6]), lens=lens,
            profiles={"fixture": dict(coefficients=dict(cu=0., cv=0., tx=0., ty=0., R=10., amp=1.), distortion=[0, 0])})))
    renderer = dict(review_to_camera_h=np.eye(3).tolist(), profile="fixture",
                    sony_distortion_table=[0, 0], sony_ca_red_table=[0, 0], sony_ca_blue_table=[0, 0])
    image = np.arange(6 * 7 * 3, dtype=np.uint16).reshape(6, 7, 3)
    actual = extension.sample_sensor_window(image, renderer, (1, 1, 4, 4), origin_xy=(1, 0), parameters=parameters)
    expected = image[1:4, 2:5].transpose(2, 0, 1).astype(np.float32)
    assert actual.dtype == np.float32 and np.array_equal(actual, expected)
    with pytest.raises(TypeError, match="full-resolution uint16 RAW demosaic"):
        extension.sample_sensor_window(image.astype(np.float32), renderer, (1, 1, 4, 4), origin_xy=(1, 0), parameters=parameters)
    with pytest.raises(ValueError, match="frozen Sony distortion profile"):
        extension.sample_sensor_window(image, dict(renderer, sony_distortion_table=[1, 0]),
            (1, 1, 4, 4), origin_xy=(1, 0), parameters=parameters)
