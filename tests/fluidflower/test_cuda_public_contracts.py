"""Tiny actual-device contracts for the unchanged optional CUDA provider.

These are unit operands, not acquired frames, calibration fits, a throughput
benchmark, or a scientific qualification. Missing CUDA is an honest skip.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass.mass_cuda import CudaHostPixelMath, CudaPixelKernels


@pytest.fixture(params=["float32", "float64"])
def host(request):
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA driver unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA device available")
    value = CudaHostPixelMath(request.param, device_id=0,
                             pool_limit_bytes=128 * 1024 * 1024)
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize("constructor", [CudaPixelKernels, CudaHostPixelMath])
@pytest.mark.parametrize("kwargs,message", [
    ({"device_id": -1, "pool_limit_bytes": 1}, "device index"),
    ({"device_id": False, "pool_limit_bytes": 1}, "device index"),
    ({"device_id": 0, "pool_limit_bytes": 0}, "device budget"),
    ({"device_id": 0, "pool_limit_bytes": True}, "device budget"),
])
def test_constructor_rejects_ambiguous_resource_admission(constructor, kwargs, message):
    with pytest.raises(ValueError, match=message):
        constructor("float32", **kwargs)


def test_device_scope_transfers_warps_events_and_releases_owned_pool(host):
    device = host.device_math
    dtype = host.dtype
    expected = np.arange(18, dtype=dtype).reshape(2, 3, 3)
    with pytest.raises(RuntimeError, match="scoped allocation"):
        device.array(expected)
    with device.scope("unit-device-contract"):
        array = device.array(expected)
        uploaded = device.upload(expected + dtype(1))
        np.testing.assert_array_equal(device.download(array), expected)
        np.testing.assert_array_equal(device.download(uploaded), expected + dtype(1))
        warped = device.warp_continuous(expected, np.eye(3), (3, 2))
        np.testing.assert_array_equal(device.download(warped), expected)
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with device.scope("nested-not-admitted"):
                pytest.fail("Nested scope must never enter")
    assert not device._active
    facts = device.facts()
    assert facts["backend"] == "cuda"
    assert facts["precision"] == host.precision
    assert facts["cpu_decode"] and not facts["gpu_decode"]
    assert not facts["scientific_eligibility"]
    assert not facts["production_permission_granted"]
    event = facts["events"][-1]
    assert event["stage"] == "unit-device-contract"
    assert event["wall_ns"] > 0 and event["cuda_elapsed_ms"] >= 0
    assert 0 <= event["pool_used_bytes"] <= event["pool_reserved_bytes"]
    assert event["device_global_used_bytes"] < event["device_total_bytes"] * .9
    del array, uploaded, warped
    device.close()
    assert device.pool.total_bytes() == 0
    host_facts = host.facts()
    assert host_facts["host_api"] == "PixelMath"
    assert host_facts["cpu_facies_scatter"]
    assert host_facts["cpu_float64_reducers"]
    assert not host_facts["cached_scientific_arrays"]
    host.close()
    assert device.pool.total_bytes() == 0


def test_host_pixel_dtype_decode_and_identity_warp(host):
    import cv2

    dtype = host.dtype
    values = host.pixels([[0, 1, 2], [3, 4, 5]])
    assert values.dtype == np.dtype(dtype)
    assert host.require_pixels(values) is values
    other_dtype = np.float64 if host.precision == "float32" else np.float32
    with pytest.raises(TypeError, match="working dtype"):
        host.require_pixels(values.astype(other_dtype))
    storage = np.zeros((2, 3, 3), np.float32)
    decoded = host.decode_log_rgb(storage)
    np.testing.assert_array_equal(decoded, np.full(storage.shape, dtype(1) / dtype(65535), dtype))
    assert decoded.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(storage, np.zeros_like(storage))
    with pytest.raises(TypeError, match="storage must remain float32"):
        host.decode_log_rgb(storage.astype(np.float64))
    rgb = host.pixels(np.arange(18).reshape(2, 3, 3))
    np.testing.assert_array_equal(
        host.warp_continuous(rgb, np.eye(3), (2, 3), cv2_module=cv2), rgb)
    with pytest.raises(ValueError, match="positive native"):
        host.warp_continuous(rgb, np.eye(3), (0, 3), cv2_module=cv2)


def test_normalization_background_modes_and_facies_scatter(host):
    dtype = host.dtype
    current = host.pixels(np.full((2, 2, 3), .75))
    baseline = host.pixels(np.full((2, 2, 3), .25))
    clear = host.pixels(np.full((2, 2, 3), .5))
    support = np.array([[True, False], [False, True]])
    a, b = host.reference_normalized(current, baseline, clear, support)
    expected_a = np.zeros_like(current)
    expected_b = np.zeros_like(baseline)
    expected_a[support] = dtype(1.5)
    expected_b[support] = dtype(.5)
    np.testing.assert_array_equal(a, expected_a)
    np.testing.assert_array_equal(b, expected_b)
    invalid_clear = clear.copy()
    invalid_clear[0, 0, 0] = 0
    with pytest.raises(ValueError, match="clear-water support"):
        host.reference_normalized(current, baseline, invalid_clear, support)
    with pytest.raises(TypeError, match="boolean normalization"):
        host.reference_normalized(current, baseline, clear, support.astype(np.uint8))
    for option, expected in [("positive", .5), ("negative", 0),
                             ("plain", .5), ("absolute", .5)]:
        np.testing.assert_array_equal(host.relative_rgb(current, baseline, option=option),
                                      np.full(current.shape, expected, dtype))
    with pytest.raises(ValueError, match="Unknown original"):
        host.relative_rgb(current, baseline, option="unsupported")
    labels = np.array([[0, 1], [2, 1]], np.int16)
    signal = host.pixels(np.ones((2, 2, 3)))
    models = {1: lambda x: x[:, 0] * dtype(2), 2: lambda x: x[:, 0] * dtype(4)}
    result = host.per_facies(signal, labels, models, ignore_labels=(0,))
    np.testing.assert_array_equal(result, np.array([[0, 2], [4, 2]], dtype))
    with pytest.raises(ValueError, match="categorical facies grid"):
        host.per_facies(signal, labels.astype(float), models)


def test_color_path_extrapolation_and_explicit_lut_endpoints(host):
    dtype = host.dtype
    supports = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    colors = host.pixels([[-.5, 0, 0], [.25, 0, 0], [1.5, 0, 0], [3, 0, 0]])
    parameter = host.fit_color_path(colors, supports, [0, 1, 2])
    np.testing.assert_array_equal(parameter, np.array([-.5, .25, 1.5, 3], dtype))
    interpreted = host.interpret_color_path(parameter, [0, 1, 2], [10, 20, 40])
    np.testing.assert_array_equal(interpreted, np.array([5, 12.5, 30, 60], dtype))
    with pytest.raises(ValueError, match="zero-length segment"):
        host.fit_color_path(colors, np.zeros((3, 3)), [0, 1, 2])
    signal = host.pixels([.5, 1, 1.5])
    np.testing.assert_array_equal(host.piecewise_linear(signal, [0, 1, 2], [0, 10, 30]),
                                  np.array([5, 10, 20], dtype))
    outside = host.pixels([-.5, 3])
    with pytest.raises(ValueError, match="outside original"):
        host.piecewise_linear(outside, [0, 1, 2], [0, 10, 30])
    np.testing.assert_array_equal(
        host.piecewise_linear(outside, [0, 1, 2], [0, 10, 30], endpoint="clamp"),
        np.array([0, 30], dtype))
    with pytest.raises(ValueError, match="endpoint rule"):
        host.piecewise_linear(signal, [0, 1], [0, 1], endpoint="unknown")


def test_titration_uses_supplied_frozen_lut_and_rejects_restoration(host):
    flash = SimpleNamespace(restoration=None, min_value_aq=0., max_value_aq=2.,
                            min_value_g=0., max_value_g=4., aqueous_blend=.25,
                            _lut_y=np.array([0., 1.]), _lut_caq=np.array([0., 2.]))
    signal = host.pixels([0, 1, 2, 3, 4])
    aqueous, gas = host.titration(signal, flash)
    np.testing.assert_array_equal(aqueous, np.array([0, .875, 1.75, 1.75, 1.75], host.dtype))
    np.testing.assert_array_equal(gas, np.array([0, .25, .5, .75, 1], host.dtype))
    flash.restoration = object()
    with pytest.raises(ValueError, match="Restoration needs"):
        host.titration(signal, flash)


def test_hydrostatic_tables_have_known_pressure_and_temperature_interpolation(host):
    kwargs = dict(temperature=host.pixels([[20, 23]]), height=host.pixels([[1, 1]]),
                  top_pressure=host.pixels([[2, 2]]), water_density_20=1000.,
                  water_density_23=1000., density_20=([1, 2, 3], [2, 4, 6]),
                  density_23=([1, 2, 3], [4, 8, 12]),
                  solubility_20=([1, 2, 3], [.25, .5, .75]),
                  solubility_23=([1, 2, 3], [.5, 1, 1.5]))
    result = host.hydrostatic_fields(**kwargs)
    eps = np.finfo(host.dtype).eps
    np.testing.assert_allclose(result["pressure"], [[2.0981, 2.0981]], rtol=4*eps, atol=0)
    np.testing.assert_allclose(result["density"], [[4.1962, 8.3924]], rtol=4*eps, atol=0)
    np.testing.assert_allclose(result["solubility"], [[.524525, 1.04905]], rtol=4*eps, atol=0)
    assert all(v.dtype == np.dtype(host.dtype) for v in result.values())
    kwargs["top_pressure"] = host.pixels([[10, 10]])
    with pytest.raises(ValueError, match="outside original"):
        host.hydrostatic_fields(**kwargs)


def test_selected_n1_fields_preserve_support_overrides_and_mass_partition(host):
    dtype = host.dtype
    prepared = host.pixels([[.25, .5], [.75, 1]])
    aq = np.array([[True, False], [False, True]])
    gas = np.array([[False, True], [False, True]])
    kwargs = dict(gain=2., factor=.5, swi=host.pixels([[.5, .25], [.5, .75]]),
                  aqueous_known=aq, gas_known=gas, solubility=host.pixels(np.full((2, 2), 2)),
                  gas_density=host.pixels(np.full((2, 2), 4)))
    result = host.selected_n1_fields(prepared, **kwargs)
    expected = {"aqueous": [[.25, 1], [0, 1]], "gas": [[0, .75], [0, .25]],
                "concentration": [[.5, 2], [0, 2]], "mass_aqueous": [[.5, .5], [0, 1.5]],
                "mass_gas": [[0, 3], [0, 1]], "mass_total": [[.5, 3.5], [0, 2.5]]}
    assert result.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(result[name], np.array(value, dtype))
    np.testing.assert_array_equal(result["mass_total"], result["mass_aqueous"]+result["mass_gas"])
    with pytest.raises(TypeError, match="boolean supports"):
        host.selected_n1_fields(prepared, **dict(kwargs, aqueous_known=aq.astype(np.uint8)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_host_integrate_keeps_cpu_float64_domain_reduction(dtype):
    value = np.array([[1, 2], [3, 4]], dtype)
    weight = np.array([[.25, .5], [1, 2]], dtype)
    mask = np.array([[True, False], [True, True]])
    result = CudaHostPixelMath.integrate(value, weight, mask)
    assert result.dtype == np.float64 and result == 11.25
    with pytest.raises(ValueError, match="domain mask"):
        CudaHostPixelMath.integrate(value, weight, mask.astype(np.uint8))
