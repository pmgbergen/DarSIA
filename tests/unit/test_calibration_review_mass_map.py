"""Small display-contract fixtures, not producer qualification; tests unrun.

The real immutable readers retain provenance/closure coverage in the storage
tests. These asymmetric arrays exercise only the mass display's view contract.
"""
import numpy as np
import pytest

from darsia.fluidflower import calibration_review_mass_map as rendering


@pytest.fixture
def stored():
    arrays = {
        "cell_mass_aqueous_g": np.array([[0., .1, .2], [.3, .4, 0.]], np.float32),
        "cell_mass_gas_g": np.array([[0., .2, .3], [.4, .5, 0.]], np.float32),
        "cell_mass_total_g": np.array([[0., .9, 1.3], [1.7, 2.1, 0.]], np.float32),
        "static_cell_has_area": np.array([[True, True, True], [True, True, False]]),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return dict(format="darsia-read-coarse-stored-frame-v1", producer_kind="n1-mass-only",
        precision="float32", imshow_origin="lower", run="synthetic", case_id="display-only",
        storage_document=dict(format="darsia-n1-storage-frame-v1", producer_kind="n1-mass-only",
                              complete=True, precision="float32", existing_native_products_unchanged=True),
        grid=dict(format="darsia-metric-storage-grid-v1", shape=[2, 3],
                  axes="row=y increasing, column=x increasing", storage_precision="float32",
                  extent_m=[.01, .0125, .02, .022], resolution_mm=1.), arrays=arrays)


def _snapshot(stored):
    return {key: (array.dtype.str, array.shape, array.tobytes(), array.flags.writeable)
            for key, array in stored["arrays"].items()}


@pytest.mark.parametrize("component", ["aqueous", "gas", "total"])
def test_uses_each_independent_saved_mass_and_preserves_float32_bytes(stored, component):
    before = _snapshot(stored)
    arrays = stored["arrays"]
    assert not np.array_equal(arrays["cell_mass_total_g"],
                              arrays["cell_mass_aqueous_g"] + arrays["cell_mass_gas_g"])
    figure = rendering.mass_figure(stored, component, 2.5)
    try:
        image = figure.axes[0].collections[0]
        np.testing.assert_array_equal(image.get_array().data, arrays["cell_mass_" + component + "_g"])
        np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), [[False, False, False],
                                                                             [False, False, True]])
        assert image.get_array().dtype == np.float32
        assert image.get_clim() == (0., 2.5)
        assert "1 mm stored cells" in figure.axes[0].get_title()
        assert figure.axes[1].get_ylabel() == "g per stored cell · fixed linear scale"
    finally:
        rendering.release_stored_figure(figure)
    assert _snapshot(stored) == before


def test_metric_mesh_preserves_orientation_fixed_internal_edges_and_last_endpoint(stored):
    figure = rendering.mass_figure(stored, "total", 2.5)
    try:
        axis = figure.axes[0]
        image = axis.collections[0]
        edges = image.get_coordinates()
        np.testing.assert_allclose(edges[0, :, 0], [.01, .011, .012, .0125], rtol=0, atol=1e-16)
        np.testing.assert_allclose(edges[:, 0, 1], [.02, .021, .022], rtol=0, atol=1e-16)
        assert edges[0, -1, 0] == stored["grid"]["extent_m"][1]
        assert axis.get_xlim() == (.01, .0125) and axis.get_ylim() == (.02, .022)
        assert image.get_array()[0, 1] == np.float32(.9)
        assert image.get_array()[1, 0] == np.float32(1.7)
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        for x, y, value in ((.0109, .0205, np.float32(0)),
                            (.0115, .0205, np.float32(.9)),
                            (.01225, .0205, np.float32(1.3)),
                            (.0105, .0215, np.float32(1.7))):
            px, py = axis.transData.transform((x, y))
            actual = rgba[rgba.shape[0] - 1 - int(py), int(px), :3]
            expected = np.floor(np.asarray(image.cmap(image.norm(value))[:3]) * 255. + .5).astype(np.uint8)
            np.testing.assert_array_equal(actual, expected)
    finally:
        rendering.release_stored_figure(figure)


def _unknown_scope(stored):
    for name in ("cell_mass_total_g", "cell_mass_aqueous_g", "cell_mass_gas_g"):
        values = stored["arrays"][name].copy()
        values[0, 2] = 0.  # Fully unknown cell has a zero quantified carrier.
        values.setflags(write=False)
        stored["arrays"][name] = values
    unknown = np.array([[0., .25, 1.], [0., 0., np.nan]], np.float32)
    unknown.setflags(write=False)
    stored["arrays"]["mask_approved_analysis_gap_area_fraction"] = unknown
    stored.update(analysis_scope=dict(status="quantified_subtotal_with_unquantified_remainder",
                  whole_rig_mass_complete=False, unknown_mass_g=None),
                  unknown_mass_g=None, mass_totals_g=dict.fromkeys(("aqueous", "gas", "total")))


def test_zero_is_valid_unknown_cells_are_hatched_and_subtotal_is_explicit(stored):
    _unknown_scope(stored)
    before = _snapshot(stored)
    figure = rendering.mass_figure(stored, "total", 2.5)
    try:
        axis = figure.axes[0]
        displayed = axis.collections[0].get_array()
        mask = np.ma.getmaskarray(displayed)
        assert displayed[0, 0] == 0 and not mask[0, 0]  # Known zero stays visible.
        assert displayed[0, 2] == 0 and not mask[0, 2]  # Carrier shown only with hatch.
        assert mask[1, 2]  # No coverage is distinct from either zero or unknown.
        assert len(axis.patches) == 1
        hatch = axis.patches[0]
        assert hatch.get_hatch() == "////"
        assert not hatch.get_path().contains_point((.0105, .0205))
        assert hatch.get_path().contains_point((.0115, .0205))
        assert hatch.get_path().contains_point((.01225, .0205))
        assert "Quantified contribution; unknown areas remain" in axis.get_title()
        assert "colour shows quantified contribution" in axis.texts[0].get_text()
    finally:
        rendering.release_stored_figure(figure)
    assert _snapshot(stored) == before and stored["unknown_mass_g"] is None


@pytest.mark.parametrize("limit", [None, False, True, 0, -1, float("nan"), float("inf"), "2.5"])
def test_no_implicit_or_invalid_mass_scale_is_admitted(stored, limit):
    with pytest.raises(ValueError, match="explicit fixed positive"):
        rendering.mass_figure(stored, "total", limit)


def test_native_phase_reader_format_remains_a_separate_supported_input(stored):
    stored.pop("producer_kind")
    stored["storage_document"].pop("producer_kind")
    stored["storage_document"]["format"] = "darsia-storage-frame-v1"
    figure = rendering.mass_figure(stored, "aqueous", 1.)
    rendering.release_stored_figure(figure)


@pytest.mark.parametrize("damage", ["reader", "producer", "dtype", "shape", "orientation", "extent"])
def test_incompatible_reader_or_grid_is_not_repaired(stored, damage):
    if damage == "reader":
        stored["format"] = "arbitrary-result"
    elif damage == "producer":
        stored["producer_kind"] = "unverified"
    elif damage == "dtype":
        stored["arrays"]["cell_mass_total_g"] = stored["arrays"]["cell_mass_total_g"].astype(np.float64)
    elif damage == "shape":
        stored["grid"]["shape"] = [3, 2]
    elif damage == "orientation":
        stored["grid"]["axes"] = "row=y decreasing"
    else:
        stored["grid"]["extent_m"][1] = .014
    with pytest.raises(ValueError):
        rendering.mass_figure(stored, "total", 2.5)


@pytest.mark.parametrize("damage", ["mask_missing", "mask_fraction", "whole_mass_claim", "scope_missing"])
def test_unknown_area_requires_its_mask_and_unknown_scope(stored, damage):
    _unknown_scope(stored)
    if damage == "mask_missing":
        stored["arrays"].pop("mask_approved_analysis_gap_area_fraction")
    elif damage == "mask_fraction":
        fraction = stored["arrays"]["mask_approved_analysis_gap_area_fraction"].copy()
        fraction[0, 1] = 1.1
        stored["arrays"]["mask_approved_analysis_gap_area_fraction"] = fraction
    elif damage == "whole_mass_claim":
        stored["mass_totals_g"]["total"] = 5.
    else:
        stored.pop("analysis_scope")
    with pytest.raises(ValueError):
        rendering.mass_figure(stored, "total", 2.5)


@pytest.mark.parametrize("failure", ["mesh", "colorbar"])
def test_figure_buffers_are_released_when_building_fails(stored, monkeypatch, failure):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    released = []
    original_release = rendering.release_stored_figure

    def release(figure):
        released.append(figure)
        original_release(figure)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic display failure")

    monkeypatch.setattr(rendering, "release_stored_figure", release)
    monkeypatch.setattr(Axes if failure == "mesh" else Figure,
                        "pcolormesh" if failure == "mesh" else "colorbar", fail)
    with pytest.raises(RuntimeError, match="synthetic display failure"):
        rendering.mass_figure(stored, "total", 2.5)
    assert len(released) == 1 and released[0].canvas is None
