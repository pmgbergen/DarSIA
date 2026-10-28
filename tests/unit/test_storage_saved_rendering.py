"""Asymmetric saved-display tests; no RAW, native field archives or model calls."""
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import storage_saved_products as saved
from darsia.fluidflower import storage_saved_rendering as rendering
from .test_storage_saved_products import _fixture, _resolver, _alter_archive


def _asymmetric(tmp_path):
    state = _fixture(tmp_path)
    def change(arrays):
        arrays["concentration_aqueous"][0, 0] = .25
        arrays["concentration_aqueous"][0, 1] = .5
        arrays["concentration_aqueous"][1, 0] = .75
    _alter_archive(state, change)
    return state


def test_asymmetric_metric_axes_and_actual_pixels_preserve_saved_rows(tmp_path):
    state = _asymmetric(tmp_path)
    stored = saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))
    before = stored["arrays"]["concentration_aqueous"].copy()
    figure, image = rendering._figure(stored, "concentration", 1.)
    coordinates = image.get_coordinates()
    np.testing.assert_array_equal(coordinates[0, :, 0], [0., .001, .002, .003])
    np.testing.assert_array_equal(coordinates[:, 0, 1], [0., .001, .002, .003])
    assert image.axes.get_xlim() == (0., .003) and image.axes.get_ylim() == (0., .003)
    assert image.get_clim() == (0., 1.)
    assert image.get_array()[0, 0] == .25 and image.get_array()[1, 0] == .75
    assert np.ma.getmaskarray(image.get_array())[2].all()
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    for x, y, value in ((.0005, .0005, .25), (.0005, .0015, .75)):
        px, py = image.axes.transData.transform((x, y))
        actual = rgba[rgba.shape[0] - 1 - int(py), int(px), :3]
        expected = np.floor(np.asarray(image.cmap(image.norm(value))[:3]) * 255. + .5).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(stored["arrays"]["concentration_aqueous"], before)



def test_partial_last_metric_cells_keep_fixed_internal_edges_and_actual_pixels():
    # Display-only fixture: no synthetic provenance or source coordinate claim.
    values = np.array([[.15, .25, .35], [.45, .55, .65], [.75, .85, .95]], dtype=np.float32)
    values.setflags(write=False)
    before = values.copy()
    stored = dict(run="synthetic", case_id="partial-edge", before_rules=dict(state="complete"),
        grid=dict(extent_m=[.01, .0125, .02, .0225], resolution_mm=1., shape_hw=[3, 3]),
        arrays=dict(concentration_aqueous=values, concentration_valid_area_fraction=np.ones((3, 3))))
    figure, image = rendering._figure(stored, "concentration", 1.)
    coordinates = image.get_coordinates()
    np.testing.assert_array_equal(coordinates[0, :, 0], [.01, .011, .012, .0125])
    np.testing.assert_array_equal(coordinates[:, 0, 1], [.02, .021, .022, .0225])
    assert image.axes.get_xlim() == (.01, .0125) and image.axes.get_ylim() == (.02, .0225)
    assert image.get_clim() == (0., 1.)
    np.testing.assert_array_equal(image.get_array(), values)
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    # Near the first internal edges distinguishes fixed-width cells from imshow
    # stretching all three cells uniformly across the shorter 2.5 mm extent.
    for x, y, value in ((.0109, .0205, .15), (.0105, .0209, .15),
                        (.01225, .0205, .35), (.0105, .02225, .75),
                        (.01225, .02225, .95)):
        px, py = image.axes.transData.transform((x, y))
        actual = rgba[rgba.shape[0] - 1 - int(py), int(px), :3]
        expected = np.floor(np.asarray(image.cmap(image.norm(value))[:3]) * 255. + .5).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, before)
    assert not values.flags.writeable


def test_native_before_pixels_have_no_invented_coarse_metric_extent(tmp_path):
    state = _fixture(tmp_path)
    stored = saved.read_stored_frame(state["record"], resolve_asset=_resolver(state), load_before=True)
    figure, image = rendering._figure(stored, "before_rules_native_pixels", 1.)
    assert image.origin == "upper" and tuple(image.get_extent()) == (-.5, 1.5, 1.5, -.5)
    assert image.axes.get_xlabel() == "Native column (px)"
    assert image.axes.get_ylabel() == "Native row (px)"
    assert image.get_array()[0, 0] == 0 and image.get_array()[0, 1] == 1
    assert np.ma.getmaskarray(image.get_array())[1, 1]
    assert image.get_clim() == (0., 1.)


def test_explicit_native_before_render_reads_only_its_saved_png_and_records_pixel_axes(tmp_path):
    state = _fixture(tmp_path)
    output = rendering.render_stored_frame(tmp_path / "native_before.png", state["record"],
        resolve_asset=_resolver(state), field="before_rules_native_pixels")
    receipt = json.loads(Path(output["receipt"]["path"]).read_bytes())
    assert receipt["axes"] == dict(frame="native pixels", origin="upper", metric_extent=None,
        mapping_to_coarse_not_inferred=True)
    assert receipt["displayed_shape_hw"] == [2, 2] and receipt["fixed_limits"] == [0., 1.]
    assert [Path(p).name for p in state["calls"] if p.endswith(".png")] == ["storage_gas_probability_before_rules.png"]


@pytest.mark.parametrize("field", ["before_rules_coarse", "before_rules_native_pixels"])
def test_pending_before_produces_honest_placeholder_without_png_fallback(tmp_path, field):
    state = _fixture(tmp_path, pending=True)
    output = rendering.render_stored_frame(tmp_path / "display.png", state["record"],
        resolve_asset=_resolver(state), field=field)
    receipt = json.loads(Path(output["receipt"]["path"]).read_bytes())
    assert output["field_state"] == "pending" and receipt["displayed_shape_hw"] is None
    assert receipt["before_rules"]["reason"] == "native_centre_not_computed"
    assert receipt["photograph"]["state"] == "missing" and not receipt["final_original_left_film_ready"]
    assert not any(p.endswith(".png") for p in state["calls"])


def test_after_uses_all_pixel_count_fraction_even_where_physical_area_is_empty(tmp_path):
    state = _fixture(tmp_path)
    stored = saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))
    figure, image = rendering._figure(stored, "after_rules_coarse", 1.)
    assert not stored["arrays"]["static_cell_has_area"][1, 1]
    assert np.isnan(stored["arrays"]["phase_gas_area_fraction"][1, 1])
    assert image.get_array()[1, 1] == 0 and not np.ma.getmaskarray(image.get_array())[1, 1]
    assert image.get_array()[1, 0] == 1
    np.testing.assert_array_equal(image.get_coordinates()[:, 0, 1], [0., .001, .002, .003])
    assert image.axes.get_ylim() == (0., .003)


@pytest.mark.parametrize("limit", [None, 0., float("nan")])
def test_concentration_never_chooses_a_data_driven_or_default_scale(tmp_path, limit):
    state = _fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit fixed"):
        rendering.render_stored_frame(tmp_path / "display.png", state["record"],
            resolve_asset=_resolver(state), field="concentration", concentration_limit_g_l=limit)
    assert not state["calls"] and not (tmp_path / "display.png").exists()


def test_new_preview_receipt_is_exclusive_missing_photo_is_not_a_film_claim(tmp_path):
    state = _asymmetric(tmp_path)
    source_bytes = {p: Path(p).read_bytes() for p in (state["entry"]["receipt"]["path"],
        *(v["path"] for v in state["entry"]["files"]))}
    output = rendering.render_stored_frame(tmp_path / "display.png", state["record"],
        resolve_asset=_resolver(state), field="concentration", concentration_limit_g_l=1.)
    receipt = json.loads(Path(output["receipt"]["path"]).read_bytes())
    assert receipt["fixed_limits"] == [0., 1.] and receipt["axes"]["origin"] == "lower"
    assert receipt["photograph"] == dict(state="missing", reason="saved_registered_pre_light_product_not_bound",
        source_image_decoded=False, corrected_fallback_used=False)
    assert receipt["stored_mass_totals_g"] == dict(total=9., aqueous=3., gas=3.)
    assert not receipt["per_frame_normalization"] and not receipt["array_regridded"]
    assert not output["final_original_left_film_ready"]
    old_preview = Path(output["preview"]["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        rendering.render_stored_frame(tmp_path / "display.png", state["record"],
            resolve_asset=_resolver(state), field="concentration", concentration_limit_g_l=1.)
    assert Path(output["preview"]["path"]).read_bytes() == old_preview
    assert all(Path(path).read_bytes() == before for path, before in source_bytes.items())
