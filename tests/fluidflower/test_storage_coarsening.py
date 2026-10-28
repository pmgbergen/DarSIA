"""Output-only cell sums preserve verified data and explicit missing channels."""

from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower import storage_coarsening as coarsening
from darsia.fluidflower import storage_extensive as extensive
from darsia.fluidflower import storage_resolution as storage
from tests.unit.test_storage_extensive import _inputs, _product, _read, _totals


def make_payload(precision="float32", *, supplied_channel=False, equal_facies=False, missing_intensive=False):
    """Use the original extensive producer and reader on a small complete grid."""
    args = _inputs(precision)
    old = args["grid"]
    x = ((old.cell_index % old.shape[1]) + .25) * .001
    y = ((old.cell_index // old.shape[1]) + .25) * .001
    area = old.native_pixel_area_m2.copy()
    if equal_facies:
        area[:] = 1.0
        args["facies"] = np.array([[1, 2, 1], [2, 3, 255]], np.uint8)
        args["pixel_pore_volume_m3"] = area * args["depth_m"] * args["porosity"]
    args["grid"] = storage.make_grid(x, y, area, include=np.ones(x.shape, bool),
        extent_m=[0., .02, 0., .02], storage_resolution_mm=1.0, precision=precision)
    if missing_intensive:
        args["intensive_validity_masks"]["concentration_aqueous"][:] = False
    if supplied_channel:
        values, support = np.zeros(x.shape), np.zeros(x.shape, bool)
        values[1, 2], support[1, 2] = 2.0, True
        args["channels"]["U"] = dict(pixel_values_g=values, support_mask=support,
            native_total_g=2.0, producer_binding={"synthetic_supplied_channel": True})
    return _read(_product(_totals(args)))


def _large_and_small_payload():
    """A complete original-reader payload with non-associative positive sums."""
    shape = (20, 20)
    row, col = np.indices(shape)
    physical = np.ones(shape, bool)
    area = np.full(shape, 2.0 ** -20)
    grid = storage.make_grid((col + .25) * .001, (row + .25) * .001,
        area, include=physical, extent_m=[0., .02, 0., .02],
        storage_resolution_mm=1.0, precision="float64")
    values = np.full(shape, 2.0 ** -40)
    values[0, :] = 1.0
    values[0, 0] = 2.0 ** 53
    labels = np.ones(shape, np.uint8)
    args = dict(grid=grid, depth_m=np.ones(shape), porosity=np.full(shape, .5),
        pixel_pore_volume_m3=area * .5, facies=(1 + (row + col) % 2).astype(np.uint8),
        physical_mask=physical, physical_properties_valid_mask=physical.copy(),
        mass_support_mask=physical.copy(), hidden_mask=np.zeros(shape, bool),
        coverage_masks=dict(measured=physical.copy(), estimated=np.zeros(shape, bool), hidden=np.zeros(shape, bool)),
        intensive_validity_masks={name: physical.copy() for name in extensive.INTENSIVE},
        phase_labels=labels, saturation_free_gas=np.zeros(shape), saturation_valid_mask=np.zeros(shape, bool),
        native_pixel_masses_g=dict(total=values, aqueous=values * .5, gas=values * .25),
        channels=dict.fromkeys(extensive.CHANNELS), provenance={"synthetic_positive_sums": True})
    return _read(_product(_totals(args)))


def _layout_payload(source, layout):
    arrays = {}
    for name, original in source["arrays"].items():
        if layout == "F":
            value = np.asfortranarray(original)
        else:
            base = np.zeros(tuple(n * 2 for n in original.shape), dtype=original.dtype)
            value = base[::2, ::2]
            value[:] = original
        value.setflags(write=False)
        assert extensive._identity(value) == source["metadata"]["arrays"][name]
        arrays[name] = value
    return dict(source, arrays=arrays)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("target", [2, 5, 10])
def test_extensive_cells_are_float64_sums_of_the_stored_source_cells(precision, target):
    source = make_payload(precision)
    result = coarsening.coarsen_verified_extensive(source, target_resolution_mm=target)
    arrays, meta = result["arrays"], result["metadata"]
    assert result["format"] == coarsening.FORMAT
    assert meta["grid"]["format"] == coarsening.GRID_FORMAT
    assert meta["grid"]["shape"] == [20 // target, 20 // target]
    assert meta["source_precision"] == precision and meta["output_float_dtype"] == "float64"
    for name in extensive.EXTENSIVE + tuple("cell_mass_" + n + "_g" for n in extensive.MASS):
        expected = source["arrays"][name].reshape(20 // target, target, 20 // target, target).sum(
            axis=(1, 3), dtype=np.float64)
        np.testing.assert_array_equal(arrays[name], expected)
        assert arrays[name].dtype == np.float64
        assert not arrays[name].flags.writeable
    assert arrays["cell_mass_total_g"].sum() == 10.0
    assert arrays["cell_mass_aqueous_g"].sum() == 6.0
    assert arrays["canvas_pixel_count"].dtype == np.int64
    assert int(arrays["canvas_pixel_count"].sum()) == 6
    assert set(arrays) == set(source["arrays"])
    assert meta["native_to_1mm_reexport_gate"] == "NOT_RUN"
    assert not meta["simulator_format_selected"] and not result["adopted"]


def test_ratios_are_rederived_from_sums_with_the_original_validity_semantics():
    source = make_payload()
    result = coarsening.coarsen_verified_extensive(source, target_resolution_mm=5)
    arrays = result["arrays"]
    assert arrays["depth_m_mean"][0, 0] == (
        arrays["bulk_volume_m3"][0, 0] / arrays["physical_area_m2"][0, 0])
    assert arrays["porosity_mean"][0, 0] == (
        arrays["pore_volume_m3"][0, 0] / arrays["bulk_volume_m3"][0, 0])
    assert arrays["concentration_aqueous"][0, 0] == (
        arrays["cell_mass_aqueous_g"][0, 0] / (arrays["pore_volume_m3"][0, 0] * 1000))
    assert not np.isnan(arrays["saturation_free_gas"][0, 0])
    assert np.isnan(arrays["saturation_free_gas"][1:, :]).all()
    assert not arrays["saturation_free_gas_has_valid_input"][1:, :].any()
    np.testing.assert_array_equal(
        sum(arrays["coverage_" + n + "_pixel_count"] for n in extensive.COVERAGE),
        arrays["physical_pixel_count"])


def test_pending_channels_stay_absent_and_supplied_channel_keeps_its_support():
    source = make_payload(supplied_channel=True)
    result = coarsening.coarsen_verified_extensive(source, target_resolution_mm=2)
    assert result["metadata"]["channels"] == source["metadata"]["channels"]
    for name in ("M_maalt", "M_kant"):
        assert result["metadata"]["channels"][name] == {"status": extensive.PENDING}
        assert "cell_" + name + "_g" not in result["arrays"]
    assert result["arrays"]["cell_U_g"].sum() == 2.0
    assert result["arrays"]["channel_U_support_pixel_count"].sum() == 1
    assert result["metadata"]["conservation"]["cell_U_g"]["difference"] == 0.0


def test_missing_intensive_evidence_stays_undefined_despite_nonzero_mass_and_pore_volume():
    result = coarsening.coarsen_verified_extensive(make_payload(missing_intensive=True), target_resolution_mm=5)
    arrays = result["arrays"]
    assert arrays["cell_mass_aqueous_g"][0, 0] > 0
    assert arrays["pore_volume_m3"][0, 0] > 0
    assert np.isnan(arrays["concentration_aqueous"]).all()
    assert not arrays["concentration_aqueous_has_valid_input"].any()
    assert not arrays["concentration_aqueous_valid_native_pixel_count"].any()


def test_majority_comes_from_summed_facies_areas_and_lowest_id_wins_a_tie():
    result = coarsening.coarsen_verified_extensive(make_payload(equal_facies=True), target_resolution_mm=5)
    arrays = result["arrays"]
    assert arrays["facies_area_m2_1"][0, 0] == arrays["facies_area_m2_2"][0, 0] == 2.0
    assert arrays["facies_majority"][0, 0] == 1
    assert arrays["facies_majority"].dtype == np.uint8
    assert np.all(arrays["facies_majority"][1:, :] == 0)


def test_explicit_one_millimetre_identity_preserves_array_bytes_dtypes_and_ties():
    source = make_payload()
    result = coarsening.coarsen_verified_extensive(source, target_resolution_mm=1)
    for name, value in source["arrays"].items():
        assert result["arrays"][name].dtype == value.dtype
        assert result["arrays"][name].tobytes() == value.tobytes()
        assert result["metadata"]["arrays"][name] == source["metadata"]["arrays"][name]
    assert result["metadata"]["reduction_dtype"] == "none"
    assert result["metadata"]["native_to_1mm_reexport_gate"] == "NOT_RUN"


def test_source_arrays_metadata_and_write_flags_are_unchanged():
    source = make_payload(supplied_channel=True)
    metadata = deepcopy(source["metadata"])
    before = {name: (value.tobytes(), value.dtype, value.flags.writeable)
              for name, value in source["arrays"].items()}
    coarsening.coarsen_verified_extensive(source, target_resolution_mm=5)
    coarsening.coarsen_verified_extensive(source, target_resolution_mm=1)
    assert source["metadata"] == metadata
    assert before == {name: (value.tobytes(), value.dtype, value.flags.writeable)
                      for name, value in source["arrays"].items()}


@pytest.mark.parametrize("target", [1, 2, 5, 10])
@pytest.mark.parametrize("layout", ["F", "strided"])
def test_same_c_order_identity_has_same_outputs_and_conservation_for_every_memory_layout(target, layout):
    source = _large_and_small_payload()
    different = _layout_payload(source, layout)
    before = {name: (value.tobytes(), value.strides, value.flags.writeable)
              for name, value in different["arrays"].items()}
    expected = coarsening.coarsen_verified_extensive(source, target_resolution_mm=target)
    actual = coarsening.coarsen_verified_extensive(different, target_resolution_mm=target)
    assert actual["metadata"] == expected["metadata"]
    for name, value in expected["arrays"].items():
        np.testing.assert_array_equal(actual["arrays"][name], value)
    assert before == {name: (value.tobytes(), value.strides, value.flags.writeable)
                      for name, value in different["arrays"].items()}


def test_supplied_frame_and_producer_envelope_is_preserved_without_new_admission():
    source = make_payload()
    source.update(state="genuine_n15_n16_io",
        original_frame_identity=dict(run="run_a", case_id="frame_a", configuration={"sha256": "a" * 64}),
        producer_binding=dict(contract="saved-producer-contract", source_binding={"sha256": "b" * 64}),
        loaded_assets=[dict(path="saved/frame.npz", sha256="c" * 64, size_bytes=123)],
        physical_binding_verified=True, producer_identity_verified=True,
        production_permission_granted=False, U_layer_present=False)
    expected = deepcopy({key: value for key, value in source.items() if key not in {"arrays", "metadata"}})
    actual = coarsening.coarsen_verified_extensive(source, target_resolution_mm=5)
    assert actual["metadata"]["source_binding"] == expected
    assert not actual["production_permission_granted"]
    assert not actual["metadata"]["physical_binding_admitted"]
    assert not actual["metadata"]["producer_identity_admitted"]
    actual["metadata"]["source_binding"]["original_frame_identity"]["run"] = "another_run"
    assert source["original_frame_identity"] == expected["original_frame_identity"]
    basic = coarsening.coarsen_verified_extensive(make_payload(), target_resolution_mm=5)
    assert "original_frame_identity" not in basic["metadata"]["source_binding"]
    assert "producer_binding" not in basic["metadata"]["source_binding"]


@pytest.mark.parametrize("target", [None, True, 0, .5, 3, "5", float("nan"), float("inf")])
def test_unsupported_or_implicit_resolution_is_refused(target):
    with pytest.raises(ValueError, match="explicit 1, 2, 5 or 10"):
        coarsening.coarsen_verified_extensive(make_payload(), target_resolution_mm=target)


def test_legacy_quantized_fields_cannot_impersonate_complete_extensive_inputs():
    with pytest.raises(ValueError, match="verified complete extensive"):
        coarsening.coarsen_verified_extensive(
            dict(format="darsia-read-coarse-stored-frame-v1", arrays={}, metadata={}), target_resolution_mm=5)


@pytest.mark.parametrize("change", ["array", "missing", "identity", "resolution", "axes", "pending"])
def test_source_identity_schema_and_explicit_channel_state_are_rechecked(change):
    source = make_payload()
    if change == "array":
        changed = source["arrays"]["cell_mass_total_g"].copy()
        changed[0, 0] += 1
        source["arrays"]["cell_mass_total_g"] = changed
    elif change == "missing":
        del source["arrays"]["pore_volume_m3"]
    elif change == "identity":
        source["metadata"]["arrays"]["pore_volume_m3"]["sha256"] = "0" * 64
    elif change == "resolution":
        source["metadata"]["grid"]["resolution_mm"] = 2
    elif change == "axes":
        source["metadata"]["grid"]["axes"] = "row=y decreasing"
    else:
        source["metadata"]["channels"]["U"]["native_total_g"] = 0
    with pytest.raises(ValueError):
        coarsening.coarsen_verified_extensive(source, target_resolution_mm=5)


def test_partial_or_shifted_output_cells_are_refused_without_padding_or_clipping():
    source = make_payload()
    source["metadata"]["grid"]["extent_m"] = [.001, .021, .001, .021]
    with pytest.raises(ValueError, match="origin-zero output lattice"):
        coarsening.coarsen_verified_extensive(source, target_resolution_mm=5)


def test_count_sums_detect_overflow_before_casting_to_int64():
    maximum = np.iinfo(np.int64).max
    values = np.array([[maximum, 1], [0, 0]], np.int64)
    with pytest.raises(ValueError, match="exceeds int64"):
        coarsening._sum_counts(values, 2)
    values[0, 1] = 0
    assert coarsening._sum_counts(values, 2)[0, 0] == maximum


def test_nonfinite_extensive_sum_is_refused():
    with pytest.raises(ValueError, match="not finite"):
        coarsening._sum_cells(np.full((2, 2), np.finfo(np.float64).max), 2)
