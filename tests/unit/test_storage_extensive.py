"""Small explicit output data only; no Main/N15/N16 producer or physical admission."""
import copy
import io
import zipfile

import numpy as np
import pytest

from darsia.fluidflower import storage_extensive as extensive
from darsia.fluidflower import storage_resolution as original


def _totals(args):
    p = args["physical_mask"]
    a, d, v = args["grid"].native_pixel_area_m2, args["depth_m"], args["pixel_pore_volume_m3"]
    gas = p & (args["phase_labels"] == 2)
    args["native_mass_totals_g"] = {
        name: float(value.sum(dtype=np.float64)) for name, value in args["native_pixel_masses_g"].items()}
    args["native_extensive_totals"] = dict(
        physical_area_m2=float(a[p].sum()), bulk_volume_m3=float((a[p] * d[p]).sum()),
        pore_volume_m3=float(v[p].sum()),
        gas_pore_volume_m3=float((v[gas] * args["saturation_free_gas"][gas]).sum()))
    return args


def _inputs(precision="float32"):
    x = np.array([[.0002, .0008, .0022], [.0003, .0012, .0028]])
    y = np.array([[.0002, .0002, .0002], [.0012, .0012, .0012]])
    area = np.array([[1., 3., 2.], [2., 4., 1.]]) * 1e-6
    grid = original.make_grid(x, y, area, include=np.ones(x.shape, bool),
                              extent_m=[0., .004, 0., .003], storage_resolution_mm=1,
                              precision=precision)
    depth = np.array([[1., 2., 1.], [2., 3., 4.]])
    phi = np.array([[.2, .5, .4], [.8, .1, .2]])
    physical = np.array([[True, True, True], [True, True, False]])
    measured = np.array([[True, True, True], [False, False, False]])
    estimated = np.array([[False, False, False], [True, False, False]])
    hidden_coverage = np.array([[False, False, False], [False, True, False]])
    labels = np.array([[0, 2, 1], [1, 2, 255]], np.uint8)
    saturation = np.full(x.shape, np.nan)
    saturation[0, 1], saturation[1, 1] = .5, .25
    values = dict(total=np.array([[2., 5., 3.], [0., 0., 0.]]),
                  aqueous=np.array([[1., 3., 2.], [0., 0., 0.]]),
                  gas=np.array([[.5, 1., .1], [0., 0., 0.]]))
    return _totals(dict(grid=grid, depth_m=depth, porosity=phi,
                       pixel_pore_volume_m3=area * depth * phi,
                       facies=np.array([[2, 1, 2], [3, 1, 255]], np.uint8),
                       physical_mask=physical, physical_properties_valid_mask=physical.copy(),
                       mass_support_mask=measured.copy(), hidden_mask=~physical,
                       coverage_masks=dict(measured=measured, estimated=estimated, hidden=hidden_coverage),
                       intensive_validity_masks={name: physical.copy() for name in extensive.INTENSIVE},
                       phase_labels=labels, saturation_free_gas=saturation,
                       saturation_valid_mask=labels == 2, native_pixel_masses_g=values,
                       channels=dict.fromkeys(extensive.CHANNELS),
                       provenance=dict(test_data_only=True, code_binding="synthetic fixture",
                                       calibration_binding=None, camera_clock=None)))


def _product(args=None):
    args = _inputs() if args is None else args
    return extensive.aggregate_extensive(**args)


def _bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _read(product, *, arrays=None, metadata=None, expected=None):
    base = product["metadata"]
    kwargs = dict(expected_input_identities=base["input_identities"],
                  native_mass_totals_g=base["native_mass_totals_g"],
                  native_extensive_totals=base["native_extensive_totals"],
                  expected_channels=base["channels"], expected_provenance=base["provenance"],
                  expected_grid_metadata=base["grid"])
    if expected:
        kwargs.update(expected)
    return extensive.read_extensive_archive(_bytes(product["arrays"] if arrays is None else arrays),
                                            base if metadata is None else metadata, **kwargs)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_extensive_sums_and_weighted_means_are_independent(precision):
    args = _inputs(precision)
    product = _product(args)
    a, m = product["arrays"], product["metadata"]
    assert m["precision"] == precision
    assert m["grid"]["storage_precision"] == precision
    assert m["reduction_dtype"] == "float64"
    assert all(v.dtype == np.dtype(precision) for k, v in a.items()
               if not k.endswith("_pixel_count") and k not in {"facies_majority", "cell_has_pore_volume"}
               and not k.endswith("_has_valid_input"))
    # Hand calculation for the first cell: areas1+3, bulk1+6, PV.2+3,
    # gasPV.5*3. Porosity is PV/V, not the area mean(.425).
    assert a["physical_area_m2"][0, 0] == pytest.approx(4e-6)
    assert a["bulk_volume_m3"][0, 0] == pytest.approx(7e-6)
    assert a["pore_volume_m3"][0, 0] == pytest.approx(3.2e-6)
    assert a["gas_pore_volume_m3"][0, 0] == pytest.approx(1.5e-6)
    assert a["depth_m_mean"][0, 0] == pytest.approx(1.75)
    assert a["porosity_mean"][0, 0] == pytest.approx(3.2 / 7)
    assert a["saturation_free_gas"][0, 0] == pytest.approx(1.5 / 3.2)
    assert a["concentration_aqueous"][0, 0] == pytest.approx(4. / .0032)
    assert a["cell_mass_total_g"].sum() == 10.
    assert a["cell_mass_aqueous_g"].sum() == 6.
    assert a["cell_mass_gas_g"].sum() == pytest.approx(1.6)
    assert not np.array_equal(a["cell_mass_total_g"], a["cell_mass_aqueous_g"] + a["cell_mass_gas_g"])
    for name in extensive.MASS:
        check = m["native_checks"]["cell_mass_" + name + "_g"]
        assert check["native_total"] == args["native_mass_totals_g"][name]
        assert check["absolute_difference"] <= 1e-6 * abs(check["native_total"])
    assert a["facies_majority"][0, 0] == 1
    assert a["facies_area_fraction_1"][0, 0] == .75
    assert a["canvas_pixel_count"].sum() == 6
    assert a["physical_pixel_count"].sum() == 5
    assert a["hidden_pixel_count"].sum() == 1


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_reader_roundtrip_is_typed_immutable_and_not_a_producer_admission(precision):
    p = _product(_inputs(precision))
    got = _read(p)
    assert not got["physical_binding_admitted"]
    assert not got["producer_identity_admitted"]
    for key, value in p["arrays"].items():
        assert np.array_equal(got["arrays"][key], value, equal_nan=True)
        assert got["arrays"][key].dtype == value.dtype
        assert not got["arrays"][key].flags.writeable


def test_supplied_original_pore_volume_is_not_rebuilt_from_cell_mean_porosity():
    args = _inputs()
    # A separately supplied original Geometry weight must be the reducer input.
    # The reference here is explicit synthetic data, not a new capacity recipe.
    args["pixel_pore_volume_m3"][0, 0] *= 1.2
    p = _product(_totals(args))
    assert p["arrays"]["pore_volume_m3"][0, 0] == pytest.approx(3.24e-6)
    assert p["metadata"]["input_identities"]["pixel_pore_volume_m3"] == extensive._identity(args["pixel_pore_volume_m3"])


@pytest.mark.parametrize("kind", ["no_physical_pixels", "physical_zero_area"])
def test_empty_or_zero_pv_cells_have_zero_extensive_sums_and_nan_intensive_values(kind):
    args = _inputs()
    if kind == "no_physical_pixels":
        args["physical_mask"][:] = False
        args["physical_properties_valid_mask"][:] = False
        for mask in args["intensive_validity_masks"].values():
            mask[:] = False
        for mask in args["coverage_masks"].values():
            mask[:] = False
    else:
        # A native shape can have geometric pixels with zero original area.
        args["grid"].native_pixel_area_m2[:] = 0
        args["pixel_pore_volume_m3"][:] = 0
    args["mass_support_mask"][:] = False
    for values in args["native_pixel_masses_g"].values():
        values[:] = 0
    p = _product(_totals(args))
    got = _read(p)["arrays"]
    for name in extensive.EXTENSIVE:
        assert np.all(got[name] == 0)
    for name in ["depth_m_mean", "porosity_mean", "concentration_aqueous", "saturation_free_gas"]:
        assert np.isnan(got[name]).all()
    assert not got["cell_has_pore_volume"].any()


@pytest.mark.parametrize("kind", ["overlap", "omission", "outside", "nonboolean"])
def test_coverage_is_explicit_and_never_guessed(kind):
    a = _inputs()
    if kind == "overlap":
        a["coverage_masks"]["estimated"][0, 0] = True
    elif kind == "omission":
        a["coverage_masks"]["measured"][0, 0] = False
    elif kind == "outside":
        a["coverage_masks"]["hidden"][1, 2] = True
    else:
        a["coverage_masks"]["hidden"] = a["coverage_masks"]["hidden"].astype(np.uint8)
    with pytest.raises(ValueError):
        _product(a)


@pytest.mark.parametrize("kind", ["property_support", "gas_support", "gas_nan", "pore_volume_nan"])
def test_missing_required_physical_or_gas_evidence_is_not_zero_imputed(kind):
    a = _inputs()
    if kind == "property_support":
        a["physical_properties_valid_mask"][0, 0] = False
    elif kind == "gas_support":
        a["saturation_valid_mask"][0, 1] = False
    elif kind == "gas_nan":
        a["saturation_free_gas"][0, 1] = np.nan
    else:
        a["pixel_pore_volume_m3"][0, 0] = np.nan
    with pytest.raises(ValueError):
        _product(a)


def test_native_mass_is_not_discarded_outside_explicit_main_support():
    a = _inputs()
    a["native_pixel_masses_g"]["total"][1, 0] = 1
    with pytest.raises(ValueError, match="explicit support"):
        _product(_totals(a))


@pytest.mark.parametrize("reference", [0., 9.])
def test_exact_existing_relative_gate_uses_independent_native_total_including_zero(reference):
    a = _inputs()
    a["native_mass_totals_g"]["total"] = reference
    with pytest.raises(ValueError, match="1e-6"):
        _product(a)


@pytest.mark.parametrize("actual,expected,passes", [(1000001., 1000000., True),
    (1000001.0001, 1000000., False), (0., 0., True), (5e-324, 0., False)])
def test_original_one_e_minus_six_gate_has_no_default_absolute_floor(actual, expected, passes):
    cells = np.array([[actual]], np.float64)
    if passes:
        check = extensive._check(cells, expected, "independent fixture scalar")
        assert check["absolute_difference"] <= 1e-6 * abs(expected)
    else:
        with pytest.raises(ValueError, match="1e-6"):
            extensive._check(cells, expected, "independent fixture scalar")


def test_pending_channels_are_absent_not_all_zero_products():
    p = _product()
    for name in extensive.CHANNELS:
        assert p["metadata"]["channels"][name] == {"status": extensive.PENDING}
        assert "cell_" + name + "_g" not in p["arrays"]
    _read(p)


def test_supplied_channel_arrays_are_only_summed_and_hidden_does_not_choose_physical_pv():
    args = _inputs()
    mask = args["hidden_mask"]
    u = np.where(mask, 2., 0.)
    args["channels"]["U"] = dict(pixel_values_g=u, support_mask=mask,
                                  native_total_g=2., producer_binding={"synthetic_explicit_channel": True})
    p = _product(args)
    got = _read(p)
    assert got["arrays"]["cell_U_g"].sum() == 2.
    assert got["arrays"]["pore_volume_m3"].sum() == pytest.approx(args["native_extensive_totals"]["pore_volume_m3"])
    assert p["metadata"]["channels"]["M_kant"]["status"] == extensive.PENDING
    assert not got["producer_identity_admitted"]


@pytest.mark.parametrize("kind", ["missing_name", "unbound", "outside_support"])
def test_channels_require_explicit_names_arrays_support_and_producer_binding(kind):
    a = _inputs()
    if kind == "missing_name":
        del a["channels"]["U"]
    else:
        a["channels"]["U"] = dict(pixel_values_g=np.ones((2, 3)),
                                  support_mask=np.zeros((2, 3), bool), native_total_g=6.,
                                  producer_binding={} if kind == "unbound" else {"explicit_test": True})
    with pytest.raises(ValueError):
        _product(a)


def test_exact_area_tie_keeps_original_lowest_facies_id_rule():
    a = _inputs()
    a["grid"].native_pixel_area_m2[0, 0] = 3e-6
    a["pixel_pore_volume_m3"] = a["grid"].native_pixel_area_m2 * a["depth_m"] * a["porosity"]
    p = _product(_totals(a))
    assert p["arrays"]["facies_majority"][0, 0] == 1
    _read(p)


def test_original_label_area_and_before_after_count_contracts_stay_separate():
    a = _inputs()
    before = np.array([[0., 1., np.nan], [np.nan, np.nan, np.nan]])
    old = original.aggregate_frame(a["grid"], concentration_aqueous=np.ones((2, 3)),
        saturation_free_gas=np.zeros((2, 3)), phase_labels=a["phase_labels"], masks={},
        probabilities=dict(gas_probability_before_rules=before,
                           gas_probability_after_rules=a["phase_labels"] == 2),
        native_pixel_masses_g=a["native_pixel_masses_g"], native_mass_totals_g=a["native_mass_totals_g"])
    assert old["arrays"]["phase_gas_area_fraction"][0, 0] == .75
    assert old["arrays"]["gas_probability_before_rules"][0, 0] == .5
    assert old["arrays"]["gas_probability_after_rules"][0, 0] == .5
    assert _product(a)["arrays"]["saturation_free_gas"][0, 0] == pytest.approx(.46875)


@pytest.mark.parametrize("kind", ["unit", "weighting", "provenance", "grid", "scalar_type", "input_shape_type"])
def test_reader_rejects_hash_valid_metadata_contract_tampering(kind):
    p = _product()
    m = copy.deepcopy(p["metadata"])
    if kind == "unit":
        m["fields"]["pore_volume_m3"]["units"] = "L"
    elif kind == "weighting":
        m["fields"]["porosity_mean"]["weighting"] = "naive pixel mean"
    elif kind == "provenance":
        m["provenance"]["camera_clock"] = "invented"
    elif kind == "grid":
        m["grid"]["extent_m"] = [.001, .005, 0., .003]
    elif kind == "scalar_type":
        m["native_mass_totals_g"]["total"] = 10
    else:
        m["input_identities"]["physical_mask"]["shape"][0] = True
    with pytest.raises(ValueError):
        _read(p, metadata=m)


@pytest.mark.parametrize("kind", ["coverage_count", "property_count", "ratio", "zero_support", "negative_count", "overflow_count", "dtype", "facies_dtype"])
def test_reader_rejects_rehashed_array_closure_tampering(kind):
    p = _product()
    a, m = copy.deepcopy(p["arrays"]), copy.deepcopy(p["metadata"])
    key = {"coverage_count": "coverage_measured_pixel_count", "property_count": "physical_properties_valid_pixel_count",
           "ratio": "concentration_aqueous", "zero_support": "mass_support_pixel_count",
           "negative_count": "canvas_pixel_count", "overflow_count": "canvas_pixel_count",
           "dtype": "physical_area_m2", "facies_dtype": "facies_majority"}[kind]
    if kind == "dtype":
        a[key] = a[key].astype(np.float64)
    elif kind == "facies_dtype":
        a[key] = a[key].astype(np.int64)
    elif kind == "zero_support":
        a[key][0, 0] = 0
    elif kind == "negative_count":
        a[key][0, 0] = -1
    elif kind == "overflow_count":
        a[key][0, 0] = np.iinfo(np.int64).max
    else:
        a[key][0, 0] += 1
    m["arrays"][key] = extensive._identity(a[key])
    with pytest.raises(ValueError):
        _read(p, arrays=a, metadata=m)


@pytest.mark.parametrize("kind", ["extra", "duplicate"])
def test_reader_rejects_extra_or_duplicate_npz_members(kind):
    p = _product()
    raw = _bytes(p["arrays"])
    stream = io.BytesIO(raw)
    with zipfile.ZipFile(stream, "a") as archive:
        if kind == "extra":
            archive.writestr("surprise.npy", b"not admitted")
        else:
            with zipfile.ZipFile(io.BytesIO(raw)) as source:
                first = source.namelist()[0]
                with pytest.warns(UserWarning):
                    archive.writestr(first, source.read(first))
    m = p["metadata"]
    with pytest.raises(ValueError, match="ZIP members"):
        extensive.read_extensive_archive(stream.getvalue(), m,
            expected_input_identities=m["input_identities"], native_mass_totals_g=m["native_mass_totals_g"],
            native_extensive_totals=m["native_extensive_totals"], expected_channels=m["channels"],
            expected_provenance=m["provenance"], expected_grid_metadata=m["grid"])


def test_api_leaves_every_supplied_native_array_unchanged():
    a = _inputs()
    arrays = [a[k] for k in ["depth_m", "porosity", "pixel_pore_volume_m3", "facies", "physical_mask",
              "physical_properties_valid_mask", "mass_support_mask", "hidden_mask", "phase_labels",
              "saturation_free_gas", "saturation_valid_mask"]]
    arrays += list(a["coverage_masks"].values()) + list(a["native_pixel_masses_g"].values())
    arrays += list(a["intensive_validity_masks"].values())
    before = [x.copy() for x in arrays]
    _product(a)
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(arrays, before))


@pytest.mark.parametrize("with_explicit_u", [False, True])
def test_positive_pv_fully_hidden_cells_keep_volumes_but_no_invented_means(with_explicit_u):
    a = _inputs()
    physical = a["physical_mask"]
    a["hidden_mask"] = physical.copy()
    a["coverage_masks"] = dict(measured=np.zeros(physical.shape, bool),
                                estimated=np.zeros(physical.shape, bool), hidden=physical.copy())
    a["phase_labels"][:] = 255
    a["saturation_valid_mask"][:] = False
    a["saturation_free_gas"][:] = np.nan
    a["mass_support_mask"][:] = False
    for values in a["native_pixel_masses_g"].values():
        values[:] = 0
    for mask in a["intensive_validity_masks"].values():
        mask[:] = False
    if with_explicit_u:
        a["channels"]["U"] = dict(pixel_values_g=physical.astype(np.float64), support_mask=physical.copy(),
            native_total_g=float(physical.sum()), producer_binding={"synthetic_caller_supplied_channel": True})
    product = _product(_totals(a))
    values = _read(product)["arrays"]
    positive = values["pore_volume_m3"] > 0
    assert positive.any()
    assert np.all(values["physical_area_m2"][positive] > 0)
    assert np.all(values["bulk_volume_m3"][positive] > 0)
    assert np.all(values["coverage_hidden_pore_volume_fraction"][positive] == 1)
    for name in extensive.MASS:
        assert np.all(values["cell_mass_" + name + "_g"] == 0)
    for name in extensive.INTENSIVE:
        assert np.isnan(values[name]).all()
        assert np.all(values[name + "_valid_native_pixel_count"] == 0)
        assert not values[name + "_has_valid_input"].any()
    if with_explicit_u:
        assert values["cell_U_g"].sum() == float(physical.sum())
    else:
        assert "cell_U_g" not in values


def test_caller_validity_never_renormalizes_full_physical_denominators():
    a = _inputs()
    a["intensive_validity_masks"]["depth_m_mean"][0, 1] = False
    a["intensive_validity_masks"]["concentration_aqueous"][:] = False
    values = _read(_product(a))["arrays"]
    assert values["depth_m_mean_valid_native_pixel_count"][0, 0] == 1
    assert values["depth_m_mean_has_valid_input"][0, 0]
    assert values["depth_m_mean"][0, 0] == pytest.approx(7. / 4.)
    assert np.isnan(values["concentration_aqueous"]).all()
    assert values["pore_volume_m3"][0, 0] == pytest.approx(3.2e-6)
    assert values["cell_mass_aqueous_g"][0, 0] == 4.


@pytest.mark.parametrize("kind", ["missing_name", "nonboolean", "outside_physical"])
def test_every_intensive_validity_mask_is_explicit_and_typed(kind):
    a = _inputs()
    name = "concentration_aqueous"
    if kind == "missing_name":
        del a["intensive_validity_masks"][name]
    elif kind == "nonboolean":
        a["intensive_validity_masks"][name] = a["physical_mask"].astype(np.uint8)
    else:
        a["intensive_validity_masks"][name][1, 2] = True
    with pytest.raises(ValueError):
        _product(a)


@pytest.mark.parametrize("kind", ["count", "flag", "ratio"])
def test_reader_rechecks_explicit_validity_even_after_array_hashes_are_updated(kind):
    p = _product()
    arrays, metadata = copy.deepcopy(p["arrays"]), copy.deepcopy(p["metadata"])
    name = "concentration_aqueous"
    key = name + {"count": "_valid_native_pixel_count", "flag": "_has_valid_input", "ratio": ""}[kind]
    arrays[key][0, 0] = 0 if kind == "count" else (False if kind == "flag" else np.nan)
    metadata["arrays"][key] = extensive._identity(arrays[key])
    with pytest.raises(ValueError, match="validity"):
        _read(p, arrays=arrays, metadata=metadata)
