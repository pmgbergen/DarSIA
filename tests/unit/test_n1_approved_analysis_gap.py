"""Bound approved exclusions are unknown remainders, never physical zero mass."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_arithmetic as arithmetic
from darsia.fluidflower.mass import n1_pipeline as pipeline
from darsia.fluidflower.mass import n1_refit as refit
from darsia.fluidflower.mass import n1_statistics as statistics
from .test_production_retained import evaluate, synthetic
from ..fixtures.tiny_saved_model_fixture import tiny_config


def _bound(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _approve(frame, folder, mask, phase=255):
    """Synthetic producer record only; no human or production approval is claimed."""
    frame.domain = frame.domain.copy()
    frame.domain[mask] = True
    frame.mass_eligible_mask = frame.mass_eligible_mask.copy()
    frame.mass_eligible_mask[mask] = False
    frame.phase_labels = frame.phase_labels.copy()
    frame.phase_labels[mask] = phase
    unknown = getattr(frame, "unknown_physical_mask", None)
    frame.unknown_physical_mask = mask.copy() if unknown is None else unknown | mask
    if "known_physical_domain" in vars(frame):
        frame.known_physical_domain = frame.known_physical_domain & ~mask
    frame.excluded_masks = dict(frame.excluded_masks or {}, approved_analysis_gap=mask.copy())
    identity = dict(dtype=mask.dtype.str, shape=list(mask.shape),
                    sha256=hashlib.sha256(mask.tobytes(order="C")).hexdigest())
    approval = _bound(folder / (frame.run_id + "-approval.json"), dict(synthetic_explicit_decision=True))
    provenance = dict(approval=approval, run_id=frame.run_id,
        native_mask=dict(artifact=approval, field="gaps", identity=deepcopy(identity)),
        mask_identity=identity, transport=dict(adopted_revision=approval,
            build_to_frame_native_h=np.eye(3).tolist(), native_origin_xy=[0, 0],
            shape_hw=list(mask.shape), source_to_frame_h=np.eye(3).tolist(),
            interpolation="original-nearest-neighbour-constant-zero-mask"),
        meaning="outside-analysis-unquantified", physical_frame_unchanged=True, labels_unchanged=True)
    frame.provenance = dict(frame.provenance, approved_analysis_gap=provenance)
    return provenance


def _summary(fixture):
    frame = fixture.response["frame"]
    shape = frame.domain.shape
    if not hasattr(frame, "facies"):
        frame.facies = np.ones(shape, np.uint8)
    rr, cc = np.indices(shape)
    geometry = statistics.EstimateGeometry(cc * .01, -rr * .01,
        np.full(shape, .008), np.full(shape, .41), np.full(shape, .231),
        np.ones(shape, np.uint8), np.ones(shape, bool), np.zeros(shape, bool),
        np.zeros(shape, bool), dict(synthetic=True))
    row = statistics.summarize_round3_response(fixture.response, geometry,
        own_colour=dict(aqueous_relative=np.full(shape, .2), available=np.ones(shape, bool), audit={}),
        photo_support=np.ones(shape, bool), no_outflow_evidence={}, light_epoch="synthetic",
        pixel_math=fixture.math, summary_settings=dict(baseline_gain=2., max_depth_m=.02,
            max_porosity=.5, max_column_distance_m=.020))
    row["selected_target_h"] = frame.elapsed_h
    return row


def _same_float(first, second):
    assert np.float64(first).tobytes() == np.float64(second).tobytes()


def test_absent_mask_keeps_historical_fields_integrals_and_export_bytes(synthetic):
    old = synthetic.old()
    result = evaluate(synthetic)
    assert "analysis_scope" not in result
    assert arithmetic.public_analysis_scope(result) is result
    for name in arithmetic.FIELDS:
        assert result["fields"][name].dtype == old["fields"][name].dtype
        assert result["fields"][name].tobytes() == old["fields"][name].tobytes()
    for name in ("total_mass_g", "aqueous_mass_g", "gas_mass_g", "dilution_factor", "affected_pore_volume_m3"):
        _same_float(result[name], old[name])
    products = arithmetic.main_products(result, synthetic.math)
    assert "analysis_scope" not in products
    for product, field in (("mass_total", "mass_density_g_l"),
                           ("mass_aqueous", "aqueous_mass_density_g_l"),
                           ("mass_free_gas", "gas_mass_density_g_l")):
        expected = np.where(old["fields"]["main_mass_eligible_mask"], old["fields"][field], 0.)
        assert products["products"][product].tobytes() == expected.tobytes()


@pytest.mark.parametrize("phase", [255, 1, 2])
def test_approved_pixels_stay_unknown_and_known_subtotals_keep_exact_bits(synthetic, tmp_path, phase):
    baseline = evaluate(synthetic)
    frame = synthetic.response["frame"]
    mask = np.zeros(frame.domain.shape, bool)
    mask[-1, -1] = True  # Historical fixture has no measured volume here.
    provenance = _approve(frame, tmp_path, mask, phase)
    synthetic.domains["full_rig"] = frame.domain
    synthetic.domains["box2"] |= mask
    before = {name: value.tobytes() for name, value in vars(frame).items() if isinstance(value, np.ndarray)}
    result = evaluate(synthetic)
    for name in ("total_mass_g", "aqueous_mass_g", "gas_mass_g", "dilution_factor", "affected_pore_volume_m3"):
        _same_float(result[name], baseline[name])
    scope = result["analysis_scope"]
    assert scope["approved_analysis_gap_pixels"] == 1 and scope["phase255_pixels"] == int(phase == 255)
    assert scope["unknown_mass_g"] is None and scope["area_cm2"] is None
    assert scope["provenance"] == provenance and scope["provenance"] is not provenance
    assert result["excluded_mass"]["approved_analysis_gap"]["total_mass_g"] is None
    assert result["fields"]["unquantified_phase_mask"][mask].all()
    assert np.isnan(result["fields"]["c_aq_g_l"][mask]).all()
    assert np.isnan(result["fields"]["s_g"][mask]).all()
    products = arithmetic.main_products(result, synthetic.math)
    for name in ("mass_total", "mass_aqueous", "mass_free_gas"):
        assert np.isnan(products["products"][name][mask]).all()
    assert np.array_equal(products["products"]["phase_labels"], frame.phase_labels)
    assert before == {name: value.tobytes() for name, value in vars(frame).items() if isinstance(value, np.ndarray)}
    public = arithmetic.public_analysis_scope(result)
    assert public["total_mass_g"] is None
    _same_float(public["quantified_total_mass_g"], baseline["total_mass_g"])
    assert all(row["total_mass_g"] is None for row in public["domains"].values())
    assert result["total_mass_g"] is not None  # Internal carrier is not overwritten.


@pytest.mark.parametrize("phase", [255, 1, 2])
def test_no_approved_pixels_enter_capacity_colour_or_column_estimates(synthetic, tmp_path, phase):
    before = _summary(synthetic)
    frame = synthetic.response["frame"]
    mask = np.zeros(frame.domain.shape, bool)
    mask[-1, -1] = True
    _approve(frame, tmp_path, mask, phase)
    row = _summary(synthetic)
    part = row["excluded_parts"]["approved_analysis_gap"]
    assert part["pixel_count"] == 1 and part["phase255_pixels"] == int(phase == 255)
    assert part["capacity_g"] is None and part["unknown_mass_g"] is None and part["area_cm2"] is None
    for name in ("own_estimate_pixels", "column_estimate_pixels", "fixed_gas_estimate_pixels", "neither_estimate_pixels"):
        assert row[name] == before[name]
    assert row["paired_estimates"] == before["paired_estimates"]
    partition = statistics.physical_partition(synthetic.response, max_depth_m=.02, max_porosity=.5)
    assert not partition["missing"][mask].any()
    assert partition["missing_capacity_upper_g"] is None
    assert partition["parts"]["approved_analysis_gap"]["capacity_upper_g"] is None


def test_canonical_compiled_and_exact_predictions_preserve_unknown_remainder(synthetic, tmp_path):
    frame = synthetic.response["frame"]
    mask = np.zeros(frame.domain.shape, bool)
    mask[-1, -1] = True
    _approve(frame, tmp_path, mask)
    synthetic.domains["full_rig"] = frame.domain
    synthetic.domains["box2"] |= mask
    row = _summary(synthetic)
    canonical = statistics.evaluate_summary(row, synthetic.parameters, synthetic.anchor)
    compiled = refit.compile_frame(row, synthetic.parameters["gain"])
    # Retained compiled scope must survive even if a caller drops the row key.
    row.pop("analysis_scope")
    fast = refit.evaluate_compiled(compiled, synthetic.parameters["beta"], synthetic.anchor)
    row["analysis_scope"] = deepcopy(compiled["analysis_scope"])
    exact = statistics.exact_prediction(row, evaluate(synthetic), synthetic.parameters, synthetic.anchor)
    for value in (canonical, fast, exact):
        assert value["analysis_scope"]["unknown_mass_g"] is None
        assert value["W_g"] is None and value["W_over_d"] is None
        assert value["U_rest_g"] is None and value["signed_interval_residual"] is None and value["delta"] is None
        assert "quantified_signed_interval_residual" not in value and "quantified_W_over_d" not in value
        assert np.isfinite(value["quantified_total_mass_g"])
    assert fast["quantified_total_mass_g"] == pytest.approx(canonical["quantified_total_mass_g"], rel=1e-12)
    prediction = refit.predict_run([row], dict(parameters=synthetic.parameters,
        anchors_by_run={frame.run_id: synthetic.anchor}), frame.run_id)[0]
    assert prediction["total_mass_g"] is None and prediction["quantified_total_mass_g"] is not None
    assert prediction["analysis_scope"]["whole_rig_mass_complete"] is False


@pytest.mark.parametrize("damage", ["mask", "provenance", "eligible", "known", "unknown"])
def test_changed_or_quantified_approval_input_is_not_repaired(synthetic, tmp_path, damage):
    frame = synthetic.response["frame"]
    mask = np.zeros(frame.domain.shape, bool)
    mask[-1, -1] = True
    _approve(frame, tmp_path, mask)
    if damage == "mask":
        frame.excluded_masks["approved_analysis_gap"][0, 3] = True
    elif damage == "provenance":
        frame.provenance["approved_analysis_gap"]["meaning"] = "free_water"
    elif damage == "eligible":
        frame.mass_eligible_mask[mask] = True
    elif damage == "unknown":
        frame.unknown_physical_mask[mask] = False
    else:
        frame.known_physical_domain[mask] = True
    with pytest.raises(ValueError, match="Approved analysis gaps"):
        evaluate(synthetic)


def test_actual_n1_publication_has_null_totals_and_nan_gap_maps(tiny_config, tmp_path, monkeypatch):
    from functools import partial
    config, bound = tiny_config
    original = pipeline.N1Pipeline.load_frame
    observed = {}

    def with_gap(context, spec):
        frame, *rest = original(context, spec)
        mask = np.zeros(frame.domain.shape, bool)
        point = np.argwhere(frame.mass_eligible_mask)[0]
        mask[tuple(point)] = True
        _approve(frame, tmp_path, mask)
        observed[spec["case_id"]] = mask
        return frame, *rest

    monkeypatch.setattr(pipeline.N1Pipeline, "load_frame", with_gap)
    output = tmp_path / "new-output"
    output.mkdir()
    outcome = pipeline.run_n1(dict(run=config["run"], configuration=bound,
        commit="dd403effe5bfb49bfaadc0bf185e319f79a66a76"), output, "cpu", "float32",
        _runner=partial(pipeline._run_n1, geometry_fields=dict(
            source_to_native="H_ac28_to_frame_native",
            original_domain_to_native="H_ac28_to_frame_native",
            carried_homography_provenance="original_H_ac28_to_native")))
    for publication in outcome["records"]:
        folder = Path(publication["nas_directory"])
        record = json.loads((folder / "result.json").read_bytes())
        assert record["total_mass_g"] is None and record["scalar_result"]["total_mass_g"] is None
        assert np.isfinite(record["quantified_total_mass_g"])
        assert record["prediction"]["interval_upper_g"] is None and record["W_g"] is None
        assert record["excluded_mass"]["approved_analysis_gap"]["total_mass_g"] is None
        assert record["analysis_scope"]["area_cm2"] is None
        with gzip.open(folder / "native_statistics.json.gz", "rt") as stream:
            assert json.load(stream)["analysis_scope"]["unknown_mass_g"] is None
        with np.load(folder / "products.npz", allow_pickle=False) as data:
            mask = observed[record["case_id"]]
            assert np.array_equal(data["approved_analysis_gap_mask"], mask)
            assert (data["phase_labels"][mask] == 255).all()
            for name in ("mass_total", "mass_aqueous", "mass_free_gas"):
                assert np.isnan(data[name][mask]).all()
