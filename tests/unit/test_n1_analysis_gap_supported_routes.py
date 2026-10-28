"""Real supported N1 publication/storage/refit paths retain an unknown remainder.

Only the explicit synthetic FrameData producer seam supplies a bound test mask.
All numerical producers, immutable writers, stored readers and optimizers are real.
"""
from copy import deepcopy
import ast
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_arithmetic as arithmetic
from darsia.fluidflower.mass import n1_pipeline as pipeline
from darsia.fluidflower.mass import n1_refit as refit
from darsia.fluidflower.mass import pixel_pipeline
from darsia.fluidflower import storage_outputs as output
from darsia.fluidflower import storage_saved_products as native_saved
from darsia.fluidflower import storage_n1_saved_products as n1_saved
from darsia.fluidflower import storage_extensive_io as extensive_io
from darsia.fluidflower import storage_saved_export as saved_export
from darsia.fluidflower import storage_saved_tables as tables
from darsia.fluidflower.pipeline import lean_capacity, n1_capacity

from .test_n1_approved_analysis_gap import _approve, synthetic
from .test_storage_n1_saved_products import (
    tiny_config, phase1_data, _inputs, _genuine_n16_config)
from .test_storage_saved_products import _fixture, _reseal, _resolver
from ..fluidflower.test_n1_pixel_domain_integration import frozen_run
from ..fluidflower.test_n1_refit import make_plan, make_rows, make_row, TRUE_GAIN, TRUE_BETA


def _execute(config, folder, precision="float32"):
    folder.mkdir()
    path = folder / "configuration.json"
    path.write_text(json.dumps(config), encoding="utf8")
    return pipeline.run_n1(dict(run=config["run"], configuration=pipeline.asset(path),
        commit="dd403effe5bfb49bfaadc0bf185e319f79a66a76"), folder / "output", "cpu", precision)


def _inject(monkeypatch, folder, point, *, damage=False):
    original = pipeline.N1Pipeline.load_frame
    observed = {}
    def approved(context, spec):
        frame, *rest = original(context, spec)
        mask = np.zeros(frame.domain.shape, bool)
        mask[point] = True
        labels = frame.phase_labels.copy()
        _approve(frame, folder, mask, phase=int(labels[point]))
        assert np.array_equal(frame.phase_labels, labels)
        if damage:
            frame.provenance["approved_analysis_gap"]["mask_identity"]["sha256"] = "0" * 64
        observed[spec["case_id"]] = mask
        return frame, *rest
    monkeypatch.setattr(pipeline.N1Pipeline, "load_frame", approved)
    return observed


def _record(record):
    return json.loads((Path(record["nas_directory"]) / "result.json").read_bytes())


def _qualified_evaluate():
    """Only the exact preimage's evaluate codeobject; no old runner import."""
    source = subprocess.check_output(["git", "show",
        "dd403effe5bfb49bfaadc0bf185e319f79a66a76:src/darsia/fluidflower/mass/pixel_pipeline.py"],
        cwd=Path(__file__).resolve().parents[2]).decode("utf8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    namespace = dict(vars(pixel_pipeline))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        "qualified dd403 pixel evaluate", "exec"), namespace)
    return namespace["evaluate"]


def _assert_unknown(record):
    assert record["analysis_scope"]["approved_analysis_gap_pixels"] == 1
    assert record["analysis_scope"]["unknown_mass_g"] is None
    assert record["analysis_scope"]["area_cm2"] is None
    assert record["analysis_scope"]["whole_rig_mass_complete"] is False
    assert record["total_mass_g"] is None and np.isfinite(record["quantified_total_mass_g"])
    assert record["prediction"]["interval_upper_g"] is None
    assert record["prediction"]["interval_lower_g"] is None
    assert record["prediction"]["signed_interval_residual"] is None
    assert record["W_g"] is None and record["W_over_d"] is None


@pytest.mark.parametrize("damage", [False, True])
def test_real_n15_dispatch_validates_approval_and_publishes_scoped_unknowns(frozen_run, tmp_path, monkeypatch, damage):
    config, _ = frozen_run
    baseline = _execute(config, tmp_path / "ordinary")
    observed = _inject(monkeypatch, tmp_path, (2, 0), damage=damage)
    if damage:
        with pytest.raises(ValueError, match="Approved analysis gaps"):
            _execute(config, tmp_path / "rejected")
        assert not list((tmp_path / "rejected").rglob("commit.json"))
        return
    actual = _execute(config, tmp_path / "approved")
    for current, prior in zip(actual["records"], baseline["records"], strict=True):
        record = _record(current)
        _assert_unknown(record)
        assert record["U_g"] is None
        assert record["analysis_scope"]["phase255_pixels"] == 1
        assert record["phase_unknown_no_capacity_area_cm2"] is None
        with np.load(Path(current["nas_directory"]) / "products.npz", allow_pickle=False) as new, \
                np.load(Path(prior["nas_directory"]) / "products.npz", allow_pickle=False) as old:
            mask = observed[record["case_id"]]
            for name in ("concentration_aqueous", "saturation_free_gas", "mass_total", "mass_aqueous", "mass_free_gas"):
                assert np.isnan(new[name][mask]).all()
                assert new[name][~mask].tobytes() == old[name][~mask].tobytes()
            assert new["unknown_without_capacity_mask"][mask].all()
            assert not new["mass_contribution_mask"][mask].any()
            assert new["phase_labels"].tobytes() == old["phase_labels"].tobytes()
        assert np.float64(record["quantified_total_mass_g"]).tobytes() == np.float64(_record(prior)["total_mass_g"]).tobytes()
    with Path(actual["table"]["path"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(json.loads(row["analysis_scope_json"])["unknown_mass_g"] is None for row in rows)
    assert all(row["U_g"] == row["unknown_mass_g"] == "" for row in rows)


@pytest.mark.parametrize("adjacent", [False, True])
def test_real_n16_storage_and_extensive_readers_keep_partial_scope(tiny_config, phase1_data, tmp_path, monkeypatch, adjacent):
    config = _genuine_n16_config(tmp_path, tiny_config, phase1_data, adjacent=adjacent)
    baseline = _execute(config, tmp_path / "ordinary")
    observed = _inject(monkeypatch, tmp_path, (1, 1))  # Existing hidden phase255; labels stay exact.
    actual = _execute(config, tmp_path / "approved")
    manifest = json.loads(Path(actual["manifest"]["path"]).read_bytes())
    for current, prior in zip(actual["records"], baseline["records"], strict=True):
        record = _record(current)
        _assert_unknown(record)
        assert record["U_g"] is None and record["quantified_U_g"] == 0.
        assert record["domains"]["full_rig"]["analysis_scope"]["approved_analysis_gap_pixels"] == 1
        assert record["pixel_domain"]["analysis_scope"]["unknown_mass_g"] is None
        stored = n1_saved.read_n1_stored_frame(current, manifest=manifest,
            manifest_asset=actual["manifest"], resolve_asset=lambda b: Path(b["path"]))
        payload = extensive_io.read_committed_payload(current, producer_kind="n1-mass-only",
            manifest=manifest, manifest_asset=actual["manifest"], resolve_asset=lambda b: Path(b["path"]))
        for read in (stored, payload):
            assert read["mass_totals_g"] == dict.fromkeys(output.TOTALS)
            assert read["unknown_mass_g"] is None
            assert read["analysis_scope"]["whole_rig_mass_complete"] is False
            assert read["quantified_mass_totals_g"]["total"] == pytest.approx(
                record["quantified_total_mass_g"], rel=1e-6, abs=0.)
        assert np.any(payload["approved_analysis_gap_area_fraction"] > 0)
        assert int(payload["arrays"]["physical_pixel_count"].sum()) == 8
        assert int(payload["arrays"]["hidden_pixel_count"].sum()) == 0
        if adjacent:
            assert payload["metadata"]["channels"]["U"]["status"] == extensive_io.extensive.PENDING
        else:
            assert payload["metadata"]["channels"]["U"]["status"] == extensive_io.extensive.KNOWN_ZERO
        assert "cell_U_g" not in payload["arrays"]
        with np.load(Path(current["nas_directory"]) / "products.npz", allow_pickle=False) as new, \
                np.load(Path(prior["nas_directory"]) / "products.npz", allow_pickle=False) as old:
            mask = observed[record["case_id"]]
            for name in ("mass_total", "mass_aqueous", "mass_free_gas"):
                assert np.isnan(new[name][mask]).all()
                assert new[name][~mask].tobytes() == old[name][~mask].tobytes()
        row, provenance = tables._row(saved_export._scalar_view(stored), 0)
        assert row["native_total_mass_g"] is None
        assert row["stored_minus_native_total_g"] is None
        assert row["quantified_native_total_mass_g"] == record["quantified_total_mass_g"]
        assert provenance["unknown_mass_g"] is None


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_real_classic_n1_storage_uses_finite_carriers_and_public_nan_maps(tiny_config, phase1_data, tmp_path, monkeypatch, precision):
    config = _inputs(tmp_path, tiny_config, phase1_data, precision=precision)
    _inject(monkeypatch, tmp_path, (0, 0))
    actual = _execute(config, tmp_path / "approved", precision)
    run = n1_saved.read_n1_stored_run(actual, resolve_asset=lambda b: Path(b["path"]))
    for stored, current in zip(run["frames"], actual["records"], strict=True):
        record = _record(current)
        _assert_unknown(record)
        assert stored["mass_totals_g"] == dict.fromkeys(output.TOTALS)
        assert np.any(stored["arrays"]["mask_approved_analysis_gap_area_fraction"] > 0)
        cells = next(iter(stored["arrays"].values())).size
        sizes = n1_capacity.coarse_sizes(cells, precision)
        assert sum(value.nbytes for value in stored["arrays"].values()) <= sum(sizes)
        assert (Path(current["nas_directory"]) / "storage_stored_frame.npz").stat().st_size <= n1_capacity.anonymous_npz_bound(sizes)
        for component, name in output.TOTALS.items():
            cells = stored["arrays"]["cell_mass_" + component + "_g"]
            assert np.isfinite(cells).all()
            assert abs(float(cells.sum(dtype=np.float64)) - record["quantified_" + name]) <= 1e-6 * abs(record["quantified_" + name])
    exported = tables._write(tmp_path / "scoped.csv", tmp_path / "scoped.receipt.json",
        [saved_export._scalar_view(frame) for frame in run["frames"]], manifest_assets=[])
    receipt = json.loads(Path(exported["receipt"]["path"]).read_bytes())
    assert "analysis_scope_json" in receipt["columns"]
    assert all(frame["native_saved_mass_targets_g"]["total"] is None for frame in receipt["frames"])


def _native_scoped(tmp_path, monkeypatch, precision):
    original = output.write_frame
    captured = {}
    def write(root, task, spec, phase, inferred, phase_capture, mass_capture, evaluated, request, **kwargs):
        frame = mass_capture["frame"]
        frame.run_id, frame.provenance = task["run"], {}
        frame.mass_eligible_mask = frame.domain.copy()
        frame.known_physical_domain = frame.domain.copy()
        mask = frame.phase_labels == 255
        _approve(frame, tmp_path, mask)
        _, scope = arithmetic.analysis_gap_scope(frame)
        evaluated["result"] = arithmetic.scoped_result(evaluated["result"], scope)
        exported = evaluated["export"]
        exported["analysis_scope"] = deepcopy(scope)
        exported["quantified_products"] = {name: exported["products"][name].copy() for name in output.DENSITIES.values()}
        for name in output.DENSITIES.values():
            exported["products"][name] = exported["products"][name].copy()
            exported["products"][name][mask] = np.nan
        exported["products"]["approved_analysis_gap_mask"] = mask
        mass_capture["volume_m3"] = mass_capture["volume_m3"].copy()
        mass_capture["volume_m3"][mask] = 0.
        captured["scope"] = scope
        return original(root, task, spec, phase, inferred, phase_capture, mass_capture, evaluated, request, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(output, "write_frame", write)
        state = _fixture(tmp_path, precision=precision)
    state["result"] = arithmetic.public_analysis_scope(arithmetic.scoped_result(state["result"], captured["scope"]))
    _reseal(state)
    return state


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_real_native_storage_roundtrip_preserves_finite_subtotals_and_unknown_mask(tmp_path, monkeypatch, precision):
    state = _native_scoped(tmp_path, monkeypatch, precision)
    stored = native_saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))
    assert stored["mass_totals_g"] == dict.fromkeys(output.TOTALS)
    assert stored["quantified_mass_totals_g"] == dict(total=9., aqueous=3., gas=3.)
    assert np.any(stored["arrays"]["mask_approved_analysis_gap_area_fraction"] > 0)
    assert stored["analysis_scope"]["area_cm2"] is None


def test_saved_scope_cannot_be_dropped_to_restore_whole_rig_totals(tmp_path, monkeypatch):
    state = _native_scoped(tmp_path, monkeypatch, "float32")
    state["document"].pop("analysis_scope")
    _reseal(state)
    with pytest.raises(ValueError, match="scope differs"):
        native_saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))


@pytest.mark.parametrize("n16", [False, True])
def test_absent_pixel_mask_matches_qualified_evaluate_and_saved_scalars(tiny_config, phase1_data, tmp_path, monkeypatch, n16):
    config = _genuine_n16_config(tmp_path, tiny_config, phase1_data, adjacent=True)
    if not n16:
        config["pixel_domain"].pop("n16")
        config.pop("storage")
    with monkeypatch.context() as patch:
        patch.setattr(pixel_pipeline, "evaluate", _qualified_evaluate())
        old = _execute(config, tmp_path / "qualified")
    new = _execute(config, tmp_path / "current")
    for current, prior in zip(new["records"], old["records"], strict=True):
        record, original = _record(current), _record(prior)
        assert "analysis_scope" not in record
        for name in ("total_mass_g", "aqueous_mass_g", "gas_mass_g", "affected_pore_volume_m3",
                     "M_measured_g", "M_edge_g", "U_g", "domains", "prediction", "scalar_result"):
            assert record[name] == original[name]
        for filename in ("products.npz", "native_geometry.npz", *(["storage_stored_frame.npz"] if n16 else [])):
            with np.load(Path(current["nas_directory"]) / filename, allow_pickle=False) as actual, \
                    np.load(Path(prior["nas_directory"]) / filename, allow_pickle=False) as expected:
                assert actual.files == expected.files
                assert all(actual[name].dtype == expected[name].dtype and
                    actual[name].tobytes() == expected[name].tobytes() for name in actual.files)
    if n16:
        stored = n1_saved.read_n1_stored_run(new, resolve_asset=lambda b: Path(b["path"]))["frames"][0]
        assert "analysis_scope" not in stored and "quantified_mass_totals_g" not in stored
        assert all(np.isfinite(value) for value in stored["mass_totals_g"].values())
        row, _ = tables._row(saved_export._scalar_view(stored), 0)
        assert set(row) == set(tables._COLUMNS)


def test_actual_refit_optimizer_omits_incomplete_whole_dose_rows(synthetic, tmp_path):
    frame = synthetic.response["frame"]
    frame.run_id = "r1"
    mask = np.zeros(frame.domain.shape, bool)
    mask[-1, -1] = True
    _approve(frame, tmp_path, mask)
    _, scope = arithmetic.analysis_gap_scope(frame)
    rows = make_rows(["r1", "r2"])
    nearest = next(row for row in rows if row["run"] == "r1" and row["elapsed_h"] == 2.25)
    nearest["analysis_scope"] = deepcopy(scope)
    partial = make_row("r1", 100, .5)
    partial["analysis_scope"] = scope
    assert partial["urest_g"] is not None  # Former stage1 shortcut admitted this row.
    rules = refit.RefitRules.from_plan(make_plan(["r1", "r2"]))
    fit = refit.fit_candidate([*rows, partial], rules)
    assert fit["parameters"]["gain"] == pytest.approx(TRUE_GAIN, rel=1e-4)
    assert fit["parameters"]["beta"] == pytest.approx(TRUE_BETA, rel=1e-3)
    assert fit["anchors_by_run"]["r1"]["frame_index"] != nearest["frame_index"]
    decision = next(row for row in fit["frame_decisions"] if row["frame_index"] == 100)
    assert not decision["included"] and "estimate_geometry_undefined" in decision["omitted_reasons"]
    assert decision["analysis_scope"]["unknown_mass_g"] is None
    rows = [row for row in rows if row["run"] == "r1"]
    for row in rows:
        row["analysis_scope"] = deepcopy(scope)
    with pytest.raises(ValueError, match="No source-bound injection-end volume anchor"):
        refit.fit_candidate(rows, rules)


def test_plain_fallback_reserves_a_real_dense_approved_mask_archive(tmp_path):
    shape, precision = (257, 263), "float32"
    config = dict(frames=[{}], precision=precision)
    bound = lean_capacity.fallback_archives(config, canvas_hw=shape, precision=precision)
    rng = np.random.default_rng(434)
    products = {name: rng.random(shape).astype(precision) if kind == "map" else
        rng.integers(0, 2, size=shape, dtype=np.uint8).astype(bool)
        for name, kind in lean_capacity.PLAIN_PRODUCTS.items()}
    products["phase_labels"] = rng.integers(0, 3, size=shape, dtype=np.uint8)
    products["approved_analysis_gap_mask"] = rng.integers(0, 2, size=shape, dtype=np.uint8).astype(bool)
    path = tmp_path / "products.npz"
    pipeline._write_npz(path, products)
    assert products["approved_analysis_gap_mask"].nbytes == bound["product_payloads"]["approved_analysis_gap_mask"]
    assert all(value.nbytes <= bound["product_payloads"][name] for name, value in products.items())
    assert path.stat().st_size <= bound["products_bytes"]
