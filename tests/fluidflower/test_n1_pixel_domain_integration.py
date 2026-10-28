"""Tiny real saved-optics runs for the optional physical-domain N1 route."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower.mass import n1_pipeline as pipeline
from darsia.fluidflower.mass import pixel_pipeline
from darsia.fluidflower.mass.production_precision import PixelMath


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return pipeline.asset(path)


@pytest.fixture
def frozen_run(tmp_path, request):
    """Reuse the unchanged mandatory test's saved optics, then freeze a tiny run."""
    source = Path(__file__).resolve().parents[1] / "test_n1_pipeline.py"
    spec = importlib.util.spec_from_file_location("_saved_n1_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, _ = module.tiny_config.__wrapped__(tmp_path)
    config["geometry_fields"] = dict(
        source_to_native="H_ac28_to_frame_native",
        original_domain_to_native="H_ac28_to_frame_native",
        carried_homography_provenance="original_H_ac28_to_native",
    )
    # The original labels/classifier omit the bottom row, although the original
    # scientific camera buffers and the measured depth field cover that row.
    labels = np.ones((3, 3), np.uint8)
    labels[1, 1] = 2
    labels[2] = 255
    if getattr(request, "param", {}).get("water_column"):
        labels[:2, 2] = 0
    Image.fromarray(labels).save(tmp_path / "frozen_phase.png")
    phase = pipeline.asset(tmp_path / "frozen_phase.png")
    rig = labels != 255
    np.savez(tmp_path / "frozen_classes.npz", rig_domain=rig, modelled=rig,
             filled_observed=rig, filled_unobserved=np.zeros_like(rig))
    classes = pipeline.asset(tmp_path / "frozen_classes.npz")
    np.save(tmp_path / "bound_floor.npy", np.full(3, 1.0))
    config["shared"]["floor_curve"] = pipeline.asset(tmp_path / "bound_floor.npy")
    for frame in config["frames"]:
        frame["labels"], frame["classes"] = phase, classes
    calibration = json.loads(Path(config["calibration"]["path"]).read_bytes())
    calibration["sandmap"] = config["physical_correction"]
    config["calibration"] = _json(tmp_path / "bound_calibration.json", calibration)
    config_bound = _json(tmp_path / "classic_config.json", config)
    reference = pipeline.run_n1(
        dict(run=config["run"], configuration=config_bound, commit="a" * 40),
        tmp_path / "reference", "cpu", "float32",
    )
    fixtures = {name: [[-4, -4], [-3, -4], [-3, -3], [-4, -3]]
                for name in ("chart_left", "chart_right", "chart_bridge")}
    decision = _json(tmp_path / "decisions.json", dict(
        fixture_scope_complete=True, fixture_names=list(fixtures),
        injector_hardware_excluded=False, below_floor_phase_unknown_policy="unknown-no-capacity",
    ))
    config["pixel_domain"] = dict(
        contract="darsia-pixel-domain-v1", adopted_for_execution=True,
        fixture_scope_complete=True, fixture_names=list(fixtures),
        injector_hardware_excluded=False, authorized_decisions=decision,
        max_column_distance_m=.02, raster_scale=256, raster_shift=8,
        geometry_chunk_pixels=100, below_floor_phase_unknown_policy="unknown-no-capacity",
        include_estimated_mass=False,
        baseline_coverage_tolerance=1e-6, heldout_fit_run=False,
        table_fields=["case_id", "M_measured_g", "M_edge_g", "U_g",
                      "phase_unknown_no_capacity_area_cm2"],
    )
    for frame, publication in zip(config["frames"], reference["records"]):
        root = Path(publication["nas_directory"])
        original = json.loads((root / "result.json").read_bytes())
        # The canonical older producer binds correction through calibration
        # and its recorded matrix, without this newer convenience field.
        original.pop("physical_correction")
        original_result = _json(tmp_path / (frame["case_id"] + "_original_result.json"), original)
        geometry = pipeline.asset(root / "native_geometry.npz")
        source_geometry = dict(
            run=config["run"], frame_index=frame["frame_index"],
            source_sha256=frame["source_binding"]["source_sha256"],
            frame_inner_polygon_native=frame["inner_polygon_native"],
            frame_fixture_polygons_native=fixtures, native_origin_xy=[0, 0],
            phase_shape_hw=[3, 3], fixture_scope_complete=False,
        )
        source = _json(tmp_path / (frame["case_id"] + "_source.json"),
                       dict(rig_geometry=source_geometry))
        frame["pixel_domain_geometry"] = dict(
            source_receipt=source, json_pointer="/rig_geometry",
            source_frame_index=frame["frame_index"],
            inner_polygon_native=frame["inner_polygon_native"],
            fixture_polygons_native=copy.deepcopy(fixtures), native_origin_xy=[0, 0],
            phase_shape_hw=[3, 3], fixture_scope_complete=False,
            target_depth=dict(geometry, array_key="depth_m"),
        )
        frame["frozen_response_basis"] = dict(
            result=original_result, geometry=geometry,
            arrays=original["static"]["arrays"], per_frame_geometry=True,
            source_sha256=frame["source_binding"]["source_sha256"],
            phase_sha256=frame["labels"]["sha256"],
            calibration_sha256=config["calibration"]["sha256"],
            # This deliberately unavailable comparison product must never be
            # opened or substituted into the new computation.
            products=dict(path=str(tmp_path / "not_an_input.npz"), sha256="0" * 64),
        )
    return config, reference


def test_optional_domain_runs_real_optics_and_publishes_frozen_pixel_identity(frozen_run, tmp_path):
    """New support preserves old finite pixels and explicitly reports phase255."""
    config, reference = frozen_run
    bound = _json(tmp_path / "domain_config.json", config)
    events = []
    actual = pipeline.run_n1(
        dict(run=config["run"], configuration=bound, commit="b" * 40),
        tmp_path / "new_domain", "cpu", "float32", events.append,
    )
    assert actual["complete"] and actual["worker_references_closed"]
    assert actual["ordered_case_ids"] == ["sample_1", "sample_3"]
    assert [e["state"] for e in events] == ["frame_started", "frame_completed"] * 2
    assert Path(actual["table"]["path"]).is_file()
    with Path(actual["table"]["path"]).open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 2 and all(float(row["phase_unknown_no_capacity_area_cm2"]) == pytest.approx(.03) for row in table)
    for current, previous in zip(actual["records"], reference["records"]):
        root, old = Path(current["nas_directory"]), Path(previous["nas_directory"])
        record = json.loads((root / "result.json").read_bytes())
        assert record["scope"] == "pixel_domain_evaluation"
        assert record["M_measured_g"] == record["total_mass_g"]
        assert record["pixel_domain"]["mass_integration_dtype"] == "float64"
        assert record["prediction"]["heldout_fit_run"] is False
        assert record["prediction"]["evaluation_is_leave_one_run_out"] is False
        assert record["prediction"]["delta"] >= 0
        basis = record["pixel_domain"]["original_response_basis"]
        assert "physical_correction" not in json.loads(Path(basis["result"]["path"]).read_bytes())
        assert record["phase_unknown_no_capacity_area_cm2"] == pytest.approx(.03)
        with np.load(root / "products.npz") as new, np.load(old / "products.npz") as prior:
            selected = prior["main_mass_eligible_mask"] & np.isfinite(prior["concentration_aqueous"])
            for name in ("concentration_aqueous", "saturation_free_gas", "mass_total",
                         "mass_aqueous", "mass_free_gas"):
                assert new[name].dtype == np.float32
                np.testing.assert_array_equal(new[name][selected], prior[name][selected])
            np.testing.assert_array_equal(new["phase_labels"], prior["phase_labels"])
            assert new["domain_mask"].all() and new["visible_mask"].all()
            assert new["unknown_without_capacity_mask"][2].all()
            assert np.isnan(new["concentration_aqueous"][2]).all()
            assert not new["mass_contribution_mask"][2].any()
            for name in ("measured_mass_g", "edge_mass_g", "unknown_capacity_g"):
                assert new[name].dtype == np.float32
                assert not new[name][2].any()
            assert new["concentration_donor_distance_m"].dtype == np.float32
            assert new["geometry_donor_distance_m"].dtype == np.float32
        with np.load(root / "native_geometry.npz") as new, np.load(old / "native_geometry.npz") as prior:
            assert new["transport_weight_l"].dtype == np.float64
            np.testing.assert_array_equal(new["depth_m"], prior["depth_m"])
            np.testing.assert_array_equal(new["transport_weight_l"][selected], prior["transport_weight_l"][selected])
            assert (new["transport_weight_l"][2] > 0).all()
            assert not prior["transport_weight_l"][2].any()
        for name, value in current["files"].items():
            assert pipeline.asset(root / name)["sha256"] == value["sha256"]


@pytest.mark.parametrize("change,match", [
    ("unadopted", "adopted policy"),
    ("foreign_source", "another source"),
    ("wrong_fixture", "differs from its source"),
    ("wrong_depth", "Target depth"),
])
def test_domain_inputs_fail_closed_without_fallback(frozen_run, change, match):
    """A stale policy/geometry/source cannot silently enter the classic route."""
    config, _ = frozen_run
    if change == "unadopted":
        config["pixel_domain"]["adopted_for_execution"] = False
    elif change == "foreign_source":
        config["frames"][0]["source_binding"]["source_sha256"] = "d" * 64
    elif change == "wrong_fixture":
        config["frames"][0]["pixel_domain_geometry"]["fixture_polygons_native"].pop("chart_left")
    else:
        config["frames"][0]["pixel_domain_geometry"]["target_depth"]["array_key"] = "porosity"
    with pytest.raises(ValueError, match=match):
        native = pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
        native.evaluate(config["frames"][0])


def test_absent_policy_keeps_classic_dispatch_and_products(frozen_run, monkeypatch):
    """The legacy path cannot invoke the optional domain implementation."""
    config, _ = frozen_run
    config.pop("pixel_domain")
    def forbidden(*args, **kwargs):
        pytest.fail("Optional domain implementation entered the classic path")
    monkeypatch.setattr(pixel_pipeline, "prepare_frame", forbidden)
    monkeypatch.setattr(pixel_pipeline, "evaluate", forbidden)
    native = pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
    value = native.evaluate(config["frames"][0])
    assert "pixel_domain" not in value
    assert set(value["export"]["products"]) == {
        "concentration_aqueous", "saturation_free_gas", "phase_labels",
        "mass_total", "mass_aqueous", "mass_free_gas",
    }


def test_exact_source_chart_polygon_is_applied_without_old_floor_cut(frozen_run, tmp_path):
    """An admitted chart removes its true pixel while the raw bottom stays inside."""
    config, _ = frozen_run
    frame = config["frames"][0]
    declared = frame["pixel_domain_geometry"]
    source = json.loads(Path(declared["source_receipt"]["path"]).read_bytes())
    polygon = [[0, 0], [.2, 0], [0, .2]]
    source["rig_geometry"]["frame_fixture_polygons_native"]["chart_left"] = polygon
    declared["fixture_polygons_native"]["chart_left"] = polygon
    declared["source_receipt"] = _json(tmp_path / "inside_chart.json", source)
    native = pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
    prepared, _, _, _, context = native.load_frame(frame)
    assert not prepared.domain[0, 0]
    assert context["fixtures"]["chart_left"][0, 0]
    assert prepared.domain[2].all()
    assert (prepared.phase_labels[2] == 255).all()


def test_target_depth_descriptor_locator_is_really_verified(frozen_run, tmp_path):
    """A correct declared digest does not excuse a nonexistent target-depth path."""
    config, _ = frozen_run
    config["frames"][0]["pixel_domain_geometry"]["target_depth"]["path"] = str(tmp_path / "absent.npz")
    native = pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
    with pytest.raises(FileNotFoundError):
        native.evaluate(config["frames"][0])


@pytest.mark.parametrize("key,value", [("max_column_distance_m", 0),
                                      ("baseline_coverage_tolerance", 1),
                                      ("heldout_fit_run", True),
                                      ("injector_hardware_excluded", None)])
def test_new_policy_rejects_zero_distance_or_unadopted_semantics(frozen_run, key, value):
    """Runtime data must satisfy the actual kernel and full-freeze contracts."""
    config, _ = frozen_run
    config["pixel_domain"][key] = value
    with pytest.raises(ValueError):
        pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))


@pytest.mark.parametrize("frozen_run", [{"water_column": True}], indirect=True)
def test_water_with_unknown_porosity_contributes_zero_not_nan(frozen_run):
    """Unknown water geometry remains visible but cannot contaminate mass totals."""
    config, _ = frozen_run
    del config["shared"]["porosity_by_facies"]["2"]
    native = pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
    value = native.evaluate(config["frames"][0])
    assert value["extra_products"]["geometry_unknown_mask"][:2, 2].all()
    assert value["extra_products"]["visible_mask"][:2, 2].all()
    assert not value["extra_products"]["mass_contribution_mask"][:2, 2].any()
    assert not value["geometry_fields"]["transport_weight_l"][:2, 2].any()
    assert np.isfinite(value["result"]["total_mass_g"])
    assert value["result"]["U_g"] == 0


def test_source_valid_zero_colour_is_visible_and_hardware_flag_is_generic(frozen_run):
    """Zero colour is not a hidden visibility cut, and polygon roles are explicit."""
    config, _ = frozen_run
    config["pixel_domain"]["injector_hardware_excluded"] = True
    native = pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
    frame = config["frames"][0]
    current = np.full((3, 3, 3), .5, np.float32)
    current[0, 0, 0] = 0
    with Image.open(native.reader.path(frame["labels"])) as image:
        labels = np.asarray(image).copy()
    prepared, _, _, visible, _ = pixel_pipeline.prepare_frame(
        native, frame, current=current, valid=np.ones((3, 3), bool),
        baseline=np.ones_like(current), coverage=np.ones((3, 3), np.float32),
        clear_rgb=np.ones_like(current), clear_supported=np.ones((3, 3), bool),
        labels=labels, area=np.full((3, 3), 1e-6), current_matrix=np.eye(3),
    )
    assert visible[0, 0] and prepared.supported_mask[0, 0]
    assert prepared.current_rgb[0, 0, 0] == 0
