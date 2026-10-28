"""New named boxes reuse finished saved products, without scientific producers."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import analysis_box_reanalysis as subject
from darsia.fluidflower.metric_analysis_domains import FORMAT, box
from darsia.fluidflower.storage_profile import _identity
from tests.unit.gui.test_production_results_view import production_job, snapshot, SHAPE
from tests.unit.published_dataset.synthetic import as_json, npz_bytes, sha
from tests.unit.gui.test_painting_analysis_boxes_ui import context  # noqa: F401


def pin(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), sha256=sha(raw), size_bytes=len(raw))


def boxes(*rows):
    return dict(format=FORMAT, units="mm", x_axis="right", y_axis="up",
                origin="inner left-wall and bottom intersection", domains=list(rows))


def arrays():
    x, y = np.meshgrid(np.arange(3) * .001 + .0005, np.arange(2) * .001 + .0005)
    values = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    return dict(masses=dict(total=values * 3, aqueous=values * 2, gas=values),
                x_m=x, y_m=y, weights_l=np.full(values.shape, .25))


def test_native_integrates_density_times_saved_litres_half_open_named_boxes():
    rows = [box("Left", [0, 1, 0, 2]), box("Top", [0, 3, 1, 2])]
    result = subject.integrate_mass_arrays(**arrays(), boxes=rows)
    assert result[0]["gas_mass_g"] == (1 + 4) * .25
    assert result[1]["gas_mass_g"] == (4 + 5 + 6) * .25
    assert result[0]["total_mass_g"] == result[0]["aqueous_mass_g"] + result[0]["gas_mass_g"]
    assert result[0]["sample_count"] == 2 and result[0]["complete"]


def test_unknown_mass_is_not_zero_and_outside_unknown_does_not_poison_other_box():
    values = arrays()
    values["masses"]["total"][0, 0] = np.nan
    rows = subject.integrate_mass_arrays(**values, boxes=[box("Unknown", [0, 1, 0, 1]),
                                                         box("Known", [1, 3, 0, 1])])
    assert rows[0]["total_mass_g"] is None
    assert rows[0]["gas_quantified_mass_g"] == 0
    assert rows[0]["unknown_sample_count"] == 1 and not rows[0]["complete"]
    assert rows[1]["gas_mass_g"] == (2 + 3) * .25
    assert rows[1]["complete"]


def test_no_samples_is_not_valid_zero_mass():
    row = subject.integrate_mass_arrays(**arrays(), boxes=[box("Outside", [30, 40, 20, 30])])[0]
    assert row["sample_count"] == 0 and not row["complete"]
    assert row["total_mass_g"] is None and row["total_quantified_mass_g"] == 0


def test_metric_cells_are_grams_not_density_or_fractional_reinterpretation():
    values = arrays()
    values.pop("weights_l")
    result = subject.integrate_mass_arrays(**values, boxes=[box("Two cells", [.4, 1.6, .4, .6])],
                                           storage="metric_cells", resolution_mm=1.0)
    assert result[0]["gas_mass_g"] == 3
    assert result[0]["sample_count"] == 2
    assert result[0]["resolution_mm"] == 1.0


def saved_job(tmp_path, monkeypatch, *, native=True, runs=("r1", "r2"), facies=None):
    fixture = production_job(tmp_path, states=tuple((run, "complete") for run in runs), frames=2)
    model_paths = {}
    for run in runs:
        model = dict(run=run, run_id=run, reference_image=dict(path="unused.raw", sha256="a" * 64),
                     metric_frame=dict(unit="mm", x_axis="right", y_axis="up",
                                       origin="inner left-wall and bottom intersection"),
                     H_camera_u_to_metric_mm=np.eye(3).tolist(), baseline_native_grid=dict(
                         H_rig_corrected_native_to_baseline_camera_u=[[1, 0, .5], [0, 1, .5], [0, 0, 1]]))
        model_path = tmp_path / (run + "-model.json")
        model_path.write_bytes(as_json(model))
        model_paths[run] = pin(model_path)
        attempt = fixture.attempts[run]
        done_path = fixture.job / "done" / (run + ".json")
        done = json.loads(done_path.read_bytes())
        index_path = fixture.job / done["products_index"]["relative_path"]
        index = json.loads(index_path.read_bytes())
        frames = []
        for number in range(1, 3):
            case = f"{run}_f{number}"
            folder = attempt / "frames" / case
            receipt_path = tmp_path / (case + "-bridge.json")
            receipt_path.write_bytes(as_json(dict(build_to_frame_native_h=np.eye(3).tolist())))
            result_path = folder / "result.json"
            result = json.loads(result_path.read_bytes())
            result["scalar_result"] = dict(local_build_accounting=dict(source_binding=dict(
                metric_model=pin(model_path), bridge_frame=pin(receipt_path))))
            if native:
                weights = np.full(SHAPE, .125, np.float64)
                fields = dict(transport_weight_l=weights) | ({} if facies is None else dict(facies=facies))
                geometry_path = folder / "native_geometry.npz"
                geometry_path.write_bytes(npz_bytes(fields))
                result["static"] = dict(asset=dict(pin(geometry_path), path="native_geometry.npz"),
                                        arrays={name: _identity(value) for name, value in fields.items()})
                index["products"].append(dict(product_id=f"frames/{case}/native_geometry.npz",
                    machine_id="synthetic", local_path=str(geometry_path),
                    sha256=pin(geometry_path)["sha256"], bytes=geometry_path.stat().st_size, kind="map"))
            result_path.write_bytes(as_json(result))
            entry = next(row for row in index["products"] if row["product_id"] == f"frames/{case}/result.json")
            entry.update(sha256=pin(result_path)["sha256"], bytes=result_path.stat().st_size)
            shared = index_path.parent / entry["metadata_copy"]["relative_path"]
            shared.write_bytes(result_path.read_bytes())
            entry["metadata_copy"].update(sha256=entry["sha256"], bytes=entry["bytes"])
            frames.append(dict(case_id=case, current=dict(native_origin_xy=[0, 0], phase_shape_hw=list(SHAPE))))
        config_path = tmp_path / (run + "-config.json")
        config_path.write_bytes(as_json(dict(frames=frames)))
        task_path = fixture.job / "results" / run / "task.json"
        task = json.loads(task_path.read_bytes())
        task["configuration"] = dict(root_alias="series", relative_path=config_path.name,
            sha256=pin(config_path)["sha256"], bytes=config_path.stat().st_size, role="configuration")
        task_path.write_bytes(as_json(task))
        index_path.write_bytes(as_json(index))
        done["products_index"].update(sha256=pin(index_path)["sha256"], bytes=index_path.stat().st_size)
        done_path.write_bytes(as_json(done))
    geometry_index = tmp_path / "geometry-index.json"
    geometry_index.write_bytes(as_json(dict(runs={run: dict(reference_model=value) for run, value in model_paths.items()})))
    config = boxes(box("Lower", [0, 2, 0, 2]), box("All", [0, 10, 0, 10]))
    config["geometry_index"] = pin(geometry_index)
    from darsia.fluidflower import store_map
    monkeypatch.setattr(store_map, "read_store_map", lambda *a: SimpleNamespace(roots=lambda series_id: {"series": str(tmp_path)}))
    return fixture, config


def read_table(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8"))))


def test_real_finished_reader_to_immutable_tables_and_manifest_preserves_sources(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    progress = []
    target = fixture.job / "analysis" / "boxes" / "new"
    result = subject.reanalyze_saved_boxes(fixture.job, config, target, actor="Tester", reason="New boxes",
                                          on_progress=progress.append)
    assert result["run_count"] == 2 and result["frame_count"] == 4
    table = read_table(result["tables"][0]["path"])
    assert len(table) == 4 and {row["name"] for row in table} == {"Lower", "All"}
    assert all(row["storage"] == "native" for row in table)
    manifest = json.loads(Path(result["manifest"]).read_bytes())
    assert manifest["complete"] and manifest["boxes_configuration"] == config
    assert manifest["runs"][0]["frames"][0]["source"]["coordinate_frame"]["mass_model"]
    assert progress[-1]["completed"] == progress[-1]["total"] == 4
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    assert all(str(target.relative_to(tmp_path)) in name for name in after.keys() - before.keys())


def test_auto_uses_metric_fallback_and_records_actual_resolution(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch, native=False)
    result = subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "derived", actor="Tester", reason="Metric")
    row = read_table(result["tables"][0]["path"])[0]
    assert row["storage"] == "metric_cells" and float(row["resolution_mm"]) == 1
    assert int(row["sample_count"]) == 4
    manifest = json.loads(Path(result["manifest"]).read_bytes())
    assert manifest["runs"][0]["frames"][0]["source"]["native_unavailable"]


def test_changed_native_data_fails_without_hiding_corruption_in_metric_fallback(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch)
    path = fixture.attempts["r1"] / "frames" / "r1_f1" / "products.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size differs|SHA|record"):
        subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "derived", actor="Tester", reason="Changed")
    assert not (tmp_path / "derived" / "manifest.json").exists()


def test_boxes_cannot_silently_use_other_origin_or_measured_transform(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch)
    config["origin"] = "another physical origin"
    with pytest.raises(ValueError, match="origins"):
        subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "derived", actor="Tester", reason="Mismatch")


def test_different_measured_frame_is_rejected_even_when_axes_and_units_match(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch)
    index = json.loads(Path(config["geometry_index"]["path"]).read_bytes())
    path = tmp_path / "different-model.json"
    model = json.loads(Path(index["runs"]["r1"]["reference_model"]["path"]).read_bytes())
    model["H_camera_u_to_metric_mm"][0][2] = 12
    path.write_bytes(as_json(model))
    index["runs"]["r1"]["reference_model"] = pin(path)
    index_path = tmp_path / "different-index.json"
    index_path.write_bytes(as_json(index))
    config["geometry_index"] = pin(index_path)
    with pytest.raises(ValueError, match="different measured coordinate"):
        subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "derived", actor="Tester", reason="Mismatch")


def test_cancellation_never_publishes_false_complete_and_keeps_source_bytes(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch)
    before = snapshot(fixture.root)
    calls = []
    def progress(value):
        calls.append(value)
        raise InterruptedError("User cancelled")
    target = tmp_path / "derived"
    with pytest.raises(InterruptedError, match="cancelled"):
        subject.reanalyze_saved_boxes(fixture.job, config, target, actor="Tester", reason="Cancel", on_progress=progress)
    assert len(calls) == 1 and not (target / "manifest.json").exists()
    assert {name: snapshot(fixture.root)[name] for name in before} == before


@pytest.mark.parametrize("target", ["results/new", "queue/new", "analysis/new", "analysis/boxes"])
def test_output_cannot_overwrite_or_pollute_job_inputs(tmp_path, monkeypatch, target):
    fixture, config = saved_job(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="analysis/boxes|new result"):
        subject.reanalyze_saved_boxes(fixture.job, config, fixture.job / target, actor="Tester", reason="Bad path")


def test_empty_or_duplicate_box_selection_and_existing_output_rejected(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch)
    empty = deepcopy(config)
    empty["domains"] = []
    with pytest.raises(ValueError, match="At least one"):
        subject.reanalyze_saved_boxes(fixture.job, empty, tmp_path / "empty", actor="Tester", reason="Empty")
    duplicate = deepcopy(config)
    duplicate["domains"].append(duplicate["domains"][0])
    with pytest.raises(ValueError, match="unique"):
        subject.reanalyze_saved_boxes(fixture.job, duplicate, tmp_path / "duplicate", actor="Tester", reason="Duplicate")
    with pytest.raises(ValueError, match="new result folder"):
        subject.reanalyze_saved_boxes(fixture.job, config, tmp_path, actor="Tester", reason="Existing")


def test_actual_analysis_gui_uses_saved_backend_and_displays_derived_masses(context, tmp_path, monkeypatch, qtbot):
    from tests.unit.gui.test_analysis_box_reanalysis_ui import _saved
    _saved(context)
    fixture, config = saved_job(tmp_path, monkeypatch)
    controller = context.controller
    controller._saved_configuration = deepcopy(config)
    controller.dirty = False
    panel = controller.reanalysis
    panel.integrator = subject.reanalyze_saved_boxes
    panel.jobs.addItem("Finished synthetic production", str(fixture.job))
    panel.output.setText(str(tmp_path / "derived"))
    panel.refresh()
    before = snapshot(fixture.root)
    assert panel.start()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=10000)
    assert "Completed: 2 runs, 4 frames" in panel.status.text()
    assert panel.results.rowCount() == 8
    assert panel.results.item(0, 0).text() == "r1 / Lower"
    assert float(panel.results.item(0, 2).text()) >= 0
    assert panel.open.isEnabled()
    manifests = list((tmp_path / "derived").glob("*/manifest.json"))
    assert len(manifests) == 1 and json.loads(manifests[0].read_bytes())["complete"]
    after = snapshot(fixture.root)
    assert {name: after[name] for name in before} == before


# Rows 0-2 are the upper sand layer (facies 1), rows 3-5 the lower one (facies 2).
FACIES = np.repeat(np.array([1, 1, 1, 2, 2, 2], np.int32)[:, None], SHAPE[1], axis=1)
# The manifest keys of a box result; the layer option must not add any when it is off.
BOX_MANIFEST_KEYS = {"format", "complete", "actor", "reason", "created_utc", "source_job", "boxes_configuration",
                     "boxes_sha256", "boxes", "source_preference", "source_metadata", "units", "runs", "frame_count",
                     "selection", "unknown_policy", "coarse_limitation", "run_count"}


def _without_time(path):
    """A result manifest without its creation time and with its tables named by file only."""
    manifest = json.loads(Path(path).read_bytes())
    manifest.pop("created_utc")
    for row in manifest["runs"]:
        row["table"]["path"] = Path(row["table"]["path"]).name
    return manifest


def test_the_layer_option_off_keeps_the_box_output_byte_identical(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch, facies=FACIES)
    off = subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "off", actor="Tester", reason="Boxes")
    on = subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "on", actor="Tester", reason="Boxes",
                                       layers=True, layer_names={1: "Upper sand"})
    assert set(off) == {"manifest", "tables", "boxes", "run_count", "frame_count"}
    assert sorted(path.name for path in (tmp_path / "off").iterdir()) == ["manifest.json", "run-0001.csv",
                                                                         "run-0002.csv"]
    for name in ("run-0001.csv", "run-0002.csv"):
        assert (tmp_path / "off" / name).read_bytes() == (tmp_path / "on" / name).read_bytes()
    manifest_off, manifest_on = _without_time(off["manifest"]), _without_time(on["manifest"])
    assert set(manifest_off) | {"created_utc"} == BOX_MANIFEST_KEYS
    assert all(set(row) == {"run", "table", "frames", "completion"} for row in manifest_off["runs"])
    # With the option on, only the layer additions differ.
    manifest_on.pop("layers")
    for row in manifest_on["runs"]:
        row.pop("layer_table")
    assert manifest_on == manifest_off


def test_the_layer_option_gives_each_sand_layer_its_own_table(tmp_path, monkeypatch):
    from tests.unit.gui.test_production_results_view import frame_arrays

    fixture, config = saved_job(tmp_path, monkeypatch, facies=FACIES)
    result = subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "derived", actor="Tester",
                                           reason="Layers", layers=True, layer_names={1: "Upper sand"})
    assert result["frames_without_native_facies"] == 0 and len(result["layer_tables"]) == 2
    rows = read_table(tmp_path / "derived" / "layers-run-0001.csv")
    assert [(row["case_id"], row["name"], row["domain_id"]) for row in rows] == [
        ("r1_f1", "Upper sand", "facies:1"), ("r1_f1", "Facies 2", "facies:2"),
        ("r1_f2", "Upper sand", "facies:1"), ("r1_f2", "Facies 2", "facies:2")]
    for row, (index, facies) in zip(rows, ((0, 1), (0, 2), (1, 1), (1, 2))):
        expected = float((frame_arrays(index)["mass_aqueous"][FACIES == facies].astype(np.float64) * .125).sum())
        assert float(row["aqueous_quantified_mass_g"]) == pytest.approx(expected, rel=1e-12)
        assert row["complete"] == "True" and float(row["aqueous_mass_g"]) == pytest.approx(expected, rel=1e-12)
    manifest = json.loads(Path(result["manifest"]).read_bytes())
    assert manifest["layers"]["names"] == {"1": "Upper sand"}
    assert manifest["runs"][0]["layer_table"]["path"].endswith("layers-run-0001.csv")


def test_frames_without_saved_facies_have_no_layer_rows(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch)
    result = subject.reanalyze_saved_boxes(fixture.job, config, tmp_path / "derived", actor="Tester",
                                           reason="Layers", layers=True)
    assert result["frames_without_native_facies"] == 4
    assert read_table(tmp_path / "derived" / "layers-run-0001.csv") == []


def test_facies_masses_follow_the_box_rules_for_unknown_support():
    masses = dict(total=np.array([[3., 3.], [np.nan, 1.]]), aqueous=np.array([[2., 2.], [1., 1.]]),
                  gas=np.array([[1., 1.], [0., 0.]]))
    facies = np.array([[1, 1], [2, 0]])
    weights = np.array([[.5, .5], [.5, 0.]])
    rows = subject.integrate_facies_arrays(masses, facies, weights_l=weights,
                                           unknown_mask=np.zeros((2, 2), bool), names={2: "Lower"})
    assert [(row["facies_id"], row["name"]) for row in rows] == [(1, "Facies 1"), (2, "Lower")]  # 0: no pore volume
    assert rows[0]["complete"] and rows[0]["total_mass_g"] == 3. and rows[0]["gas_mass_g"] == 1.
    assert not rows[1]["complete"] and rows[1]["total_mass_g"] is None and rows[1]["unknown_sample_count"] == 1
    assert rows[1]["aqueous_quantified_mass_g"] == 0.
    for bad in (facies - 1, facies.astype(float)):
        with pytest.raises(ValueError):
            subject.integrate_facies_arrays(masses, bad, weights_l=weights, unknown_mask=np.zeros((2, 2), bool))


def test_layer_names_come_from_the_series_facies_table(tmp_path, monkeypatch):
    from darsia.fluidflower import geometry_facies

    table = tmp_path / "facies_properties.csv"
    table.write_text("id,name,porosity\n1,Upper sand,0.4\n2,,0.4\n", encoding="utf-8")
    current = dict(artifacts=dict(facies_properties_path=str(table)))
    monkeypatch.setattr(geometry_facies, "GeometryRevisionStore", lambda project: SimpleNamespace(current=lambda: current))
    assert subject.layer_names(object()) == {1: "Upper sand", 2: "Facies 2"}
    current["artifacts"]["facies_properties_path"] = str(tmp_path / "missing.csv")
    assert subject.layer_names(object()) == {}
