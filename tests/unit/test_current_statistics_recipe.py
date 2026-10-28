"""The retained optical recipe must stay bound to the selected acquisition run."""
from copy import deepcopy
import json

import pytest
import toml

from darsia.fluidflower.mass.n1_pipeline import DataReader, asset
from darsia.fluidflower.pipeline.current_statistics_recipe import retained_statistics_recipe


def _inputs(root, *, clear_run="sample", fresh=False):
    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return asset(path)

    models = {}
    for name in ("metadata.json", "flash/flash.json",
                 "color_path_interpretation/color_path_interpretation_0.json",
                 "color_path_interpretation/color_path_interpretation_255.json",
                 "signal_model/signal_model_0.csv", "signal_model/signal_model_255.csv"):
        value = (dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)
                 if name == "flash/flash.json" else {})
        binding = write("models/" + name, value)
        models[binding["path"]] = binding["sha256"]
    freeze = write("freeze.json", dict(path_folder=str(root / "models"), calibration_files=models))
    rgb = write("rgb.npy", {"payload": "must not be decoded by the metadata reader"})
    valid = write("valid.npy", {})
    clear_data = dict(run=clear_run, encoding="float32-camera-linear-unit-rgb",
        rgb_path=rgb["path"], rgb_sha256=rgb["sha256"],
        valid_mask_path=valid["path"], valid_mask_sha256=valid["sha256"])
    if fresh:
        raw = root / "clearwater.ARW"
        raw.write_bytes(b"exact chosen wash acquisition")
        raw_pin = asset(raw)
        processing = write("clearwater.processing.json", dict(run=clear_run,
            raw_source_path=str(raw), raw_source_sha256=raw_pin["sha256"],
            selected_rule="middle of configured wash acquisitions"))
        clear_data.update(raw_source_sha256=raw_pin["sha256"],
            correction_receipt_path=processing["path"], correction_receipt_sha256=processing["sha256"])
    clear = write("clear.json", clear_data)
    matrix = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    transport = write("transport.json", dict(run="sample", clearwater_receipt=clear["path"],
        clearwater_receipt_sha256=clear["sha256"], H_ac28_to_clearwater_grid=matrix))
    recipe = write("recipe.json", dict(runs=[dict(run="sample", alkalinity_M=.001, btb_M=.0007)]))
    atmosphere = write("atmosphere.json", dict(runs=[dict(run="sample", rows=[
        dict(time_hours=-1., pressure_bar=1., temperature_c=22.,
             pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.),
        dict(time_hours=10., pressure_bar=.99, temperature_c=23.,
             pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.)])]))
    base = write("base.json", dict(recipes=recipe, atmosphere=atmosphere))
    dose = write("dose.json", dict(runs=[dict(run="sample", dose_kind="nominal_setpoint",
        injection_stages=[dict(start_h=0., end_h=1., mass_rate_g_s=.01)])]))
    epochs = write("epochs.json", dict(runs=[dict(run="sample", epoch="rig_light")]))
    r2 = write("r2.json", dict(base_input_plan=base, dose_inventory=dose, light_epochs=epochs,
        clearwater_transports={"sample": transport}, swi_by_facies={"1": .32},
        physical_limits=dict(max_depth_m=.0155, max_porosity=.45)))
    parts = write("parts.json", dict(r2_input_plan=r2, r1_freeze=freeze))
    plan = write("plan.json", dict(saved_parts_plan=parts))
    fit = write("fit.json", {})
    calibration = write("calibration.json", dict(calibration_files={"global": fit},
        selected_input_plan=plan, round4_spec_sha256="a" * 64, amendment_sha256="b" * 64))
    series = root / "series.toml"
    series.write_text(toml.dumps(dict(method=dict(active="selected", selected=dict(
        mass_calibration=calibration["path"])))), encoding="utf-8")
    current = dict(format="darsia-current-run-protocols-v1", run="sample",
        atmosphere=dict(run="sample", rows=[dict(time_hours=-1., pressure_bar=1.02,
            temperature_c=23., pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.),
            dict(time_hours=10., pressure_bar=1.01, temperature_c=23.,
                 pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.)]),
        dose=dict(run="sample", dose_kind="nominal_setpoint",
            injection_stages=[dict(start_h=0., end_h=1., mass_rate_g_s=.01)]))
    current_source = write("current-observations.json", current)
    current["sources"] = [current_source]
    selection = dict(runs=[dict(run="sample", current_protocols=current)], input_authority=dict(
        current_series=asset(series), global_fit=fit))
    return selection, rgb, valid


def test_retained_recipe_keeps_exact_optics_and_observations_without_physical_fallback(tmp_path):
    selection, rgb, valid = _inputs(tmp_path)

    class MetadataReader(DataReader):
        def array(self, *args, **kwargs):
            raise AssertionError("Metadata preparation decoded a scientific array")

    result = retained_statistics_recipe(selection, "sample", reader=MetadataReader())
    statistics = result["statistics"]
    assert statistics["shared"]["clearwater"]["rgb"] == rgb
    assert statistics["shared"]["clearwater"]["valid"] == valid
    assert set(statistics["shared"]["optics"]["color_paths"]) == {"0", "255"}
    assert statistics["recipe"] == dict(alkalinity_M=.001, btb_M=.0007)
    assert statistics["summary"]["baseline_gain"] == 24.253536351793787
    assert result["atmosphere"]["rows"][0]["time_hours"] == -1.
    assert result["atmosphere"]["rows"][0]["pressure_bar"] == 1.02
    assert result["dose"]["injection_stages"][0]["mass_rate_g_s"] == .01
    assert result["light_epoch"] == "rig_light"
    assert "physical_reference" not in statistics["shared"]
    assert "physical_correction" not in statistics
    assert "reference_matrix" not in statistics["shared"]


def test_another_calibration_cannot_supply_the_current_recipe(tmp_path):
    selection, _, _ = _inputs(tmp_path)
    selection = deepcopy(selection)
    selection["input_authority"]["global_fit"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="another mass calibration"):
        retained_statistics_recipe(selection, "sample")


def test_clearwater_from_another_run_is_not_accepted(tmp_path):
    selection, _, _ = _inputs(tmp_path, clear_run="other")
    with pytest.raises(ValueError, match="another run or encoding"):
        retained_statistics_recipe(selection, "sample")


def test_new_images_use_original_raw_choice_and_flash_rules_without_old_optics(tmp_path):
    selection, rgb, valid = _inputs(tmp_path, fresh=True)
    # Old pixels and fitted colour paths need not be available for a fresh fit.
    from pathlib import Path
    Path(rgb["path"]).unlink()
    Path(valid["path"]).unlink()
    for pattern in ("color_path_interpretation/*", "signal_model/*"):
        for path in (tmp_path / "models").glob(pattern):
            path.unlink()
    class MetadataReader(DataReader):
        def array(self, *args, **kwargs):
            pytest.fail("fresh recipe decoded old scientific pixels")
    result = retained_statistics_recipe(selection, "sample", reader=MetadataReader(), fresh_images=True)
    assert result["statistics"]["shared"]["optics"] is None
    assert result["statistics"]["shared"]["clearwater"] is None
    assert result["clearwater_source"]["raw"] == asset(tmp_path / "clearwater.ARW")
    assert result["clearwater_source"]["selection_rule"] == "middle of configured wash acquisitions"
    assert result["flash_bounds"] == dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)


def test_retrained_model_retains_recipe_inputs_without_restoring_old_calibration(tmp_path):
    selection, _, _ = _inputs(tmp_path, fresh=True)
    series_path = tmp_path / "series.toml"
    manifest = toml.loads(series_path.read_text())
    method = manifest["method"]["selected"]
    method["previous_mass_calibrations"] = [method.pop("mass_calibration")]
    series_path.write_text(toml.dumps(manifest), encoding="utf-8")
    selection["input_authority"]["current_series"] = asset(series_path)
    result = retained_statistics_recipe(selection, "sample", fresh_images=True)
    assert result["clearwater_source"]["raw"] == asset(tmp_path / "clearwater.ARW")
    assert "mass_calibration" not in toml.loads(series_path.read_text())["method"]["selected"]


def test_setup_optics_uses_identical_clearwater_and_flash_without_model_or_chemistry(tmp_path):
    from darsia.fluidflower.pipeline.current_statistics_recipe import setup_optical_sources
    selection, rgb, valid = _inputs(tmp_path, fresh=True)
    expected = retained_statistics_recipe(selection, "sample", fresh_images=True)
    # These old pixels, chemistry and observations are not Setup optical inputs.
    for name in ("recipe.json", "dose.json", "atmosphere.json", "base.json", "epochs.json",
                 "rgb.npy", "valid.npy", "current-observations.json"):
        (tmp_path / name).unlink()
    class MetadataReader(DataReader):
        def array(self, *args, **kwargs):
            pytest.fail("Setup source metadata decoded a scientific payload")
    actual = setup_optical_sources(selection["input_authority"]["current_series"], "sample", reader=MetadataReader())
    assert actual["clearwater_source"] == expected["clearwater_source"]
    assert actual["flash_bounds"] == expected["flash_bounds"]


def test_setup_optics_refuses_a_different_runs_clearwater(tmp_path):
    from darsia.fluidflower.pipeline.current_statistics_recipe import setup_optical_sources
    selection, _, _ = _inputs(tmp_path, fresh=True, clear_run="different")
    with pytest.raises(ValueError, match="clear-water RAW"):
        setup_optical_sources(selection["input_authority"]["current_series"], "sample")


def _gui_clearwater(root, *, approved=True, runs=("sample",)):
    project = root / "geometry"
    project.mkdir()
    raw = root / "shared_gui_clearwater.ARW"
    raw.write_bytes(b"reviewed common GUI clear-water source")
    pin = asset(raw)
    payload = dict(series_id="generic", runs=[dict(run_id=run, source_mapping=dict(
        clearwater_candidate=str(raw), clearwater_selector="last",
        candidate_status="user_confirmed" if approved else "ready_for_review")) for run in runs])
    (project / "series.json").write_text(json.dumps(payload), encoding="utf-8")
    source = dict(pin, kind="clearwater_image", run_id="")
    (project / "source_manifest.json").write_text(json.dumps(dict(sources=[source])), encoding="utf-8")
    return project, pin


def test_new_series_uses_reviewed_shared_gui_clearwater_without_old_model_or_calibration(tmp_path):
    from darsia.fluidflower.pipeline.current_statistics_recipe import setup_optical_sources
    from darsia.fluidflower.mass.optical_initialization import initial_flash_bounds
    project, raw = _gui_clearwater(tmp_path, runs=("sample", "another"))
    series = tmp_path / "series.toml"
    series.write_text(toml.dumps(dict(schema_version=1, series_id="generic", paths=dict(geometry_facies=str(project)),
        method=dict(active="phase_model", phase_model={}))), encoding="utf-8")
    class MetadataReader(DataReader):
        def path(self, pin):
            if pin["path"] == raw["path"]:
                pytest.fail("GUI preflight must not hash or decode the RAW image")
            return super().path(pin)
        def array(self, *args, **kwargs):
            pytest.fail("GUI preflight must not decode scientific pixels")
    for run in ("sample", "another"):
        value = setup_optical_sources(asset(series), run, reader=MetadataReader())
        assert value["clearwater_source"]["raw"] == raw
        assert value["clearwater_source"]["run"] == run
        assert value["flash_bounds"] == initial_flash_bounds()
        assert "unfitted" in value["flash_convention"]


def test_gui_clearwater_takes_precedence_but_keeps_imported_exact_flash_windows(tmp_path):
    from darsia.fluidflower.pipeline.current_statistics_recipe import setup_optical_sources
    selection, _, _ = _inputs(tmp_path, fresh=True)
    project, raw = _gui_clearwater(tmp_path)
    series = tmp_path / "series.toml"
    manifest = toml.loads(series.read_text())
    manifest.update(schema_version=1, series_id="generic", paths=dict(geometry_facies=str(project)))
    series.write_text(toml.dumps(manifest), encoding="utf-8")
    value = setup_optical_sources(asset(series), "sample")
    assert value["clearwater_source"]["raw"] == raw
    assert value["flash_bounds"] == dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)


def test_unreviewed_gui_candidate_does_not_silently_fall_back_to_old_calibration(tmp_path):
    from darsia.fluidflower.pipeline.current_statistics_recipe import setup_optical_sources
    selection, _, _ = _inputs(tmp_path, fresh=True)
    project, _raw = _gui_clearwater(tmp_path, approved=False)
    series = tmp_path / "series.toml"
    manifest = toml.loads(series.read_text())
    manifest.update(schema_version=1, series_id="generic", paths=dict(geometry_facies=str(project)))
    series.write_text(toml.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Review and approve"):
        setup_optical_sources(asset(series), "sample")
