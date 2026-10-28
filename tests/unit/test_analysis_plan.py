"""Staged plan metadata tests with explicitly synthetic completion fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from darsia.fluidflower import analysis_plan as plans
from darsia.fluidflower.mass.n1_pipeline import DataReader, _atmosphere
from darsia.fluidflower.pipeline.job_queue import descriptor
from .test_staged_data import phase1_data, _closure, _bound


def _original(data):
    result = {key: deepcopy(data[key]) for key in (
        "run", "frames", "native", "phase", "history_policy",
        "selection_index_field", "ordered_case_ids", "provenance")}
    calibration = _bound("calibration.json")
    template = dict(deepcopy(data["statistics"]), format="darsia-n1-run-v1",
                    calibration=calibration, anchor={"original": "not a phase-one input"})
    result.update(format="darsia-native-phase-n1-run-v1", table_fields=["mass_g"],
                  mass=dict(calibration=calibration, template=template))
    return result


def _job():
    return dict(commit="1" * 40, precision="float32", created_utc="2026-01-01T00:00:00Z",
                environment_lock=dict(path="requirements/worker.lock", sha256="2" * 64))


def _resources(data):
    return {data["run"]: dict(backend="cpu", threads=1)}


def test_phase1_maps_complete_roles_without_calibration_or_input_mutation(phase1_data):
    original = _original(phase1_data)
    before = deepcopy(original)
    result = plans.phase1_configuration(original, precision="float32")
    assert result == phase1_data
    assert "mass" not in result and "calibration" not in result["statistics"]
    assert original == before
    with pytest.raises(ValueError, match="float32"):
        plans.phase1_configuration(original, precision="float64")


def _current_build_original_TESTONLY(phase1_data):
    """Unsealed RAW DATA, not an actual Build, approval or physical acceptance."""
    from .test_current_build_binding import binding_TESTONLY
    original = _original(phase1_data)
    frame = original["frames"][0]
    raw = dict(_bound("selected.ARW"), sha256=frame["source"]["source_sha256"])
    frame["source"]["source_binding"] = {"raw_acquisition": raw}
    frame["native_task"] = dict(kind="raw_rig", geometry=_bound("native_geometry.json"),
                                processor={"raw_binding": _bound("raw_binding.json")})
    roles = frame["phase_roles"]
    roles.pop("native_row")
    roles.pop("probability_receipt")
    roles.update(encoding_contract=None, photo_sand_reference=_bound("sand.npz"),
                 photo_proof={"TESTONLY": "unsealed"})
    binding, _ = binding_TESTONLY()
    binding["run"] = original["run"]
    binding["frames"][0].update(case_id=frame["case_id"], raw_acquisition=deepcopy(raw),
                                 geometry=deepcopy(frame["geometry"]))
    original["current_build_binding"] = binding
    return original


def test_current_build_binding_survives_p1_conversion_and_original_configuration_seal(phase1_data, tmp_path):
    """UNRUN: immutable metadata roundtrip; no referenced scientific inputs open."""
    from darsia.fluidflower.pipeline.staged_data import seal_phase1_configuration, validate_phase1_run
    original = _current_build_original_TESTONLY(phase1_data)
    before = deepcopy(original)
    config = plans.phase1_configuration(original, precision="float32")
    assert config["current_build_binding"] == original["current_build_binding"]
    assert config["current_build_binding"] is not original["current_build_binding"]
    assert original == before and "mass" not in config
    pin = seal_phase1_configuration(tmp_path / "phase1_TESTONLY.json", config)
    saved = DataReader(unc_only=False).json(pin)
    assert saved == config and validate_phase1_run(saved) == config
    changed = deepcopy(saved)
    changed["current_build_binding"]["frames"][0]["raw_acquisition"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="queued RAW source"):
        validate_phase1_run(changed)
    assert DataReader(unc_only=False).json(pin) == saved


def test_p1_conversion_refuses_geometry_that_detaches_current_build(phase1_data):
    """UNRUN: frame_data cannot silently replace the declared Build geometry."""
    original = _current_build_original_TESTONLY(phase1_data)
    frame = original["frames"][0]
    measured = dict(format="darsia-measured-frame-geometry-v1", run=original["run"],
        case_id=frame["case_id"], original_geometry=deepcopy(frame["geometry"]),
        source_to_frame_native=deepcopy(frame["phase_roles"]["reference_to_current_h"]),
        mass_shared_reference_matrix=[[.001, 0., 0.], [0., -.001, 0.], [0., 0., 1.]],
        worker_geometry=_bound("other-measured.json"), phase_area=deepcopy(frame["phase_roles"]["area"]))
    before = deepcopy(original)
    with pytest.raises(ValueError, match="queued RAW source or geometry"):
        plans.phase1_configuration(original, precision="float32", frame_data={frame["case_id"]: measured})
    assert original == before


def test_running_pressure_uses_original_consumer_and_keeps_required_frame_abi(phase1_data):
    original = _original(phase1_data)
    atmosphere = dict(run=original["run"], complete=True, rows=[
        dict(time_hours=t, pressure_bar=p, temperature_c=23.,
             pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.)
        for t, p in [(0., 1.), (2., 1.2)]])
    before = deepcopy((original, atmosphere))
    result = plans.phase1_configuration(original, precision="float32", atmosphere=atmosphere)
    actual = result["frames"][0]["mass_spec"]["atmosphere"]
    assert actual == _atmosphere(dict(atmosphere_series=atmosphere), {"elapsed_h": 1.})
    assert actual["pressure_bar"] == 1.1
    assert _atmosphere(result["statistics"], dict(elapsed_h=1., atmosphere=actual)) == actual
    assert result["statistics"]["atmosphere_series"] == atmosphere
    assert (original, atmosphere) == before
    atmosphere["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        plans.phase1_configuration(original, precision="float32", atmosphere=atmosphere)


def test_legacy_method_digests_are_explicit_and_clearwater_proof_remains_provenance(phase1_data):
    original = _original(phase1_data)
    keys = ("round4_spec_sha256", "amendment_sha256")
    bindings = {key: original["mass"]["template"]["bindings"].pop(key) for key in keys}
    original["mass"]["template"]["shared"]["clearwater"]["receipt"] = _bound("proof.json")
    before = deepcopy(original)
    result = plans.phase1_configuration(original, precision="float32", method_bindings=bindings)
    assert "receipt" not in result["statistics"]["shared"]["clearwater"]
    assert result["statistics"]["provenance"]["original_clearwater_receipt"] == _bound("proof.json")
    assert all(result["statistics"]["bindings"][key] == bindings[key] for key in keys)
    assert original == before
    with pytest.raises(ValueError, match="method binding"):
        plans.phase1_configuration(original, precision="float32")


def test_geometry_requires_exact_case_source_positioning_and_complete_coverage(phase1_data):
    original = _original(phase1_data)
    frame = original["frames"][0]
    matrix = [[.001, 0., 0.], [0., -.001, 0.], [0., 0., 1.]]
    measured = dict(format="darsia-measured-frame-geometry-v1", run=original["run"],
        case_id=frame["case_id"], original_geometry=deepcopy(frame["geometry"]),
        source_to_frame_native=deepcopy(frame["phase_roles"]["reference_to_current_h"]),
        mass_shared_reference_matrix=matrix, worker_geometry=_bound("measured.json"),
        phase_area=deepcopy(frame["phase_roles"]["area"]))
    measured["phase_area"]["factor"] = 1.03
    measured["original_geometry"]["path"] = "//unit-only/relocated/geometry.json"
    frames = {frame["case_id"]: measured}
    result = plans.phase1_configuration(original, precision="float32", frame_data=frames)
    assert result["frames"][0]["geometry"] == measured["worker_geometry"]
    assert result["frames"][0]["native_task"] == frame["native_task"]
    assert result["frames"][0]["mass_spec"] == frame["mass_spec"]
    assert result["statistics"]["shared"]["reference_matrix"] == matrix
    with pytest.raises(ValueError, match="complete selection"):
        plans.phase1_configuration(original, precision="float32", frame_data={})
    measured["source_to_frame_native"][0][2] = 2
    with pytest.raises(ValueError, match="native positioning"):
        plans.phase1_configuration(original, precision="float32", frame_data=frames)


def test_parent_area_array_hints_remain_exact_provenance(phase1_data):
    original = _original(phase1_data)
    frame = original["frames"][0]
    frame["source"]["studio_anchor"] = True
    frame["phase_roles"]["area"] = dict(role="measured_parent", factor=1.03,
        historical=dict(_bound("historical.npy"), dtype="float32", shape_hw=[2, 2]),
        measured=dict(_bound("measured.npy"), dtype="float32", shape_hw=[2, 2]))
    before = deepcopy(original)
    result = plans.phase1_configuration(original, precision="float32")
    area = result["frames"][0]["phase_roles"]["area"]
    assert area["historical"] == _bound("historical.npy")
    assert area["revision"]["original_array_metadata"]["historical"] == dict(dtype="float32", shape_hw=[2, 2])
    assert original == before


def test_missing_component_size_uses_root_admission_and_original_catalog_stat(phase1_data):
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    original = _original(phase1_data)
    bound = original["frames"][0]["phase_roles"]["static"]["components"]
    del bound["size_bytes"]
    seen = []
    def file_stat(path):
        seen.append(path)
        return SimpleNamespace(st_mode=stat.S_IFREG, st_size=47)
    catalog = InputCatalog({"data": "//unit-only/share"}, file_stat=file_stat)
    result = plans.phase1_configuration(original, precision="float32", role_catalog=catalog)
    actual = result["frames"][0]["phase_roles"]["static"]["components"]
    assert actual == dict(bound, size_bytes=47)
    assert len(seen) == 1
    reference = next(iter(catalog.consumers.values()))
    assert reference["root_alias"] == "data" and reference["relative_path"] == "static.npz"
    assert reference["sha256"] == bound["sha256"]
    with pytest.raises(ValueError, match="rooted input catalogue"):
        plans.phase1_configuration(original, precision="float32")


def test_parent_area_sizes_are_taken_from_their_own_catalog_entries(phase1_data):
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    original = _original(phase1_data)
    frame = original["frames"][0]
    frame["source"]["studio_anchor"] = True
    frame["phase_roles"]["area"] = dict(role="measured_parent", factor=1.03,
        historical=dict(path="//unit-only/share/historical.npy", sha256="a" * 64),
        measured=dict(path="//unit-only/share/measured.npy", sha256="b" * 64))
    def file_stat(path):
        return SimpleNamespace(st_mode=stat.S_IFREG, st_size=47 if "historical" in path else 53)
    catalog = InputCatalog({"data": "//unit-only/share"}, file_stat=file_stat)
    result = plans.phase1_configuration(original, precision="float32", role_catalog=catalog)
    area = result["frames"][0]["phase_roles"]["area"]
    assert area["historical"]["size_bytes"] == 47
    assert area["measured"]["size_bytes"] == 53


def test_legacy_array_alias_is_normalized_without_losing_hints_or_changing_identity(phase1_data):
    original = _original(phase1_data)
    original["data_aliases"] = [dict(logical=dict(path="//unit-only/share/old.npy",
        sha256="f" * 64, bytes=47, dtype="float32", shape=[2, 2]),
        physical=dict(path="//unit-only/share/new.npy", sha256="f" * 64, size_bytes=47))]
    before = deepcopy(original)
    result = plans.phase1_configuration(original, precision="float32")
    assert result["data_aliases"][0]["physical"] == before["data_aliases"][0]["physical"]
    assert result["data_aliases"][0]["logical"]["size_bytes"] == 47
    assert result["provenance"]["legacy_alias_array_metadata"][0]["shape"] == [2, 2]
    assert original == before
    original["data_aliases"][0]["logical"]["bytes"] = 48
    with pytest.raises(ValueError, match="byte identity"):
        plans.phase1_configuration(original, precision="float32")


@pytest.mark.parametrize("mutation", ["gpu", "publish"])
def test_unadmitted_resources_and_publish_paths_fail_before_any_plan_write(tmp_path, phase1_data, mutation):
    job, resources = _job(), _resources(phase1_data)
    if mutation == "gpu":
        resources[phase1_data["run"]]["backend"] = "cuda"
    else:
        job["publish"] = ["../outside"]
    target = tmp_path / "prepared"
    with pytest.raises(ValueError):
        plans.write_phase1_plan(target, [phase1_data], job_id="testonly", job=job, resources=resources)
    assert not target.exists()


def test_preparation_stays_inert_and_adoption_writes_a_separate_reviewed_file(tmp_path, phase1_data):
    result = plans.write_phase1_plan(tmp_path / "prepared", [phase1_data],
        job_id="testonly", job=_job(), resources=_resources(phase1_data))
    bound = result["plan"]
    assert not result["adopted_for_execution"] and not result["job_created"]
    with pytest.raises(ValueError, match="inert"):
        plans.read_phase1_plan(bound["path"], expected_sha256=bound["sha256"])
    value, configs, actual = plans.read_phase1_plan(bound["path"],
        expected_sha256=bound["sha256"], require_adopted=False)
    assert actual == bound and configs == [phase1_data]
    before = Path(bound["path"]).read_bytes()
    approved = plans.adopt_prepared_plan(bound["path"], tmp_path / "approved.toml",
        expected_sha256=bound["sha256"], actor="reviewer", reason="Reviewed the bound data")
    value, configs, _ = plans.read_phase1_plan(approved["plan"]["path"],
        expected_sha256=approved["plan"]["sha256"])
    assert value["adoption"]["prepared_sha256"] == bound["sha256"]
    assert not approved["job_created"] and not approved["worker_started"]
    assert Path(bound["path"]).read_bytes() == before
    assert not list(tmp_path.rglob("job.json"))


@pytest.mark.parametrize("change", ["selection", "configuration", "plan"])
def test_changed_reviewed_bytes_are_rejected_before_adoption(tmp_path, phase1_data, change):
    result = plans.write_phase1_plan(tmp_path / "prepared", [phase1_data],
        job_id="testonly", job=_job(), resources=_resources(phase1_data))
    bound = result["plan"]
    if change == "plan":
        path = Path(bound["path"])
    elif change == "selection":
        path = tmp_path / "prepared/selection.json"
    else:
        path = Path(result["runs"][0]["configuration"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    destination = tmp_path / "approved.toml"
    with pytest.raises(ValueError, match="changed|differs"):
        plans.adopt_prepared_plan(bound["path"], destination,
            expected_sha256=bound["sha256"], actor="reviewer", reason="Reviewed")
    assert not destination.exists()


def test_gas_probability_is_the_existing_explicit_native_request(phase1_data):
    result = plans.phase1_configuration(_original(phase1_data), precision="float32",
                                       gas_probability=True)
    from darsia.fluidflower.phase.probability_storage import parse_request
    assert parse_request(result) == result["gas_probability_storage"]
    assert result["storage"]["resolution_mm"] == "native"


def test_provisional_pressure_cannot_be_sealed_as_a_plan_before_its_decision(tmp_path, phase1_data):
    phase1_data["statistics"]["atmosphere_series"] = dict(provisional_gap_default=True)
    target = tmp_path / "prepared"
    with pytest.raises(ValueError, match="provisional pressure gaps"):
        plans.write_phase1_plan(target, [phase1_data], job_id="testonly", job=_job(), resources=_resources(phase1_data))
    assert not target.exists()


def test_phase2_requires_a_verified_completion_and_never_invents_one(tmp_path, phase1_data):
    closure = _closure(tmp_path / "synthetic", phase1_data)
    job = dict(_job(), calibration_sha256=closure["calibration"]["sha256"])
    source = dict(completion_bound=closure["completion"],
        configuration_bound=closure["configuration"], expected_phase1_commit="1" * 40,
        fitted_calibration=closure["calibration"],
        fitted_anchor=dict(elapsed_h=1., affected_pore_volume_m3=.001))
    reader = DataReader(unc_only=False)
    result = plans.write_phase2_plan(tmp_path / "prepared", [source], job_id="testonly",
        job=job, resources=_resources(phase1_data), reader=reader)
    config = reader.json(result["runs"][0]["configuration"])
    assert config["phase1"]["completion"] == closure["completion"]
    assert config["calibration"] == closure["calibration"]
    assert not result["adopted_for_execution"] and not result["worker_started"]
    value, configs, _ = plans.read_staged_plan(result["plan"]["path"],
        expected_sha256=result["plan"]["sha256"], require_adopted=False)
    assert value["pipeline"] == "n1" and configs == [config]
    Path(closure["completion"]["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError):
        plans.write_phase2_plan(tmp_path / "bad", [source], job_id="testonly",
            job=job, resources=_resources(phase1_data), reader=reader)
    assert not (tmp_path / "bad/execution.toml").exists()
