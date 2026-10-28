"""Metadata-only production profile routing; no scientific fixture execution.

The original rooted catalogue, profile-choice validator and common application
predicate are exercised. Series science admission and the mass consumer are
small metadata doubles; external provider closure and capacity are doubles too.
These tests do not claim qualification, worker execution or numerical coverage.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import jobs_metadata, production_resources as resources
from darsia.fluidflower import staged_jobs, store_map
from darsia.fluidflower.pipeline import alias_job, capacity, job_inputs
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia_cache import bootstrap, common_qualification as common, first_qualification, policy


def _put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return descriptor(path)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version=1\nseries_id="unit"\n[roots]\nseries="."\n', encoding="utf-8")
    plan_path = root / "execution.toml"
    plan_path.write_text('pipeline="n1"\n', encoding="utf-8")
    plan = descriptor(plan_path)
    calibration = _put(root / "calibration.json", dict(synthetic_metadata=True))
    logical_calibration = dict(calibration, path="C:/unavailable-source/calibration.json")
    config = dict(format="darsia-n1-run-v1", run="unit", precision="float32",
        calibration=logical_calibration, storage_profile=dict(profile="lean", colour_intermediates="keep"),
        frames=[dict(case_id="first"), dict(case_id="last")],
        data_aliases=[dict(logical=logical_calibration, physical=calibration)])
    physical = _put(root / "configuration.json", config)
    logical = dict(physical, path="C:/unavailable-source/configuration.json")
    value = dict(pipeline="n1", job_id="unit-job",
        job=dict(commit="a" * 40, precision="float32", calibration_sha256=calibration["sha256"]),
        runs=[dict(run="unit", configuration=logical, resources=dict(backend="cpu", threads=2))],
        data_aliases=[dict(logical=logical, physical=physical)])
    binding = _put(tmp_path / "provider.json", dict(format=common.PROVIDER))
    authority = _put(tmp_path / "authority.json", dict(synthetic_authority=True))
    store_pin = _put(tmp_path / "stores.json", dict(synthetic_store_map=True))
    stores = SimpleNamespace(source=dict(store_pin, locality={}), document={})
    profile = dict(format=common.JOB_PROFILE, purpose="jobs",
        scope=dict(commit="a" * 40, backend="cpu", precision="float32"),
        stage_routes=dict(phase2=dict(pipeline="n1")),
        compute_profiles=dict(phase2=dict(control=None)))
    state = SimpleNamespace(root=root, plan=plan, config=config, value=value, stores=stores,
        binding=binding, authority=authority, profile=profile, events=[], estimates=[],
        cache_binding=binding, worker_binding=binding, producer_calls=[])

    def rewrite():
        physical = _put(root / "configuration.json", config)
        logical = dict(physical, path="C:/unavailable-source/configuration.json")
        value["runs"][0]["configuration"] = logical
        value["data_aliases"] = [dict(logical=logical, physical=physical)]

    def read_series(path, *, expected_sha256, series_manifest):
        assert Path(path) == plan_path and expected_sha256 == plan["sha256"]
        assert series_manifest == descriptor(root / "series.toml")
        catalog = job_inputs.InputCatalog(dict(series=str(root)), value["data_aliases"])
        actual = catalog.doc(value["runs"][0]["configuration"], "configuration", "test.plan.configuration")
        return deepcopy(value), [actual], descriptor(plan_path)

    def mass_metadata(catalog, actual, frames, *, standalone=False):
        assert standalone and frames == config["frames"]
        catalog.doc(actual["calibration"], "calibration", "mass.calibration")

    def resolve(pin, kind, read, *, stores_source, read_authorities):
        state.events.append("provider")
        assert pin == state.binding and kind == "jobs" and stores_source == store_pin
        assert read(pin)["format"] == common.PROVIDER
        assert read_authorities(authority) == [{"synthetic": "authority"}]
        return deepcopy(state.profile)

    def read_authorities(pin, package_file):
        assert pin == authority and package_file == capacity.__file__
        return [{"synthetic": "authority"}]

    def estimate(actual_stores, series_id, task, actual, *, repository):
        state.events.append("capacity")
        assert actual_stores is stores and series_id == "unit"
        state.estimates.append((deepcopy(task), deepcopy(actual)))
        return SimpleNamespace(products_estimated_bytes=123)

    def producer(*args, **kwargs):
        state.producer_calls.append((args, kwargs))
        return {"synthetic_producer": True}

    monkeypatch.setattr(alias_job, "read_alias_series", read_series)
    monkeypatch.setattr(job_inputs.InputCatalog, "mass", mass_metadata)
    monkeypatch.setattr(store_map, "read_store_map", lambda *args, **kwargs: stores)
    monkeypatch.setattr(capacity, "settings", lambda *args: (tmp_path / "state", 40, state.binding))
    monkeypatch.setattr(policy, "cache_settings", lambda *args, **kwargs:
        SimpleNamespace(profiles=dict(jobs=state.cache_binding)))
    monkeypatch.setattr(bootstrap, "select_entry_role", lambda *args, **kwargs:
        dict(profile=state.worker_binding))
    monkeypatch.setattr(bootstrap, "_exact_authority_list", read_authorities)
    monkeypatch.setattr(first_qualification, "resolve_profile", resolve)
    monkeypatch.setattr(capacity, "load_estimate", estimate)
    monkeypatch.setattr(jobs_metadata, "create_job_from_plan", producer)
    state.rewrite = rewrite
    return state


@pytest.mark.parametrize("choice,expected,version", [(None, "full", 1), ("full", "full", 1), ("lean", "lean", 2)])
def test_original_profile_choice_and_rooted_task_reach_original_estimator(admitted, choice, expected, version):
    s = admitted
    s.config["storage_profile"]["profile"] = expected
    s.rewrite()
    before = deepcopy((s.value, s.config))
    result = resources.inspect_production_resources(s.plan, series_root=s.root, storage_profile=choice)
    assert result["ready"] and result["storage_profile"] == expected
    assert result["products_expected_bytes"] == 123
    assert s.events == (["provider", "capacity"] if expected == "lean" else ["capacity"])
    task, config = s.estimates[0]
    assert config == s.config and (s.value, s.config) == before
    assert task["format_version"] == version and task["storage_profile"] == expected
    assert task["pipeline"] == "n1" and task["ordered_case_ids"] == ["first", "last"]
    assert task["backend"] == "cpu" and task["threads"] == 2
    assert task["configuration"]["relative_path"] == "configuration.json"
    assert task["series"]["relative_path"] == "execution.toml"
    assert task["series_manifest"]["relative_path"] == "series.toml"
    assert task["input_catalog"]["runtime_support"] == "darsia-role-io-runtime-v1"
    assert any("mass.calibration" in row["uses"] for row in task["input_catalog"]["inputs"])


@pytest.mark.parametrize("problem", ["absent_setting", "full_setting", "drop", "cache_binding",
    "worker_binding", "old_provider", "direct_profile", "incomplete", "precision", "compute"])
def test_lean_refuses_before_capacity_or_producer(admitted, monkeypatch, problem):
    s = admitted
    if problem == "absent_setting":
        s.config.pop("storage_profile")
    elif problem == "full_setting":
        s.config["storage_profile"]["profile"] = "full"
    elif problem == "drop":
        s.config["storage_profile"]["colour_intermediates"] = "drop"
    elif problem == "cache_binding":
        s.cache_binding = dict(s.binding, sha256="b" * 64)
    elif problem == "worker_binding":
        s.worker_binding = dict(s.binding, sha256="b" * 64)
    elif problem in {"old_provider", "direct_profile"}:
        s.binding.update(_put(Path(s.binding["path"]), dict(format=
            common.JOB_PROFILE if problem == "direct_profile" else "darsia-expected-space-profile-v1")))
    elif problem == "incomplete":
        def incomplete(*args, **kwargs):
            raise ValueError("Actual common provider/source/stores are not completed")
        monkeypatch.setattr(first_qualification, "resolve_profile", incomplete)
    elif problem == "precision":
        s.profile["scope"]["precision"] = "float64"
    else:
        s.profile["compute_profiles"]["phase2"]["control"] = {"mode": "auto"}
    s.rewrite()
    target = s.root.parent / "never-created-job"
    with pytest.raises(ValueError, match="Production creation is blocked"):
        resources.create_profiled_analysis_job(s.plan, target, series_root=s.root,
            series_id="unit", storage_profile="lean", display_name="Name", series_label="Unit")
    assert not s.estimates and not s.producer_calls and not target.exists()


@pytest.mark.parametrize("changed", ["plan", "manifest", "configuration", "stores", "provider"])
def test_changed_source_after_estimate_cannot_report_ready(admitted, monkeypatch, changed):
    s = admitted
    paths = dict(plan=s.plan["path"], manifest=s.root / "series.toml",
        configuration=s.root / "configuration.json", stores=s.stores.source["path"], provider=s.binding["path"])
    original = capacity.load_estimate

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = Path(paths[changed])
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(capacity, "load_estimate", mutate)
    result = resources.inspect_production_resources(s.plan, series_root=s.root, storage_profile="lean")
    assert not result["ready"] and "changed" in result["reason"]
    assert not s.producer_calls


@pytest.mark.parametrize("choice", ["full", "lean"])
def test_creation_passes_exact_profile_to_original_jobs_producer(admitted, choice):
    s = admitted
    s.config["storage_profile"]["profile"] = choice
    s.rewrite()
    resolver = object()
    kwargs = dict(series_root=s.root, series_id="unit", storage_profile=choice,
        display_name="Name", series_label="Unit", selection_provider=resolver)
    target = s.root.parent / "job"
    assert resources.create_profiled_analysis_job(s.plan, target, stores_path="chosen-map", **kwargs) == {
        "synthetic_producer": True}
    assert s.producer_calls == [((s.plan, target), kwargs)]
    assert not target.exists()


def test_full_profile_qualification_failure_remains_a_creation_blocker(admitted, monkeypatch):
    s = admitted
    s.config["storage_profile"]["profile"] = "full"
    s.rewrite()

    def refused(*args, **kwargs):
        raise ValueError("Original Full profile commit differs")

    monkeypatch.setattr(capacity, "load_estimate", refused)
    with pytest.raises(ValueError, match="Original Full profile commit differs"):
        resources.create_profiled_analysis_job(s.plan, s.root.parent / "job", series_root=s.root,
            series_id="unit", storage_profile="full")
    assert not s.events and not s.producer_calls


@pytest.mark.parametrize("pipeline", ["native-phase-statistics", "native-phase-n1"])
def test_other_routes_delegate_without_relabelling(tmp_path, monkeypatch, pipeline):
    path = tmp_path / "execution.toml"
    path.write_text('pipeline="' + pipeline + '"\n', encoding="utf-8")
    plan = descriptor(path)
    seen, marker = [], object()
    monkeypatch.setattr(staged_jobs, "inspect_staged_resources", lambda *args, **kwargs:
        (seen.append((args, kwargs)), marker)[1])
    assert resources.inspect_production_resources(plan, series_root=tmp_path, stores_path="map") is marker
    assert seen == [((plan,), dict(series_root=tmp_path, stores_path="map"))]
    monkeypatch.setattr(staged_jobs, "create_analysis_job", lambda *args, **kwargs:
        (seen.append((args, kwargs)), marker)[1])
    kwargs = dict(series_root=tmp_path, storage_profile="full")
    assert resources.create_profiled_analysis_job(plan, tmp_path / "job", stores_path="map", **kwargs) is marker
    assert seen[-1] == ((plan, tmp_path / "job"), dict(stores_path="map", **kwargs))
    if pipeline == "native-phase-statistics":
        result = resources.inspect_production_resources(plan, series_root=tmp_path, storage_profile="lean")
        assert not result["ready"] and "requires Full" in result["reason"]
        assert len(seen) == 2
