"""Typed Setup metadata joins, with no actual IO, producer or adoption claim."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import job_queue  # Preload the reused pure name API.
from darsia.fluidflower.pipeline.job_inputs import catalog_order_key
from darsia.fluidflower.pipeline import series_setup_protocol as setup


def _descriptor(name, *, sha="a", size=17):
    return dict(path="C:/private/" + name, sha256=sha * 64, size_bytes=size)


def _reference(name, *, sha="a", size=17):
    return dict(root_alias="configuration", relative_path=name,
                sha256=sha * 64, bytes=size, role="configuration")


def _task(operation="build", *, run="run-a"):
    references = dict(configuration=_reference("operations/" + run + ".json"),
        series=_reference("execution-series.toml", sha="b"),
        series_manifest=_reference("series-manifest.json", sha="c"),
        setup_plan=_reference("setup-plan.json", sha="d"))
    uses = dict(configuration="run.configuration", series="run.execution_series",
                series_manifest="run.series_manifest", setup_plan="setup.plan")
    rows = [dict(reference=deepcopy(value), uses=[uses[key]]) for key, value in references.items()]
    rows.sort(key=lambda row: catalog_order_key(row["reference"]))
    return dict(format="darsia-alias-task-v1", format_version=1, legacy_paths=False,
        pipeline=setup.PIPELINE, commit="1" * 40, backend="cpu", precision="float32", target_machine="fixture",
        operation=operation, run=run, ordered_case_ids=[operation], **references,
        input_catalog=dict(format="darsia-role-input-catalog-v1", complete=True, run=run,
            configuration=deepcopy(references["configuration"]), inputs=rows,
            payload_hashes_verified=False, numerical_calls_started=False))


def _job(task):
    return dict(format="darsia-alias-job-v1", pipeline=setup.PIPELINE,
        commit=task["commit"], backend=task["backend"], precision=task["precision"],
        expected_runs=[task["run"]], setup_plan=deepcopy(task["setup_plan"]))


def _operation(operation="build", *, run="run-a"):
    value = dict(format=setup.FORMAT, run=run, operation=operation, target_machine="fixture",
        project_root="C:/private/project", project_manifest=_descriptor("project/series.json"),
        adopted_revision=_descriptor("project/revisions/selected/revision.json", sha="b"),
        series_manifest=_descriptor("series.toml", sha="c"),
        native_parameters=_descriptor("native-parameters.json", sha="e"),
        setup_plan=_descriptor("setup-plan.json", sha="d"),
        metadata_output=dict(maximum_output_bytes=4096, maximum_file_bytes=1024),
        output_scope=dict(metadata_roots=["C:/private/project", "C:/private/config"], scratch_roots=[]))
    if operation == "apply":
        value.update(runs=["run-a", "run-b"],
            build_plans=[dict(run_id="run-a", output_template={"declared": "first"}),
                         dict(run_id="run-b", output_template={"declared": "second"})])
    else:
        value.update(configuration_files=[_descriptor("config/common.toml", sha="f"),
                                          _descriptor("config/merged/run-a.toml", sha="b")],
            raw_build_request=_descriptor("actual-build-request.json", sha="c"),
            apply_completion=_descriptor("export/apply-complete.json", sha="f"),
            coordinator_completion=_descriptor("coordinator/completed.json", sha="e"))
    return value


@pytest.mark.parametrize("operation", ["apply", "build"])
def test_one_explicit_operation_joins_original_alias_scope_without_a_calibration(operation):
    task, config = _task(operation), _operation(operation)
    job = _job(task)
    before = deepcopy((task, job, config))
    assert setup.validate_setup_claim(task, job, config=config) is task
    assert setup.validate_setup_operation(config) is config
    assert (task, job, config) == before
    assert all("calibration_sha256" not in value for value in (task, job, config))


def test_an_operation_token_never_claims_an_analysis_frame_prefix():
    task = _task()
    task["ordered_case_ids"] = ["frame-0001"]
    with pytest.raises(ValueError, match="operation token"):
        setup.validate_setup_task(task)
    task["ordered_case_ids"] = ["build", "build"]
    with pytest.raises(ValueError, match="operation token"):
        setup.validate_setup_task(task)


@pytest.mark.parametrize("where", ["job", "task"])
def test_setup_requires_no_calibration_and_does_not_consume_unrelated_metadata(where):
    task = _task()
    value = task if where == "task" else _job(task)
    validate = setup.validate_setup_task if where == "task" else setup.validate_setup_job
    assert "calibration_sha256" not in value
    assert validate(value) is value
    # Retained caller metadata is opaque here, not a new fitted-model claim.
    value["calibration_sha256"] = "unrelated retained metadata"
    before = deepcopy(value)
    assert validate(value) is value
    assert value == before


def test_claim_refuses_another_literal_plan_even_when_its_bytes_are_equal():
    task = _task()
    job = _job(task)
    changed = dict(task["setup_plan"], relative_path="another-plan.json")
    old = task["setup_plan"]
    task["setup_plan"] = changed
    row = next(row for row in task["input_catalog"]["inputs"] if row["reference"] == old)
    row["reference"] = deepcopy(changed)
    task["input_catalog"]["inputs"].sort(key=lambda row: catalog_order_key(row["reference"]))
    assert setup.validate_setup_task(task) is task
    assert changed["sha256"] == job["setup_plan"]["sha256"]
    with pytest.raises(ValueError, match="immutable plan"):
        setup.validate_setup_claim(task, job)


def test_task_plan_must_be_part_of_the_original_verified_input_catalogue():
    task = _task()
    task["input_catalog"]["inputs"] = [row for row in task["input_catalog"]["inputs"]
        if row["reference"] != task["setup_plan"]]
    with pytest.raises(ValueError, match="catalogue must consume"):
        setup.validate_setup_task(task)


@pytest.mark.parametrize("part", ["commit", "backend", "precision", "run", "plan"])
def test_claim_identity_cannot_drift_from_the_selected_job(part):
    task = _task()
    job = _job(task)
    if part == "commit":
        job["commit"] = "2" * 40
    elif part == "backend":
        job["backend"] = "cuda"
    elif part == "precision":
        job["precision"] = "float64"
    elif part == "run":
        job["expected_runs"] = ["another-run"]
    else:
        job["setup_plan"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        setup.validate_setup_claim(task, job)


def test_job_refuses_casefold_duplicate_operation_run_names():
    job = _job(_task())
    job["expected_runs"] = ["run-a", "RUN-A"]
    with pytest.raises(ValueError, match="distinct"):
        setup.validate_setup_job(job)


@pytest.mark.parametrize("part", ["run", "operation", "plan_sha", "plan_size"])
def test_operation_document_cannot_replace_its_task_scope(part):
    task, config = _task(), _operation()
    if part == "run":
        config["run"] = "another-run"
    elif part == "operation":
        config = _operation("apply")
    elif part == "plan_sha":
        config["setup_plan"]["sha256"] = "f" * 64
    else:
        config["setup_plan"]["size_bytes"] += 1
    with pytest.raises(ValueError, match="another task|another immutable"):
        setup.validate_setup_operation(config, task=task)


@pytest.mark.parametrize("change", ["missing", "extra", "bool_size", "short_sha", "relative_path"])
def test_original_metadata_descriptors_remain_exact_without_reading_a_source(change):
    config = _operation()
    if change == "missing":
        config["adopted_revision"].pop("size_bytes")
    elif change == "extra":
        config["adopted_revision"]["approved"] = True
    elif change == "bool_size":
        config["adopted_revision"]["size_bytes"] = True
    elif change == "short_sha":
        config["adopted_revision"]["sha256"] = "abc"
    else:
        config["adopted_revision"]["path"] = "relative/revision.json"
    with pytest.raises(ValueError):
        setup.validate_setup_operation(config)


def test_project_manifest_cannot_be_borrowed_from_another_root():
    config = _operation()
    config["project_manifest"]["path"] = "C:/another-project/series.json"
    with pytest.raises(ValueError, match="another selected project"):
        setup.validate_setup_operation(config)


def test_native_parameter_json_cannot_be_the_run_configuration_toml():
    config = _operation()
    config["native_parameters"]["path"] = "c:\\PRIVATE\\CONFIG\\MERGED\\RUN-A.TOML"
    with pytest.raises(ValueError, match="native parameter JSON is distinct"):
        setup.validate_setup_operation(config)



def test_build_keeps_the_original_literal_configuration_order_and_repeats():
    config = _operation()
    config["configuration_files"].append(deepcopy(config["configuration_files"][0]))
    before = deepcopy(config["configuration_files"])
    assert setup.validate_setup_operation(config) is config
    assert config["configuration_files"] == before


def test_coordinator_completion_is_distinct_from_per_run_apply_receipt():
    config = _operation()
    config["coordinator_completion"] = deepcopy(config["apply_completion"])
    with pytest.raises(ValueError, match="coordinator completion is distinct"):
        setup.validate_setup_operation(config)


@pytest.mark.parametrize("change", ["duplicate_run", "foreign_plan", "reordered_plan", "future_request"])
def test_apply_keeps_exact_ordered_source_templates_without_future_raw_authority(change):
    config = _operation("apply")
    if change == "duplicate_run":
        config["runs"] = ["run-a", "run-a"]
    elif change == "foreign_plan":
        config["build_plans"][1]["run_id"] = "run-c"
    elif change == "reordered_plan":
        config["build_plans"].reverse()
    else:
        config["build_plans"][0]["raw_build_request"] = _descriptor("future-request.json")
    with pytest.raises(ValueError):
        setup.validate_setup_operation(config)


def test_apply_does_not_accept_an_actual_raw_request_before_postimages_exist():
    config = _operation("apply")
    config["raw_build_request"] = _descriptor("future-request.json")
    with pytest.raises(ValueError, match="metadata roles"):
        setup.validate_setup_operation(config)


@pytest.mark.parametrize("caps", [dict(maximum_output_bytes=True, maximum_file_bytes=1),
    dict(maximum_output_bytes=0, maximum_file_bytes=1),
    dict(maximum_output_bytes=1, maximum_file_bytes=2)])
def test_metadata_caps_are_literal_enforced_authorities_not_guessed_estimates(caps):
    config = _operation()
    config["metadata_output"] = caps
    with pytest.raises(ValueError):
        setup.validate_setup_operation(config)


def test_metadata_validation_never_opens_or_stats_any_declared_path(monkeypatch):
    task, config = _task(), _operation()
    job = _job(task)
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata validation must not touch a declared source")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    assert setup.validate_setup_claim(task, job, config=config) is task
