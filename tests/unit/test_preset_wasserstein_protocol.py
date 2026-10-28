"""Typed native Comparison queue controls; no scientific execution evidence."""
from copy import deepcopy

import pytest

from darsia.fluidflower import preset_wasserstein_protocol as protocol
from darsia.fluidflower.pipeline.job_queue import validate_task


def original_controls(*, runs=None, times=None):
    """Explicit no-match selections exercise original missing-result semantics."""
    source = dict(path="C:/original/comparison.toml", sha256="1" * 64, size_bytes=13)
    runs = ["B", "A", "C"] if runs is None else runs
    times = [dict(order=0, time_hours=2.0, tolerance_hours=0.05)] if times is None else times
    return dict(format=protocol.INPUTS_FORMAT, source_configuration=source,
        working_directory="C:/original", rig_folder="C:/original/rig", config_paths=[source],
        runs=runs, rois=["whole"], requested_times=times, reference_run=runs[0],
        files=[dict(descriptor=source, role="configuration", uses=["preset-input:0"])],
        selected_mass=[dict(time_order=time["order"], run=run, descriptor=None)
                       for time in times for run in runs], series_manifest=None,
        inventories=[dict(path="C:/original/mass", pattern=None, names=[])],
        existence=[dict(path="C:/original/series.toml", kind="is_file", exists=False)],
        config_roots=[], path_aliases=[])


def task_and_job():
    frozen = original_controls()
    time = frozen["requested_times"][0]
    run = protocol.operation_id(time["order"])
    reference = dict(root_alias="series", relative_path="operation.json",
                     role="configuration", sha256="2" * 64, bytes=11)
    common = dict(format_version=1, legacy_paths=False, storage_profile="full",
        execution_ready=True, runtime_support="darsia-role-io-runtime-v1",
        pipeline=protocol.PIPELINE, backend="cpu", precision="original", job_id="comparison",
        commit="3" * 40, preset_inputs_sha256=protocol.digest(frozen),
        series=deepcopy(reference), series_manifest=deepcopy(reference))
    task = dict(common, format="darsia-alias-task-v1", run=run, threads=1,
        configuration=reference, comparison_time=time, ordered_case_ids=[run], expected_comparisons=3,
        input_catalog=dict(format="darsia-role-input-catalog-v1", complete=True, run=run,
            configuration=reference, runtime_support="darsia-role-io-runtime-v1",
            inputs=[dict(reference=reference, uses=["run.configuration", "run.execution_series",
                                                   "run.series_manifest"])],
            payload_hashes_verified=False, numerical_calls_started=False))
    job = dict(common, format="darsia-alias-job-v1", expected_runs=[run],
               preset_comparison_times=[time], comparisons_per_time=3)
    config = dict(format=protocol.OPERATION_FORMAT, run=run,
                  preset_inputs=frozen, comparison_time=time)
    return deepcopy(task), deepcopy(job), deepcopy(config)


def test_original_nested_traversal_keeps_first_selected_reference():
    frozen = protocol.validate_inputs_contract(original_controls())
    assert frozen["reference_run"] == "B"
    rows = protocol.expected_comparisons(frozen, frozen["requested_times"][0])
    assert [(row["run_1"], row["run_2"]) for row in rows] == [("B", "C"), ("A", "B"), ("A", "C")]


def test_original_numeric_run_ids_are_not_string_sorted_or_coerced():
    frozen = protocol.validate_inputs_contract(original_controls(runs=[10, 2]))
    rows = protocol.expected_comparisons(frozen, frozen["requested_times"][0])
    assert frozen["reference_run"] == 10
    assert [(row["run_1"], row["run_2"]) for row in rows] == [(2, 10)]


def test_original_filename_rounding_collision_is_refused_before_job_creation():
    times = [dict(order=0, time_hours=1.0001, tolerance_hours=0.01),
             dict(order=1, time_hours=1.0002, tolerance_hours=0.01)]
    with pytest.raises(ValueError, match="collide"):
        protocol.validate_inputs_contract(original_controls(times=times))


@pytest.mark.parametrize("mutation", ["reference", "time_order", "selected_coverage", "selected_unbound",
                                     "duplicate_consumer", "escaping_roi"])
def test_changed_original_source_controls_refuse(mutation):
    value = original_controls()
    if mutation == "reference": value["reference_run"] = "A"
    elif mutation == "time_order": value["requested_times"][0]["order"] = 1
    elif mutation == "selected_coverage": value["selected_mass"].pop()
    elif mutation == "selected_unbound": value["selected_mass"][0]["descriptor"] = value["source_configuration"]
    elif mutation == "duplicate_consumer": value["files"][0]["uses"] *= 2
    else: value["rois"] = ["outside/roi"]
    with pytest.raises(ValueError):
        protocol.validate_inputs_contract(value)


def test_queue_accepts_only_typed_original_precision_without_fake_calibration():
    task, job, config = task_and_job()
    validate_task(task)
    protocol.validate_preset_wasserstein_claim(task, job, config=config)
    assert "calibration_sha256" not in task and "calibration_sha256" not in job


@pytest.mark.parametrize("field,value", [("backend", "cuda"), ("precision", "float32"),
    ("commit", "4" * 40), ("preset_inputs_sha256", "5" * 64),
    ("expected_comparisons", 1), ("threads", 2), ("calibration_sha256", "6" * 64)])
def test_original_task_job_identity_cannot_change(field, value):
    task, job, config = task_and_job()
    task[field] = value
    with pytest.raises(ValueError):
        protocol.validate_preset_wasserstein_claim(task, job, config=config)


def test_original_precision_exception_does_not_admit_other_pipelines():
    task, _job, _config = task_and_job()
    task.update(pipeline="native-phase-n1", calibration_sha256="7" * 64)
    with pytest.raises(ValueError, match="precision"):
        validate_task(task)


def test_consumed_configuration_must_match_frozen_controls():
    task, job, config = task_and_job()
    config["preset_inputs"]["reference_run"] = "A"
    with pytest.raises(ValueError):
        protocol.validate_preset_wasserstein_claim(task, job, config=config)
