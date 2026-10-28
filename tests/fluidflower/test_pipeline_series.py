"""Tiny public-API DATA/ownership checks; no NAS or scientific fixture inputs."""

from copy import deepcopy
import hashlib
import json

import pytest

from darsia.fluidflower.pipeline import runner, series


def _bound(path, raw):
    return dict(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )


def _configuration():
    source = dict(
        run="run-one",
        frame_index=1,
        elapsed_h=1.0,
        source_sha256="a" * 64,
        previous_source_sha256="b" * 64,
        selected=True,
        studio_anchor=False,
        source_binding=dict(source_sha256="a" * 64),
    )
    frame = dict(
        case_id="case-one",
        frame_index=1,
        elapsed_h=1.0,
        source=source,
        native_task=dict(kind="rig"),
        geometry={},
        mass_spec=dict(selected_target_h=1.0),
        phase_roles=dict(area=dict(role="selected", factor=1.0)),
    )
    calibration = dict(
        path=r"\\server\share\calibration.json", sha256="c" * 64, size_bytes=1
    )
    return dict(
        format=series.RUN_FORMAT,
        run="run-one",
        frames=[frame],
        native={},
        phase=dict(initial_previous=dict(source=dict(source_sha256="b" * 64))),
        mass=dict(calibration=calibration, template=dict(calibration=calibration)),
        table_fields=["case_id", "total_mass_g"],
        history_policy="chronological acquisitions",
        selection_index_field="frame_index",
    )


def test_fixed_selection_keeps_input_document_unchanged():
    config = _configuration()
    original = deepcopy(config)
    assert series.validate_run(config) is config
    assert config == original


@pytest.mark.parametrize(
    "change",
    ["duplicate_case", "duplicate_source", "reverse_time", "wrong_predecessor"],
)
def test_selection_rejects_ambiguity_before_numerical_calls(change):
    config = _configuration()
    second = deepcopy(config["frames"][0])
    second.update(case_id="case-two", frame_index=2, elapsed_h=2.0)
    second["source"].update(
        frame_index=2,
        elapsed_h=2.0,
        source_sha256="d" * 64,
        previous_source_sha256="a" * 64,
    )
    if change == "duplicate_case":
        second["case_id"] = "case-one"
    elif change == "duplicate_source":
        second["source"]["source_sha256"] = "a" * 64
    elif change == "reverse_time":
        second["elapsed_h"] = second["source"]["elapsed_h"] = 0.5
    else:
        second["source"]["previous_source_sha256"] = "e" * 64
    config["frames"].append(second)
    with pytest.raises(ValueError):
        series.validate_run(config)


def test_expected_current_products_are_not_phase_inputs():
    config = _configuration()
    config["frames"][0]["mass_spec"]["labels"] = {"path": "expected.png"}
    with pytest.raises(ValueError, match="fresh stage outputs"):
        series.validate_run(config)
    config = _configuration()
    config["phase"]["expected_current_classes"] = "expected.npz"
    with pytest.raises(ValueError, match="DATA"):
        series.validate_run(config)


@pytest.mark.parametrize("missing", ["probability_encoding", "anchor_flag"])
def test_RAW_source_metadata_errors_fail_before_numerical_work(missing):
    config = _configuration()
    frame = config["frames"][0]
    frame["native_task"] = dict(kind="raw_rig", geometry={}, processor=dict(raw_binding={}))
    frame["phase_roles"]["encoding_contract"] = None
    if missing == "probability_encoding":
        frame["phase_roles"]["encoding_contract"] = {"historical": True}
    else:
        frame["source"]["studio_anchor"] = True
    with pytest.raises(ValueError, match="probability encoding|source-geometry flag"):
        series.validate_run(config)


def test_model_entrypoint_cannot_execute_series_python():
    config = _configuration()
    config["phase"]["entrypoint"] = "some_module:function"
    with pytest.raises(ValueError, match="DATA"):
        series.validate_run(config)


def test_anchor_area_requires_both_original_and_measured_roles():
    config = _configuration()
    config["frames"][0]["source"]["studio_anchor"] = True
    with pytest.raises(ValueError, match="area routes"):
        series.validate_run(config)
    config["frames"][0]["phase_roles"]["area"]["role"] = "measured_parent"
    with pytest.raises(ValueError, match="separate original and measured"):
        series.validate_run(config)


def test_legacy_n1_task_is_not_mutated_or_reinterpreted():
    task = dict(run="old-run", configuration={"path": "original.json"})
    original = deepcopy(task)
    assert series.pipeline_kind(task) == "n1"
    assert series.pipeline_kind(dict(task, pipeline="n1")) == "n1"
    assert task == original
    with pytest.raises(ValueError, match="pipeline"):
        series.pipeline_kind(dict(task, pipeline="unknown"))


def test_fresh_mass_binding_cannot_change_recipe_or_source():
    spec = _configuration()["frames"][0]
    spec["mass_spec"].update(
        atmosphere=dict(pressure_bar=1.02), nominal_injected_mass_g=2.5
    )
    original = deepcopy(spec)
    native = dict(source_sha256="a" * 64, full=dict(actual_raw=True))
    phase = dict(
        source_sha256="a" * 64,
        labels=dict(actual_labels=True),
        classes=dict(actual_classes=True),
        area=dict(actual_measured_area=True),
    )
    bound = series.bind_mass_frame(
        spec, native, phase, baseline=dict(separate_reference=True)
    )
    assert bound["nominal_injected_mass_g"] == 2.5
    assert bound["atmosphere"] == spec["mass_spec"]["atmosphere"]
    assert bound["current"] == native["full"]
    assert bound["labels"] == phase["labels"]
    assert bound["source_binding"] == spec["source"]["source_binding"]
    assert spec == original
    with pytest.raises(ValueError, match="another acquisition"):
        series.bind_mass_frame(
            spec, dict(native, source_sha256="f" * 64), phase, baseline={}
        )


def test_history_never_uses_predicted_labels_as_classifier_features():
    spec = _configuration()["frames"][0]
    history = dict(
        source=deepcopy(spec["source"]), current_h=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert series.restore_history(history, spec) is history
    with pytest.raises(ValueError, match="history features"):
        series.restore_history(dict(history, labels="predicted.png"), spec)
    wrong = deepcopy(history)
    wrong["source"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="selected acquisition"):
        series.restore_history(wrong, spec)


def _series(config):
    return dict(
        job_id="tiny-job",
        job=dict(
            commit="d" * 40,
            precision="float32",
            calibration_sha256="c" * 64,
            environment_lock=dict(path="requirements-worker.lock", sha256="e" * 64),
        ),
        runs=[
            dict(
                run=config["run"],
                configuration=dict(path=r"\\server\share\config.json", sha256="a" * 64),
                resources=dict(backend="cpu", threads=1),
            )
        ],
    )


def test_tasks_keep_static_whole_run_assignment_and_canonical_output():
    config = _configuration()
    value = _series(config)
    before = deepcopy(value)
    job, tasks = series.series_tasks(
        value, [config], job_root=r"\\server\share\job", series_asset={}
    )
    assert job["expected_runs"] == ["run-one"]
    assert len(tasks) == 1 and tasks[0]["pipeline"] == series.PIPELINE
    assert tasks[0]["configuration"] == value["runs"][0]["configuration"]
    assert value == before
    value["runs"][0]["resources"]["output_directory"] = r"\\server\share\outside"
    with pytest.raises(ValueError, match="canonical"):
        series.series_tasks(
            value, [config], job_root=r"\\server\share\job", series_asset={}
        )


def test_missing_run_or_escaped_environment_lock_rejected():
    config = _configuration()
    value = _series(config)
    with pytest.raises(ValueError, match="Every selected run"):
        series.series_tasks(value, [], job_root=r"\\server\share\job", series_asset={})
    value["job"]["environment_lock"]["path"] = "../other.lock"
    with pytest.raises(ValueError, match="repository-relative"):
        series.series_tasks(
            value, [config], job_root=r"\\server\share\job", series_asset={}
        )


def test_scalar_table_reads_genuine_ordered_json_without_npz(tmp_path):
    records = []
    for case, mass in (("first", 1.25), ("second", None)):
        directory = tmp_path / case
        directory.mkdir()
        raw = json.dumps(dict(complete=True, case_id=case, total_mass_g=mass)).encode()
        (directory / "result.json").write_bytes(raw)
        records.append(
            dict(
                frame_id=case,
                nas_directory=str(directory),
                files={"result.json": _bound(directory / "result.json", raw)},
            )
        )
    target = tmp_path / "table.csv"
    receipt = runner.scalar_table(records, ["case_id", "total_mass_g"], target)
    assert target.read_text().splitlines() == [
        "case_id,total_mass_g",
        "first,1.25",
        "second,",
    ]
    assert receipt["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        runner.scalar_table(records, ["case_id", "total_mass_g"], target)


def test_scalar_table_rejects_changed_committed_metadata(tmp_path):
    directory = tmp_path / "frame"
    directory.mkdir()
    expected = b'{"complete":true,"case_id":"first","total_mass_g":1}'
    (directory / "result.json").write_bytes(expected.replace(b"1}", b"2}"))
    records = [
        dict(
            frame_id="first",
            nas_directory=str(directory),
            files={"result.json": _bound(directory / "result.json", expected)},
        )
    ]
    with pytest.raises(ValueError, match="SHA"):
        runner.scalar_table(
            records, ["case_id", "total_mass_g"], tmp_path / "fresh.csv"
        )


def test_toml_content_guard_precedes_any_configuration_read(tmp_path):
    path = tmp_path / "series.toml"
    path.write_text('format = "darsia-series-v1"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="TOML bytes"):
        series.read_series(path, expected_sha256="0" * 64)


def test_camera_index_is_distinct_from_selected_work_index():
    config = _configuration()
    config["selection_index_field"] = "work_index"
    config["frames"][0]["source"].update(frame_index=28, work_index=1)
    assert series.validate_run(config) is config
    assert config["frames"][0]["source"]["frame_index"] == 28
