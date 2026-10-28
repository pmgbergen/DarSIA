"""GUI metadata routing tests; preparation doubles start no worker or claim."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import staged_jobs as jobs
from darsia.fluidflower.analysis_plan import write_phase1_plan, adopt_prepared_plan
from darsia.fluidflower.pipeline.job_queue import descriptor, write_immutable
from .test_analysis_plan import _job, _resources
from .test_staged_data import phase1_data


def adopted(tmp_path, data):
    prepared = write_phase1_plan(tmp_path / "prepared", [data], job_id="testonly",
                                 job=_job(), resources=_resources(data))["plan"]
    return adopt_prepared_plan(prepared["path"], tmp_path / "approved.toml",
        expected_sha256=prepared["sha256"], actor="reviewer", reason="Synthetic metadata test")["plan"]


def test_positive_phase1_uses_its_own_reader_without_relabelling_old_series(tmp_path, phase1_data, monkeypatch):
    bound = adopted(tmp_path, phase1_data)
    from darsia.fluidflower.pipeline import series
    monkeypatch.setattr(series, "read_series", lambda *a, **k: pytest.fail("No old series dispatch"))
    value, configs, actual = jobs.read_execution_plan(bound["path"], expected_sha256=bound["sha256"])
    assert value["pipeline"] == "native-phase-statistics"
    assert configs == [phase1_data] and actual == bound


def test_existing_reader_and_creator_keep_the_original_arguments(tmp_path, monkeypatch):
    path = tmp_path / "old.toml"
    path.write_text('format = "darsia-series-v1"\n', encoding="utf-8")
    bound = descriptor(path)
    from darsia.fluidflower.pipeline import series
    from darsia.fluidflower import jobs_metadata
    calls, marker = [], object()
    def read(*args, **kwargs):
        calls.append((args, kwargs))
        return marker
    monkeypatch.setattr(series, "read_series", read)
    assert jobs.read_execution_plan(path, expected_sha256=bound["sha256"]) is marker
    assert calls == [((path,), dict(expected_sha256=bound["sha256"]))]
    monkeypatch.setattr(jobs_metadata, "create_job_from_plan", read)
    target = tmp_path / "not-created"
    assert jobs.create_analysis_job(bound, target, stores_path="chosen-stores.toml",
        display_name="Name", series_label="Series", storage_profile="lean") is marker
    assert calls[-1] == ((bound, target), dict(display_name="Name", series_label="Series", storage_profile="lean"))
    assert not target.exists()


def test_uninstalled_phase1_route_reports_the_actual_blocker_without_input_io(monkeypatch):
    monkeypatch.setattr(jobs.importlib.util, "find_spec", lambda name: None)
    result = jobs.inspect_phase1_resources({}, series_root="not-opened")
    assert result["ready"] is False and "capacity admission is not installed" in result["reason"]


def test_staged_mass_cannot_create_before_its_original_profile_admission(tmp_path, monkeypatch):
    path = tmp_path / "mass.toml"
    path.write_text('format = "darsia-series-v1"\npipeline = "n1"\n', encoding="utf-8")
    from darsia.fluidflower import jobs_metadata
    monkeypatch.setattr(jobs_metadata, "create_job_from_plan", lambda *a, **k: pytest.fail("No creation before admission"))
    monkeypatch.setattr(jobs, "inspect_staged_resources", lambda *a, **k: dict(ready=False, reason="Phase-two profile is absent"))
    with pytest.raises(ValueError, match="Phase-two profile is absent"):
        jobs.create_analysis_job(descriptor(path), tmp_path / "job", storage_profile="full")
    assert not (tmp_path / "job").exists()


def test_phase1_lean_and_missing_resource_admission_create_no_job(tmp_path, phase1_data, monkeypatch):
    plan = adopted(tmp_path, phase1_data)
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('series_id = "testonly"\n', encoding="utf-8")
    target = tmp_path / "job"
    with pytest.raises(ValueError, match="requires Full"):
        jobs.create_analysis_job(plan, target, storage_profile="lean")
    from darsia.fluidflower import jobs_metadata
    monkeypatch.setattr(jobs_metadata, "validate_current_selection", lambda *a, **k: dict(runs=[]))
    monkeypatch.setattr(jobs, "inspect_phase1_resources", lambda *a, **k: dict(ready=False, reason="Profile commit differs"))
    with pytest.raises(ValueError, match="Profile commit differs"):
        jobs.create_analysis_job(plan, target, series_root=root, series_id="testonly", storage_profile="full")
    assert not target.exists()


def test_phase1_gui_create_calls_original_preparer_and_writes_the_existing_master_contract(tmp_path, phase1_data, monkeypatch):
    plan = adopted(tmp_path, phase1_data)
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('series_id = "testonly"\n', encoding="utf-8")
    target = tmp_path / "job"
    from darsia.fluidflower import jobs_metadata
    from darsia.fluidflower.pipeline import alias_job
    monkeypatch.setattr(jobs_metadata, "validate_current_selection", lambda *a, **k: dict(runs=[]))
    monkeypatch.setattr(jobs, "inspect_phase1_resources", lambda *a, **k: dict(ready=True))
    job = dict(_job(), format="darsia-alias-job-v1", pipeline="native-phase-statistics", expected_runs=[phase1_data["run"]])
    task = dict(pipeline="native-phase-statistics", run=phase1_data["run"])
    monkeypatch.setattr(jobs, "phase1_inputs", lambda *a, **k: (
        dict(job_id="testonly"), [phase1_data], job, [task], [], dict(series_id="testonly")))
    monkeypatch.setattr(alias_job, "portable_selection_binding", lambda value, **k: value)
    seen = []
    def prepare(actual_job, tasks, folder, **kwargs):
        seen.append((deepcopy(actual_job), deepcopy(tasks), kwargs))
        write_immutable(Path(folder) / "job.json", actual_job)
        return dict(complete=True, numerical_calls_started=False, series=dict(root_alias="data", relative_path="execution.toml", sha256=plan["sha256"], bytes=plan["size_bytes"], role="configuration"))
    monkeypatch.setattr(alias_job, "prepare_phase1_alias_job", prepare)
    result = jobs.create_analysis_job(plan, target, series_root=root, series_id="testonly",
                                      display_name="Example", series_label="Example series", storage_profile="full")
    assert seen[0][0] == job and seen[0][1] == [task]
    assert "calibration_sha256" not in seen[0][0] and "storage_profile" not in seen[0][0]
    assert seen[0][2]["execution_series"] == plan
    assert result["worker_started"] is False
    reader = jobs_metadata.JobSnapshotReader()
    master = reader.read(target / "ui/master.json", result["master"])
    assert master["format"] == "darsia-job-ui-master-v2" and master["numerical_calls_started"] is False
    assert (target / "ui/image_selection_binding.json").is_file()
