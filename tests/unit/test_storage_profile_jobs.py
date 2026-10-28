"""Per-job storage profile choice (N55): job schema, estimator, writer, capacity, Jobs panel.

Synthetic metadata and the tiny synthetic N1 frames only; no NAS, worker or
real capacity profile. 'full' must keep today's behaviour exactly; 'lean' is
chosen explicitly per job and admitted with its own estimate.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import storage_profile as storage
from darsia.fluidflower.jobs_metadata import JobSnapshotReader, create_job_from_plan
from darsia.fluidflower.mass import n1_pipeline as n1
from darsia.fluidflower.pipeline import capacity as cap
from darsia.fluidflower.pipeline import runner
from darsia.fluidflower.pipeline.job_queue import validate_task
from darsia.fluidflower.pipeline.store_runtime import RunUnavailable

from .gui.test_alias_job_metadata import assembly_fixture
from .gui.test_capacity import (claim_fixture, ledger_fixture, measured, native_metadata, owner,
                                settings_doc)
from .test_lean_storage import CODE, _bind, _synthetic

MB = 1_000_000
SERIES, PHASE_TWO = storage.SERIES_ROUTE, storage.PHASE_TWO_ROUTE
AC52_CANVAS = list(storage.AC52_CANVAS_HW)


def _canvas(frame, shape=AC52_CANVAS):
    """Declare a frame canvas the way both N1 routes do (metadata only)."""
    frame.setdefault("current", {})["phase_shape_hw"] = list(shape)
    frame.setdefault("mass_spec", {}).setdefault("photo_top", {})["shape_hw"] = list(shape)
    return frame


def _prepare(tmp_path, monkeypatch, profile=None, *, change=None, name="shared_job"):
    _root, manifest, source, config, _value = assembly_fixture(tmp_path, monkeypatch)
    for frame in config["frames"]:
        _canvas(frame)
    if change is not None:
        change(config)
    destination = tmp_path / name
    kwargs = {} if profile is None else {"storage_profile": profile}
    result = runner.prepare_job(source["path"], destination, expected_sha256=source["sha256"],
                                series_manifest=manifest, **kwargs)
    return destination, result, (manifest, source)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _rewrite(path, value):
    Path(path).write_bytes(json.dumps(value).encode())


# ------------------------------------------------------------ job schema


@pytest.mark.parametrize("choice,expected,version", [(None, "full", 1), ("full", "full", 1),
                                                     ("lean", "lean", 2)])
def test_prepare_job_writes_the_explicit_choice_in_job_and_every_task(tmp_path, monkeypatch, choice,
                                                                      expected, version):
    destination, result, _ = _prepare(tmp_path, monkeypatch, choice)
    job = _read(destination / "job.json")
    documents = [job, _read(destination / "results/ac52/task.json"), _read(destination / "queue/ac52.json")]
    for document in documents:
        assert document["storage_profile"] == expected and document["format_version"] == version
        assert storage.job_storage_profile(document) == expected
    validate_task(documents[1])
    assert result["storage_profile"] == expected
    estimate = result["storage_estimate"]
    assert estimate["profile"] == expected and estimate["complete"] and estimate["estimate"] is True
    assert "ESTIMATE" in estimate["label"] and "not a measurement" in estimate["label"]
    row = estimate["runs"][0]
    assert row["run"] == "ac52" and row["frames"] == 1 and row["pipeline"] == SERIES
    assert row["per_frame_bytes"] == (842_600_000 if expected == "full" else 522_500_000)
    assert row["per_run_bytes"] == (0 if expected == "full" else 218 * MB)
    assert estimate["job_bytes"] == row["job_bytes"]
    for text in (json.dumps(job), json.dumps(documents[1])):
        assert "storage_estimate" not in text and "ESTIMATE" not in text  # Never job DATA.
    snapshot = JobSnapshotReader()({"root": str(destination)})
    assert snapshot["job_facts"]["storage_profile"] == expected


def test_default_constant_chooses_new_jobs_only(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DEFAULT_STORAGE_PROFILE", "lean")
    destination, result, _ = _prepare(tmp_path, monkeypatch)
    assert _read(destination / "job.json")["storage_profile"] == "lean" == result["storage_profile"]
    assert storage.job_storage_profile({"run": "legacy"}) == "full"


@pytest.mark.parametrize("bad", ["small", "", "LEAN", 1, True])
def test_an_unknown_choice_is_refused_before_any_job_folder(tmp_path, monkeypatch, bad):
    with pytest.raises(ValueError, match="storage_profile"):
        _prepare(tmp_path, monkeypatch, bad)
    assert not (tmp_path / "shared_job").exists()


@pytest.mark.parametrize("entry,job,refused", [
    ({"profile": "lean"}, "full", True), ({"profile": "full"}, "lean", True),
    ({"profile": "lean", "colour_intermediates": "drop"}, "lean", False),
    ({"profile": "full", "store_colour_coordinate": True}, "full", False)])
def test_a_run_configuration_entry_must_agree_with_the_job_choice(tmp_path, monkeypatch, entry, job,
                                                                    refused):
    change = lambda config: config.update(storage_profile=deepcopy(entry))
    if refused:
        with pytest.raises(ValueError, match="its job chose"):
            _prepare(tmp_path, monkeypatch, job, change=change)
        assert not (tmp_path / "shared_job").exists()
    else:
        destination, result, _ = _prepare(tmp_path, monkeypatch, job, change=change)
        assert _read(destination / "job.json")["storage_profile"] == job
        assert result["storage_estimate"]["runs"][0]["colour_intermediates"] == entry.get(
            "colour_intermediates", "keep")


def test_a_legacy_job_without_the_field_is_full_and_never_reinterpreted(tmp_path, monkeypatch):
    destination, _result, (manifest, source) = _prepare(tmp_path, monkeypatch)
    # Simulate job metadata written before the field existed (format_version 1).
    for path in (destination / "job.json", destination / "results/ac52/task.json",
                 destination / "queue/ac52.json"):
        value = _read(path)
        del value["storage_profile"]
        _rewrite(path, value)
        assert storage.job_storage_profile(value) == "full"
    legacy = {path.name: path.read_bytes() for path in destination.rglob("*.json")}
    validate_task(_read(destination / "results/ac52/task.json"))
    assert JobSnapshotReader()({"root": str(destination)})["job_facts"].get("storage_profile") is None
    again = runner.prepare_job(source["path"], destination, expected_sha256=source["sha256"],
                               series_manifest=manifest)
    assert [row["already_enqueued"] for row in again["tasks"]] == [True]
    assert {path.name: path.read_bytes() for path in destination.rglob("*.json")} == legacy
    with pytest.raises(ValueError, match="differs"):
        runner.prepare_job(source["path"], destination, expected_sha256=source["sha256"],
                           series_manifest=manifest, storage_profile="lean")


@pytest.mark.parametrize("document", [
    {"storage_profile": "small"}, {"storage_profile": None}, {"storage_profile": 2},
    {"format": "darsia-alias-job-v1", "format_version": 1, "storage_profile": "lean"},
    {"format": "darsia-alias-job-v1", "format_version": 2, "storage_profile": "full"},
    {"format": "darsia-alias-job-v1", "format_version": 2},
    {"format": "darsia-alias-task-v1", "format_version": 1, "storage_profile": "lean"},
    {"format": "darsia-alias-job-v1", "format_version": 1, "pipeline": "native-phase-statistics",
     "storage_profile": "full"}])
def test_invalid_values_and_versions_are_refused_on_read(document):
    with pytest.raises(ValueError):
        storage.job_storage_profile(document)


def test_readers_refuse_a_lean_job_or_task_declared_as_version_1(tmp_path, monkeypatch):
    from darsia.fluidflower import job_control
    destination, _result, _ = _prepare(tmp_path, monkeypatch, "lean")
    task = _read(destination / "results/ac52/task.json")
    validate_task(task)
    with pytest.raises(ValueError, match="version"):
        validate_task(dict(task, format_version=1))
    job = _read(destination / "job.json")
    _rewrite(destination / "job.json", dict(job, format_version=1))
    with pytest.raises(ValueError, match="Unsupported alias-job metadata version"):
        JobSnapshotReader()({"root": str(destination)})
    with pytest.raises(ValueError, match="Unknown alias job version"):
        job_control._bindings(destination)
    _rewrite(destination / "job.json", dict(job, storage_profile="full"))
    with pytest.raises(ValueError, match="Unsupported alias-job metadata version"):
        JobSnapshotReader()({"root": str(destination)})


def test_a_claimed_task_must_carry_its_jobs_choice():
    lean_job = {"format": "darsia-alias-job-v1", "format_version": 2, "storage_profile": "lean"}
    lean_task = {"format": "darsia-alias-task-v1", "format_version": 2, "storage_profile": "lean"}
    storage.require_task_matches_job(lean_task, lean_job)
    storage.require_task_matches_job({"run": "old"}, {"expected_runs": ["old"]})
    with pytest.raises(ValueError, match="differs from job.json"):
        storage.require_task_matches_job({"run": "old"}, lean_job)
    with pytest.raises(ValueError, match="differs from job.json"):
        storage.require_task_matches_job(lean_task, {"storage_profile": "full"})


def test_the_worker_fails_a_claim_whose_profile_differs_from_job_json(monkeypatch, tmp_path):
    """Worker wiring: the mismatch is a FAIL before input admission or science."""
    from contextlib import contextmanager
    import queue
    from darsia.fluidflower.pipeline import worker, worker_io
    job = dict(format="darsia-alias-job-v1", format_version=2, storage_profile="lean",
               precision="float32", calibration_sha256="b" * 64)
    task = dict(format="darsia-alias-task-v1", format_version=1, run="ac52", attempt_id="1" * 32,
                commit="a" * 40, calibration_sha256="b" * 64, precision="float32")
    claim = SimpleNamespace(task=task, token=task["attempt_id"], output=tmp_path / "not-created")
    failures = []

    class Shared:
        def __init__(self, *a, **k):
            pass

        def recover_publications(self):
            return []

        def reclaim_stale(self):
            return []

        def claim(self, *a, **k):
            return claim

        def fail(self, actual, text, owner):
            failures.append(text)

    class Runtime:
        def __init__(self, *a):
            self.publications = SimpleNamespace(publish_progress=lambda *a: None)

        def availability(self, *a):
            pass

        @contextmanager
        def capacity_claim(self, *a):
            yield SimpleNamespace(availability=self.availability, bind=lambda claim: None)

        def output_factory(self, *a):
            pass

        def prepare(self, *a):
            pytest.fail("A mismatched claim must not reach input admission")

        def capacity_close(self, *a):
            pass

        def release(self, *a):
            pass

    from darsia_cache import adapters
    monkeypatch.setattr(worker, "JobQueue", Shared)
    monkeypatch.setattr(worker_io, "WorkerRunIO", Runtime)
    monkeypatch.setattr(worker, "_identity", lambda machine: dict(machine=machine))
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", lambda owner, **kwargs: None)
    monkeypatch.setattr(worker, "_job_control_boundary", lambda root: None)
    messages = queue.Queue()
    with pytest.raises(SystemExit):
        worker._run_tasks(str(tmp_path), job, "a" * 40, "main-cpu", "cpu", messages,
                          SimpleNamespace(is_set=lambda: False), once=True, exit_when_idle=False,
                          stores=SimpleNamespace(machine_id="Main"))
    assert len(failures) == 1 and "Claimed task storage_profile differs from job.json" in failures[0]
    assert not claim.output.exists()


# -------------------------------------------------------------- estimator


def test_estimator_returns_the_documented_ac52_numbers():
    full = storage.estimate_job_bytes("full", 39)
    assert full["estimate"] is True and full["label"] == storage.ESTIMATE_LABEL
    assert "ESTIMATE" in full["label"] and "not a measurement" in full["label"]
    assert full["per_frame_bytes"] == 842_600_000 and full["per_run_bytes"] == 0
    assert full["job_bytes"] == 39 * 842_600_000  # 32.9 GB, as measured on AC52.
    assert full["files"] == storage.AC52_FRAME_BYTES and full["not_retained"] == {}
    assert full["files"]["native_log.npy"] == full["files"]["temporal_log.npy"] == 196 * MB
    assert full["files"]["native_geometry.npz"] == 307 * MB
    assert full["files"]["phase_area.npz"] == 65_600_000 and full["files"]["products.npz"] == 40 * MB
    assert full["inflight_bytes"] == 842_600_000
    assert full["needed_bytes"] == full["job_bytes"] + full["inflight_bytes"]
    keep = storage.estimate_job_bytes("lean", 39)
    assert keep["per_frame_bytes"] == 522_500_000 and keep["per_run_bytes"] == 218 * MB
    assert keep["not_retained"] == {"native_geometry.npz": 307 * MB,
                                    "mass_total in products.npz": 13_100_000}
    assert keep["files"]["products.npz"] == 26_900_000 and "native_geometry.npz" not in keep["files"]
    drop = storage.estimate_job_bytes("lean", 39, colour_intermediates="drop")
    assert drop["per_frame_bytes"] == 130_500_000
    assert drop["inflight_bytes"] == 130_500_000 + 2 * 392 * MB
    two_full = storage.estimate_job_bytes("full", 39, pipeline=PHASE_TWO)
    two_lean = storage.estimate_job_bytes("lean", 39, pipeline=PHASE_TWO)
    assert two_full["per_frame_bytes"] == 352 * MB and set(two_full["files"]) == set(storage.PHASE_TWO_FILES)
    assert two_lean["per_frame_bytes"] == 31_900_000 and two_lean["per_run_bytes"] == 218 * MB
    assert full["source"] == storage.AC52_SOURCE and "843 MB per frame" in full["source"]


@pytest.mark.parametrize("frames", [1, 39, 3090])
@pytest.mark.parametrize("route,colours", [(SERIES, ("keep", "drop")), (PHASE_TWO, ("keep",))])
def test_the_lean_estimate_is_smaller(frames, route, colours):
    full = storage.estimate_job_bytes("full", frames, pipeline=route)
    for colour in colours:
        lean = storage.estimate_job_bytes("lean", frames, pipeline=route, colour_intermediates=colour)
        assert lean["per_frame_bytes"] < full["per_frame_bytes"]
        assert lean["needed_bytes"] < full["needed_bytes"]


def test_the_estimator_takes_explicit_per_file_sizes_and_refuses_invalid_input():
    sizes = {name: 10 for name in storage.AC52_FRAME_BYTES}
    value = storage.estimate_job_bytes("lean", 2, runs=3, frame_bytes=sizes, mass_total_bytes=4,
                                       run_store_bytes=100, extra_frame_bytes=1, extra_run_bytes=7,
                                       source="TESTONLY sizes")
    assert value["per_frame_bytes"] == 10 * 5 + 6 + 1 and value["per_run_bytes"] == 107
    assert value["job_bytes"] == 2 * 57 + 3 * 107 and value["source"] == "TESTONLY sizes"
    assert value["parameters"]["frame_bytes"] == sizes
    bad = [dict(profile="small"), dict(frames=0), dict(frames=True), dict(runs=0),
           dict(pipeline="native-phase-statistics"), dict(colour_intermediates="delete"),
           dict(profile="full", colour_intermediates="drop"),
           dict(pipeline=PHASE_TWO, colour_intermediates="drop"),
           dict(frame_bytes={"products.npz": 1}), dict(frame_bytes=dict(sizes, extra=1)),
           dict(frame_bytes=dict(sizes, **{"products.npz": -1})),
           dict(frame_bytes=dict(sizes, **{"products.npz": 1.5})),
           dict(mass_total_bytes=41 * MB), dict(run_store_bytes=-1), dict(extra_frame_bytes=True)]
    for change in bad:
        arguments = dict(profile="lean", frames=1)
        arguments.update(change)
        profile, frames = arguments.pop("profile"), arguments.pop("frames")
        with pytest.raises(ValueError):
            storage.estimate_job_bytes(profile, frames, **arguments)


def _config(frames=2, shape=AC52_CANVAS, **extra):
    return dict(run="ac52", frames=[_canvas(dict(case_id="ac52_f%d" % i), shape)
                                    for i in range(1, frames + 1)], **extra)


def test_run_estimate_reads_the_runs_own_configuration():
    value = storage.run_estimate("lean", _config(storage_profile={"profile": "lean",
                                                                  "colour_intermediates": "drop"},
                                                 mass={"template": {}}), pipeline=SERIES)
    assert value["colour_intermediates"] == "drop" and value["frames"] == 2
    coarse = storage.run_estimate("lean", _config(storage={"resolution_mm": 1.0}), pipeline=PHASE_TWO)
    plain = storage.run_estimate("lean", _config(), pipeline=PHASE_TWO)
    assert coarse["per_frame_bytes"] - plain["per_frame_bytes"] == 1_850_000
    assert coarse["per_run_bytes"] - plain["per_run_bytes"] == 4_700_000
    half = storage.run_estimate("full", _config(storage={"resolution_mm": 0.5}), pipeline=PHASE_TWO)
    assert half["parameters"]["extra_frame_bytes"] == 4 * 1_850_000
    native = storage.run_estimate("full", _config(storage={"resolution_mm": "native"}), pipeline=PHASE_TWO)
    assert native["parameters"]["extra_frame_bytes"] == 0
    colour = storage.run_estimate("full", _config(storage_profile={"profile": "full",
                                                                   "store_colour_coordinate": True}),
                                  pipeline=PHASE_TWO)
    assert colour["parameters"]["extra_frame_bytes"] == storage.AC52_COLOUR_COORDINATE_BYTES
    for config, route, match in (
            (_config(storage={"resolution_mm": 1.0}), SERIES, "no documented size"),
            (_config(gas_probability_storage={}), PHASE_TWO, "no documented size"),
            (_config(storage_profile={"profile": "full"}), PHASE_TWO, "its job chose"),
            (_config(mass={"template": {"storage_profile": {"profile": "full"}}}), SERIES, "mass template"),
            (dict(run="ac52", frames=[]), PHASE_TWO, "nonempty frame list")):
        with pytest.raises(ValueError, match=match):
            storage.run_estimate("lean", config, pipeline=route)


@pytest.mark.parametrize("route", [SERIES, PHASE_TWO])
def test_a_larger_declared_canvas_scales_the_documented_sizes_up_never_down(route):
    same = storage.run_estimate("lean", _config(), pipeline=route)
    assert same["ac52_size_scale"] == 1.0 and same["declared_canvas_pixels"] == 3572 * 6304
    smaller = storage.run_estimate("lean", _config(shape=[100, 200]), pipeline=route)
    assert smaller["per_frame_bytes"] == same["per_frame_bytes"] and smaller["ac52_size_scale"] == 1.0
    double = storage.run_estimate("lean", _config(shape=[2 * 3572, 6304]), pipeline=route)
    assert double["ac52_size_scale"] == 2.0 and "scaled by 2.0000" in double["source"]
    assert double["per_frame_bytes"] == 2 * same["per_frame_bytes"]
    assert double["per_run_bytes"] == 2 * same["per_run_bytes"]
    explicit = storage.run_estimate("lean", _config(shape=[2 * 3572, 6304]), pipeline=route,
                                    frame_bytes=dict(storage.AC52_FRAME_BYTES))
    assert explicit["ac52_size_scale"] is None and explicit["per_frame_bytes"] == same["per_frame_bytes"]
    mixed = _config()
    mixed["frames"][1] = _canvas(dict(case_id="ac52_f2"), [3 * 3572, 6304])
    assert storage.run_estimate("full", mixed, pipeline=route)["ac52_size_scale"] == 3.0
    undeclared = _config()
    del undeclared["frames"][0]["current"], undeclared["frames"][0]["mass_spec"]
    with pytest.raises(ValueError, match="declares no frame canvas"):
        storage.run_estimate("lean", undeclared, pipeline=route)


def test_prepare_reports_an_unavailable_run_estimate_without_inventing_zero():
    tasks = [dict(run="ac52", pipeline=PHASE_TWO), dict(run="ac53", pipeline=PHASE_TWO)]
    report = storage.estimate_prepared_job("lean", tasks, [_config(), _config(gas_probability_storage={})])
    assert report["complete"] is False and report["job_bytes"] is None and report["needed_bytes"] is None
    assert report["runs"][0]["available"] is True and report["runs"][1]["available"] is False
    assert "no documented size" in report["runs"][1]["reason"]
    whole = storage.estimate_prepared_job("lean", tasks, [_config(), _config(3)])
    assert whole["complete"] and whole["frames"] == 5
    assert whole["job_bytes"] == 5 * 31_900_000 + 2 * 218 * MB
    assert whole["needed_bytes"] == whole["job_bytes"] + 2 * 31_900_000


# --------------------------------------------------------- writer selection


@pytest.mark.parametrize("task,entry,expected", [
    ({"storage_profile": "lean"}, None, dict(profile="lean", colour_intermediates="keep")),
    ({"storage_profile": "lean"}, {"profile": "lean", "colour_intermediates": "drop"},
     dict(profile="lean", colour_intermediates="drop")),
    ({"storage_profile": "full"}, None, dict(profile="full", colour_intermediates="keep")),
    ({}, None, dict(profile="full", colour_intermediates="keep")),
    ({}, {"profile": "full"}, dict(profile="full", colour_intermediates="keep"))])
def test_the_run_setting_follows_the_jobs_choice(task, entry, expected):
    config = {} if entry is None else {"storage_profile": entry}
    assert storage.run_setting(task, config) == expected


@pytest.mark.parametrize("task,entry,match", [
    ({}, {"profile": "lean"}, "legacy full task"), ({"storage_profile": "full"}, {"profile": "lean"}, "its job chose"),
    ({"storage_profile": "lean"}, {"profile": "full"}, "its job chose"),
    ({"storage_profile": "small"}, None, "Unknown job storage_profile")])
def test_a_conflicting_or_unknown_choice_is_refused(task, entry, match):
    config = {} if entry is None else {"storage_profile": entry}
    with pytest.raises(ValueError, match=match):
        storage.run_setting(task, config)


def _run_n1(tmp_path, name, config, **task):
    bound = _bind(tmp_path / (name + "_config.json"), config)
    return n1.run_n1(dict(run=config["run"], configuration=bound, commit=CODE, **task),
                     tmp_path / name, "cpu", "float32")


def test_the_frame_writer_follows_the_tasks_choice(tmp_path):
    config = _synthetic(tmp_path)
    lean = _run_n1(tmp_path, "lean_job", config, storage_profile="lean")
    assert lean["lean_storage"]["profile"] == "lean"
    assert all("native_geometry.npz" not in record["files"] for record in lean["records"])
    for chosen, name in (({"storage_profile": "full"}, "full_job"), ({}, "legacy_job")):
        full = _run_n1(tmp_path, name, config, **chosen)
        assert "lean_storage" not in full
        assert all("native_geometry.npz" in record["files"] for record in full["records"])
    asking = dict(deepcopy(config), storage_profile={"profile": "lean"})
    with pytest.raises(ValueError, match="legacy full task"):
        _run_n1(tmp_path, "legacy_asking_lean", asking)
    assert not (tmp_path / "legacy_asking_lean").exists()  # Refused before any output.


def test_an_explicit_full_task_writes_byte_for_byte_what_the_base_writer_writes(tmp_path, monkeypatch):
    """The explicit field changes no output byte of the full layout (base 3ee2e7b3 writer)."""
    from .test_lean_storage import _Clock, _base_writer, _run, _tree
    config = _synthetic(tmp_path)
    base = _base_writer()
    clock = _Clock(monkeypatch, (n1, base))
    clock.reset()
    expected = _run(tmp_path, "out", config, module=base, job_id="job")
    (tmp_path / "out").rename(tmp_path / "base_out")
    clock.reset()
    actual = _run(tmp_path, "out", config, job_id="job", storage_profile="full")
    old, new = _tree(tmp_path / "base_out"), _tree(tmp_path / "out")
    assert sorted(old) == sorted(new) and [name for name in old if old[name] != new[name]] == []
    assert json.dumps(expected, sort_keys=True, default=str) == json.dumps(actual, sort_keys=True, default=str)


def test_the_series_runner_follows_the_tasks_choice(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import series
    config = dict(storage_profile={"profile": "lean"}, mass=dict(template={}))
    monkeypatch.setattr(series, "load_run", lambda task: deepcopy(config))
    task = dict(run="ac52", configuration=dict(sha256="a" * 64))
    output = tmp_path / "series_attempt"
    with pytest.raises(ValueError, match="legacy full task"):
        runner._run_series(task, output, "cpu", "float32", None, {})
    with pytest.raises(ValueError, match="mass template"):
        monkeypatch.setattr(series, "load_run", lambda task: dict(
            mass=dict(template=dict(storage_profile={"profile": "full"}))))
        runner._run_series(dict(task, storage_profile="lean"), output, "cpu", "float32", None, {})
    assert not output.exists()


# --------------------------------------------------------------- capacity


LEAN_FILES = ("src/darsia/fluidflower/storage_profile.py", "src/darsia/fluidflower/mass/n1_pipeline.py",
              "src/darsia/fluidflower/mass/n1_statistics.py", "src/darsia/fluidflower/pipeline/series.py",
              "src/darsia/fluidflower/native/sidecar.py", "src/darsia/fluidflower/pipeline/runner.py")


def _lean_repository(tmp_path, monkeypatch):
    """Synthetic writer sources at a synthetic commit; never the real repository."""
    repository = tmp_path / "repo"
    source = {key: ("# synthetic writer source " + key + "\n").encode() for key in LEAN_FILES}
    for key, raw in source.items():
        path = repository / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def git(args):
        commit, relative = args[-1].split(":", 1)
        assert commit == "a" * 40
        return source[relative]

    monkeypatch.setattr(cap.subprocess, "check_output", git)
    return repository, source


def _lean_task():
    task, config, *_ = measured()
    task.update(storage_profile="lean", format="darsia-alias-task-v1", format_version=2)
    for frame in config["frames"]:
        _canvas(frame)
    return task, config


def test_a_lean_task_is_admitted_with_its_own_estimate_and_writer_identity(tmp_path, monkeypatch):
    repository, source = _lean_repository(tmp_path, monkeypatch)
    task, config = _lean_task()
    # No capacity map at all: the full measured profile is never read for lean.
    estimate = cap.load_estimate(SimpleNamespace(document={}), "FF_AC", task, config, repository=repository)
    expected = storage.run_estimate("lean", config, pipeline="native-phase-n1")
    assert isinstance(estimate, cap.StorageProfileEstimate) and estimate.storage_profile == "lean"
    assert estimate.case_count == 2 and estimate.estimate == expected
    assert estimate.products_per_frame_estimated_bytes == 522_500_000
    assert estimate.products_estimated_bytes == 2 * 522_500_000
    assert estimate.transient_estimated_bytes == 522_500_000 + 218 * MB
    assert estimate.needed(other_remaining_bytes=0, reserve_bytes=0) == expected["needed_bytes"]
    assert "ESTIMATE" in estimate.model and "never reused" in estimate.model
    assert estimate.writer_identity == [dict(relative_path=key, sha256=hashlib.sha256(source[key]).hexdigest())
                                        for key in LEAN_FILES]
    (repository / LEAN_FILES[0]).write_bytes(source[LEAN_FILES[0]] + b"# changed on disk\n")
    with pytest.raises(ValueError, match="writer identity differs"):
        cap.load_estimate(SimpleNamespace(document={}), "FF_AC", task, config, repository=repository)
    with pytest.raises(RunUnavailable, match="explicit applicable"):  # A full task needs its profile.
        cap.load_estimate(SimpleNamespace(document={}), "FF_AC", measured()[0], config,
                          repository=repository)


def test_a_full_task_keeps_the_measured_profile_path_and_its_pins(tmp_path, monkeypatch):
    """The existing measured-profile test, with and without the explicit full field."""
    import ast
    task, config, profile, measurement, completion = measured()
    config, metadata_rows, profile["measured_native_layout"] = native_metadata(tmp_path / "native-layout")
    from darsia.fluidflower.pipeline import store_runtime
    monkeypatch.setattr(store_runtime, "resolve_catalog", lambda *a: metadata_rows)
    repository = tmp_path / "repo"
    required = ["src/darsia/fluidflower/mass/n1_pipeline.py", "src/darsia/fluidflower/mass/n1_statistics.py",
                "src/darsia/fluidflower/pipeline/series.py", "src/darsia/fluidflower/native/sidecar.py"]
    source = {key: b"# Tiny synthetic identity fixture, never scientific code.\n" for key in required}
    functions = ["_phase_outputs", "_inventory", "validate_frame_artifacts", "_restore_predecessor",
                 "scalar_table", "run_series", "_run_series"]
    runner_path = "src/darsia/fluidflower/pipeline/runner.py"
    source[runner_path] = ("\n".join("def " + key + "(): return None" for key in functions) + "\n").encode()
    for key, raw in source.items():
        path = repository / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    checked = []
    original_identity = cap._profile_writer_identity

    def pinned(*args):
        checked.append(args[3]["relative_path"])
        return original_identity(*args)

    monkeypatch.setattr(cap, "_profile_writer_identity", pinned)
    monkeypatch.setattr(cap.subprocess, "check_output", lambda args: source[args[-1].split(":", 1)[1]])

    def put(path, value):
        raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    row = dict(relative_path=runner_path, measured_sha256=digest(source[runner_path]),
               current_sha256=digest(source[runner_path]),
               functions={node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                          for node in ast.parse(source[runner_path]).body})
    profile.update(measurement=put(tmp_path / "measurement.json", measurement),
                   completion=put(tmp_path / "completion.json", completion),
                   writers=[dict(relative_path=key, sha256=digest(source[key])) for key in required],
                   orchestration=[row], layout_receipt=None)
    document = settings_doc(tmp_path)
    document["capacity"]["profiles"]["FF_AC"] = put(tmp_path / "profile.json", profile)
    stores = SimpleNamespace(document=document)
    legacy = cap.load_estimate(stores, "FF_AC", task, config, repository=repository)
    explicit = cap.load_estimate(stores, "FF_AC", dict(task, storage_profile="full"), config,
                                 repository=repository)
    assert isinstance(legacy, cap.SpaceEstimate) and explicit == legacy
    assert legacy.native_expected_bytes == 250 and checked == required + required
    report = cap.report(legacy, {"free_bytes": 10**12}, 31, 40)
    assert report["needed_bytes"] == legacy.needed(reserve_bytes=40)
    assert report["other_active_remaining_bytes"] == 31
    assert report["other_active_remaining_included_in_results_estimate"] is False
    assert report["message"] == (
        f"ac52: estimated needed {legacy.needed(reserve_bytes=40)} bytes / free {10**12} bytes "
        "(this job, estimated transient space and reserve; estimate, not coordinated across hosts)")


class _Ledger:
    reserve = 40

    def __init__(self, free, other):
        self.free, self.other = free, other

    def locked(self):
        return nullcontext(self)

    def remaining(self, **kwargs):
        return self.other

    def fresh_volume(self):
        return dict(volume_id="fixture-volume", free_bytes=self.free)


class _Cache:
    def prepare(self, task, estimate):
        self.prepared = estimate

    def acquire(self):
        pass

    def check(self, task, estimate, other):
        return dict(state="TESTONLY")

    def unlock(self):
        pass


@pytest.mark.parametrize("shortfall", [0, 1])
def test_claim_admission_refuses_when_free_minus_reserve_is_below_the_lean_estimate(monkeypatch, shortfall):
    task, config = _lean_task()
    rows = ("TESTONLY input rows",)
    monkeypatch.setattr(cap, "configuration_for_task", lambda *args: config)
    monkeypatch.setattr(cap, "_lean_writer_identity", lambda *args: [dict(relative_path=LEAN_FILES[0],
                                                                          sha256="c" * 64)])
    expected = storage.run_estimate("lean", config, pipeline="native-phase-n1")
    other = 31
    # N51: other retained reservations are informative; this job plus reserve
    # determines the result-volume admission boundary on this host.
    free = expected["needed_bytes"] + _Ledger.reserve - shortfall
    refusals = []
    runtime = SimpleNamespace(capacity_refusals={}, availability=lambda actual: rows,
                              stores=SimpleNamespace(), job=dict(series_id="FF_AC"))
    context = cap.ClaimCapacity(runtime, owner(), on_refusal=refusals.append)
    context.ledger, context.cache = _Ledger(free, other), _Cache()
    if shortfall:
        with pytest.raises(RunUnavailable, match="storage profile lean: ESTIMATE"):
            context.availability(task)
        assert refusals and "free " + str(free) + " bytes" in refusals[0]["message"]
        assert task["attempt_id"] not in context.estimates
    else:
        assert context.availability(task) == rows
        admitted = context.estimates[task["attempt_id"]]
        assert admitted.storage_profile == "lean" and context.cache.prepared is admitted


def test_a_lean_reservation_counts_its_own_estimate(tmp_path, monkeypatch):
    task, config = _lean_task()
    monkeypatch.setattr(cap, "_lean_writer_identity", lambda *args: [])
    estimate = cap.lean_estimate(task, config, repository=tmp_path)
    ledger, _stores, _state = ledger_fixture(tmp_path, monkeypatch, free=10**13)
    with ledger.locked():
        path = ledger.bind(tmp_path / "job", claim_fixture(), owner(), estimate)
        row = json.loads((path / "reserved.json").read_bytes())
        assert row["products_per_frame_estimated_bytes"] == 522_500_000
        assert row["transient_estimated_bytes"] == 522_500_000 + 218 * MB
        assert row["estimate"]["storage_profile"] == "lean" and "ESTIMATE" in row["estimate"]["model"]
        assert ledger.remaining() == 2 * 522_500_000 + 522_500_000 + 218 * MB
        ledger.progress(path, owner(), dict(state="frame_completed", complete=True, run="ac52", case_id="ac52_f1"))
        assert ledger.remaining() == 522_500_000 + 522_500_000 + 218 * MB


def test_whole_job_report_labels_lean_runs_as_estimates(tmp_path, monkeypatch):
    from darsia.fluidflower import job_filesystem as fs
    task, config = _lean_task()
    monkeypatch.setattr(cap, "_lean_writer_identity", lambda *args: [])
    monkeypatch.setattr(fs, "inspect_results_root", lambda _: SimpleNamespace(allowed=True, protocol="smb",
                                                                               local=False))
    monkeypatch.setattr(cap, "CapacityLedger", lambda *a: SimpleNamespace(
        reserve=40, remaining=lambda **kw: 31, fresh_volume=lambda: dict(free_bytes=10**13), remote_scope=None))
    stores = SimpleNamespace(machine_id="main", roots=lambda _: dict(results=str(tmp_path / "results")))
    report = cap.whole_job_report(stores, "FF_AC", [task], [config], repository=tmp_path)
    expected = storage.run_estimate("lean", config, pipeline="native-phase-n1")
    assert report["needed_bytes"] == expected["needed_bytes"] + 40
    assert report["other_active_remaining_bytes"] == 31
    assert report["other_active_remaining_included_in_results_estimate"] is False
    assert report["results_space_policy"] == "estimate, not coordinated across hosts"
    assert report["runs"][0]["storage_profile"] == "lean"
    assert "lean runs: ESTIMATE" in report["message"]


@pytest.mark.parametrize("route", [cap.n43_entitlement, cap.first_output_entitlement])
def test_qualification_routes_bind_the_full_profile(route):
    task, config = _lean_task()
    with pytest.raises(ValueError, match="binds the full storage profile"):
        route(None, task, config, ())


# --------------------------------------------------------------- Jobs panel


def _snapshot(root, facts):
    return dict(root=root, name="Tiny", series="Local", state="Waiting", job_facts=facts,
                master_machine="Main", run_count=1, completed_runs=0, committed=0, total_frames=2,
                remaining=2, frames_per_minute=None, eta_seconds=None, workers=[], machines=[],
                logs=[], errors=[], runs=[])


def test_the_dialog_chooser_switches_the_displayed_estimate(qtbot):
    from darsia.fluidflower import jobs_panel as ui
    plan = dict(path=r"\\server\share\fixed.toml", sha256="a" * 64, size_bytes=15, label="Frozen")
    dialog = ui.StartAnalysisDialog([plan])
    qtbot.addWidget(dialog)
    combo = dialog.profile_combo
    assert [combo.itemData(index) for index in range(combo.count())] == ["full", "lean"]
    assert combo.currentData() == storage.DEFAULT_STORAGE_PROFILE == "full"
    full_text = dialog.estimate_label.text()
    assert "Storage estimate for full" in full_text and "842.6 MB per frame" in full_text
    assert "AC52" not in full_text and "documented reference size" in full_text
    assert ui.LEAN_NOTE == "Lean uses the same computation as Full and keeps fewer intermediate files."
    assert "352.0 MB per frame in phase two" in full_text and "ESTIMATE" in full_text
    assert "not checked on this machine" in full_text
    combo.setCurrentIndex(combo.findData("lean"))
    lean_text = dialog.estimate_label.text()
    assert lean_text != full_text and "Storage estimate for lean" in lean_text
    assert "522.5 MB per frame" in lean_text and "31.9 MB per frame in phase two" in lean_text
    assert "218.0 MB per run" in lean_text and "130.5 MB per frame" in lean_text
    dialog.job_edit.setText(r"\\server\share\jobs\new")
    assert dialog.choices()["storage_profile"] == "lean"
    known = ui.StartAnalysisDialog([plan], free_space=dict(free_bytes=1_234_000_000_000))
    qtbot.addWidget(known)
    assert "(last check): 1.2 TB" in known.estimate_label.text()


def test_an_existing_job_shows_its_stored_profile_read_only(qtbot):
    from darsia.fluidflower import jobs_panel as ui
    widget = ui.JobsPanel(local_machine="Main")
    qtbot.addWidget(widget)
    assert widget.profile_label.text() == "Storage profile: no job selected."
    widget._show_snapshot(_snapshot("lean-root", dict(format="darsia-alias-job-v1", format_version=2,
                                                      storage_profile="lean")))
    text = widget.profile_label.text()
    assert "Storage profile: lean" in text and "stored in job.json" in text and "read-only" in text
    widget._show_snapshot(_snapshot("legacy-root", dict(precision="float32")))
    text = widget.profile_label.text()
    assert "Storage profile: full" in text and "legacy job.json without the field" in text
    widget._show_snapshot(_snapshot("bad-root", dict(storage_profile="small")))
    assert "invalid job metadata" in widget.profile_label.text()
    # The panel offers no profile chooser: an existing job's profile cannot change here.
    from PySide6.QtWidgets import QComboBox
    for combo in widget.findChildren(QComboBox):
        assert "lean" not in [combo.itemData(index) for index in range(combo.count())]
    assert widget.can_close()


def test_create_passes_the_choice_and_shows_the_creation_estimate(qtbot):
    from darsia.fluidflower import jobs_panel as ui
    calls = []
    estimate = storage.estimate_prepared_job("lean", [dict(run="ac52", pipeline=PHASE_TWO)], [_config()])

    def creator(plan, root, **kwargs):
        calls.append(kwargs)
        return dict(root=root, name="Created", series_label="Local", active=True, worker_started=False,
                    producer_receipt=dict(storage_estimate=estimate))

    facts = dict(format="darsia-alias-job-v1", format_version=2, storage_profile="lean")
    widget = ui.JobsPanel(local_machine="Main", job_creator=creator, reader=lambda row: _snapshot(row["root"], facts))
    qtbot.addWidget(widget)
    widget.set_context(series_label="Local", series_root="series", series_id="unit",
                       selection_provider=lambda runs: [])
    widget.create_job(dict(path="explicit-plan"), "fresh-job", "Name", storage_profile="lean")
    qtbot.waitUntil(widget.can_close, timeout=10000)
    qtbot.waitUntil(lambda: (widget.current or {}).get("root") == "fresh-job", timeout=10000)
    assert calls[0]["storage_profile"] == "lean"
    text = widget.profile_label.text()
    assert "Storage profile: lean" in text
    assert "Estimate at creation: 281.8 MB for 2 frames, plus 31.9 MB in flight" in text
    widget.create_job(dict(path="explicit-plan"), "second-job", "Name")
    qtbot.waitUntil(widget.can_close, timeout=10000)
    assert "storage_profile" not in calls[1]  # No choice: the producer's default for new jobs.


def test_open_create_dialog_offers_the_free_space_the_panel_knows(qtbot, monkeypatch):
    from darsia.fluidflower import jobs_panel as ui
    seen = []

    class Cancel:
        def __init__(self, plans, parent=None, **kwargs):
            seen.append(kwargs)

        def exec(self):
            return ui.QDialog.DialogCode.Rejected

    monkeypatch.setattr(ui, "StartAnalysisDialog", Cancel)
    widget = ui.JobsPanel(local_machine="Main", plans_provider=lambda: [])
    qtbot.addWidget(widget)
    widget.set_context(series_label="Local", series_root="series", series_id="unit",
                       selection_provider=lambda runs: [])
    widget.current = dict(root="r", job_facts=dict(format="darsia-alias-job-v1"))
    widget._capacity_ready(dict(generation=widget._generation, root="r",
                                result=dict(message="m", sufficient=True, free_bytes=5 * 10**12,
                                            results_root="results")))
    widget.open_create_dialog()
    assert seen[-1]["free_space"]["free_bytes"] == 5 * 10**12
    widget.set_context(series_label="Other", series_root="other", series_id="other-series",
                       selection_provider=lambda runs: [])
    widget.open_create_dialog()
    assert seen[-1]["free_space"] is None  # Another series: its results volume is not known.


def test_create_from_plan_refuses_an_unknown_choice_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="storage_profile"):
        create_job_from_plan(dict(path=str(tmp_path / "missing.toml")), tmp_path / "job",
                             display_name="x", series_label="x", storage_profile="small")
    assert not (tmp_path / "job").exists()
