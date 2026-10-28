"""Passive catalogue boundaries; source only until an authorized test run.

Most tests mock the registry/snapshot boundary to stay out of queue execution.
A regression test uses the real JobsUIRegistry register/read/hash-key boundary;
only its UNC path gate admits an exact local temporary fixture. Original
immutable metadata reads and shared references remain real. The integration
fixtures republish genuine tiny N1 and native-phase output using the original
metadata publisher and construct the real NativeSavedReview. Their synthetic
job envelope retains the original schema's omission of pipeline; each task
selects the protocol. These tests do not qualify the queue's terminal validator.
"""
from copy import deepcopy
import json
from pathlib import Path, PureWindowsPath

import pytest

from darsia.fluidflower import calibration_review_native_catalog as catalog
from darsia.fluidflower import native_jobs_registry as registry_module
from darsia.fluidflower import storage_n1_saved_products as saved_n1
from darsia.fluidflower import storage_saved_products as saved_phase
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.pipeline import job_queue, products_publication as publication
from darsia.fluidflower.storage_saved_products import StoredProductCollectionRequired
from .test_calibration_review_native import native_attachment, phase1_data, tiny_config
from .test_calibration_review_native_phase import native_phase_attachment


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return job_queue.descriptor(path)


def _forbidden(*args, **kwargs):
    raise AssertionError("Passive catalogue crossed its explicit metadata boundary")


class _Registry:
    def __init__(self, registrations):
        self.registrations, self.calls = registrations, []

    def jobs(self, root, series):
        self.calls.append((root, series))
        return self.registrations

    known_jobs = _forbidden


def _refresh_job(value):
    value["registration"]["job"] = _write(value["root"] / "job.json", value["job"])
    value["snapshot"]["job_facts"] = deepcopy(value["job"])


@pytest.fixture
def registered_job(tmp_path, monkeypatch):
    root = tmp_path / "registered-job"
    series_root = tmp_path / "selected-series-root"
    series_root.mkdir()
    (series_root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "FF_AC"\n', encoding="utf-8-sig")
    run = "ac52"
    job = dict(format="darsia-alias-job-v1", format_version=1, series_id="FF_AC",
        expected_runs=[run], calibration_sha256="f" * 64, execution_ready=True,
        runtime_support="darsia-role-io-runtime-v1", precision="float32")
    registration = dict(root=str(root), name="Evening execution", series_label="UI association with ffac")
    snapshot = dict(job_facts={}, runs=[dict(run=run, state="Complete")], errors=[])
    value = dict(root=root, series_root=series_root, run=run, job=job, registration=registration,
        snapshot=snapshot, snapshot_calls=[])
    _refresh_job(value)
    _write(root / "results" / run / "task.json",
        dict(format="darsia-alias-task-v1", pipeline="n1", run=run,
             calibration_sha256=job["calibration_sha256"]))
    value["registry"] = _Registry([registration])
    def read_snapshot(_reader, item):
        value["snapshot_calls"].append(deepcopy(item))
        return deepcopy(snapshot)
    monkeypatch.setattr(JobSnapshotReader, "__call__", read_snapshot)
    return value


@pytest.fixture
def native_catalogue(registered_job, native_attachment):
    """Real publication binding to a local synthetic immutable job envelope."""
    _config, completion, original_index = native_attachment
    return _register_native_completion(registered_job, completion, original_index, "n1")


@pytest.fixture
def native_phase_catalogue(registered_job, native_phase_attachment):
    return _register_native_completion(registered_job, native_phase_attachment["completion"],
        native_phase_attachment["index"], "native-phase-n1")


def _register_native_completion(value, completion, original_index, pipeline):
    root, run = value["root"], completion["run"]
    value["run"] = run
    value["job"].update(expected_runs=[run], calibration_sha256=completion["calibration"]["sha256"])
    # prepare_alias_job excludes pipeline from job.json for both routes;
    # series_tasks records it on each task. Do not invent a job-level field.
    assert "pipeline" not in value["job"]
    value["snapshot"]["runs"] = [dict(run=run, state="Complete")]
    _refresh_job(value)
    _write(root / "results" / run / "task.json",
        dict(format="darsia-alias-task-v1", pipeline=pipeline, run=run,
             calibration_sha256=value["job"]["calibration_sha256"]))
    _old_root, original_document, original_rows = publication._read_index(original_index)
    completed_path = root / "results" / run / "completed.json"
    completed_path.write_bytes(Path(original_rows["completed.json"]["local_path"]).read_bytes())
    products = [publication.product(key, **{field: row[field] for field in
        ("machine_id", "local_path", "sha256", "bytes", "kind")})
        for key, row in original_rows.items() if key != "completed.json"]
    metadata = [key for key, row in original_rows.items()
                if key != "completed.json" and "metadata_copy" in row]
    index = publication.publish_run_metadata(root / "publication",
        job=value["registration"]["job"], run=run, attempt_id=original_document["attempt_id"],
        machine_id="Main", completed=job_queue.descriptor(completed_path),
        products=products, metadata_product_ids=metadata,
        file_writers_closed=True, all_final_products_listed=True)
    done = dict(format="darsia-alias-done-v1", run=run,
        attempt_id=original_document["attempt_id"],
        completed=job_queue._own_reference(root, completed_path),
        products_index=job_queue._own_reference(root, index["path"]))
    _write(root / "done" / (run + ".json"), done)
    value.update(index=index, done=done, completion=completion)
    return value


def test_native_phase_catalogue_dispatches_original_metadata_and_selected_frame(native_phase_catalogue, monkeypatch):
    value = native_phase_catalogue
    original_job = Path(value["registration"]["job"]["path"]).read_bytes()
    task = json.loads((value["root"] / "results" / value["run"] / "task.json").read_bytes())
    assert "pipeline" not in value["job"] and "pipeline" not in value["snapshot"]["job_facts"]
    assert "pipeline" not in json.loads(original_job)
    assert task["pipeline"] == value["completion"]["pipeline"] == "native-phase-n1"
    with monkeypatch.context() as guard:
        guard.setattr(saved_n1, "read_n1_stored_run", _forbidden)
        guard.setattr(saved_phase, "read_stored_frame", _forbidden)
        rows, errors = _discover(value, current_machine_id="Main")
    assert errors == [] and len(rows) == 1
    assert value["snapshot_calls"] == [value["registration"]]
    assert Path(value["registration"]["job"]["path"]).read_bytes() == original_job
    review = rows[0]["reader"]
    assert review.pipeline == "native-phase-n1"
    stored = review.read_frame(0)
    assert "producer_kind" not in stored
    assert stored["storage_document"]["format"] == "darsia-storage-frame-v1"
    # The native-phase completion does not bind original photo receipts. The
    # catalogue never invents a mapping from indexed filenames or source paths.
    assert review.photographs == {}
    with pytest.raises(FileNotFoundError, match="explicit indexed receipt"):
        review.read_photo(stored)


@pytest.mark.parametrize("job_pipeline,task_pipeline", [("n1", "native-phase-n1"), ("native-phase-n1", "n1")])
def test_authenticated_job_and_task_protocols_must_agree(registered_job, monkeypatch, job_pipeline, task_pipeline):
    value = registered_job
    value["job"]["pipeline"] = job_pipeline
    _refresh_job(value)
    path = value["root"] / "results" / value["run"] / "task.json"
    task = json.loads(path.read_bytes())
    task["pipeline"] = task_pipeline
    _write(path, task)
    monkeypatch.setattr(catalog, "_bound_owned", _forbidden)
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1 and "protocol differs" in errors[0]


@pytest.mark.parametrize("fixture_name,wrong", [("native_catalogue", "native-phase-n1"), ("native_phase_catalogue", "n1")])
@pytest.mark.parametrize("explicit_job_protocol", [False, True])
def test_authenticated_task_cannot_admit_other_completion_protocol(
        request, monkeypatch, fixture_name, wrong, explicit_job_protocol):
    value = request.getfixturevalue(fixture_name)
    assert "pipeline" not in value["job"]
    if explicit_job_protocol:
        value["job"]["pipeline"] = wrong
        _refresh_job(value)
    # Republish the unchanged genuine completion bytes. The opposite task
    # must refuse both the original omitted-field job and an explicit agreeing
    # job field: neither can relabel a different original completion protocol.
    _root, document, original_rows = publication._read_index(value["index"])
    products = [publication.product(key, **{field: row[field] for field in
        ("machine_id", "local_path", "sha256", "bytes", "kind")})
        for key, row in original_rows.items() if key != "completed.json"]
    metadata = [key for key, row in original_rows.items()
                if key != "completed.json" and "metadata_copy" in row]
    index = publication.publish_run_metadata(value["root"] / "opposite-publication",
        job=value["registration"]["job"], run=value["run"], attempt_id=document["attempt_id"],
        machine_id="Main", completed=job_queue.descriptor(value["root"] / "results" / value["run"] / "completed.json"),
        products=products, metadata_product_ids=metadata,
        file_writers_closed=True, all_final_products_listed=True)
    value["done"]["products_index"] = job_queue._own_reference(value["root"], index["path"])
    _write(value["root"] / "done" / (value["run"] + ".json"), value["done"])
    _write(value["root"] / "results" / value["run"] / "task.json", dict(pipeline=wrong,
        run=value["run"], calibration_sha256=value["job"]["calibration_sha256"]))
    monkeypatch.setattr(catalog, "NativeSavedReview", _forbidden)
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1 and "completion protocol differs" in errors[0]


@pytest.mark.parametrize("fixture_name", ["native_catalogue", "native_phase_catalogue"])
def test_only_missing_task_protocol_uses_original_n1_default(request, fixture_name):
    value = request.getfixturevalue(fixture_name)
    assert "pipeline" not in value["job"]
    task_path = value["root"] / "results" / value["run"] / "task.json"
    task = json.loads(task_path.read_bytes())
    task.pop("pipeline")
    _write(task_path, task)
    rows, errors = _discover(value)
    if value["completion"].get("pipeline") == "native-phase-n1":
        assert rows == [] and len(errors) == 1 and "completion protocol differs" in errors[0]
    else:
        assert errors == [] and len(rows) == 1
        assert rows[0]["reader"].pipeline == "n1"


def _discover(value, **options):
    return catalog.discover_native_reviews(value["registry"],
        options.pop("series_root", value["series_root"]), options.pop("series_id", "ffac"),
        current_machine_id=options.pop("current_machine_id", "Second"), **options)


def test_real_native_constructor_is_metadata_only_and_preserves_original_bindings(native_catalogue, monkeypatch):
    value = native_catalogue
    assert "pipeline" not in value["job"]
    original_tree = set(value["root"].parent.rglob("*"))
    original_files = {path: path.read_bytes() for path in original_tree if path.is_file()}
    with monkeypatch.context() as guard:
        guard.setattr(saved_n1, "read_n1_stored_frame", _forbidden)
        guard.setattr(publication, "collect_products", _forbidden)
        guard.setattr(job_queue, "JobQueue", _forbidden)
        guard.setattr(Path, "glob", _forbidden)
        guard.setattr(Path, "rglob", _forbidden)
        rows, errors = _discover(value)
    assert errors == []
    assert len(rows) == 1 and set(rows[0]) == {"reader", "dataset", "calibration_name", "registration", "source_registrations"}
    assert rows[0]["registration"] == value["registration"]
    assert rows[0]["registration"] is not value["registration"]
    review = rows[0]["reader"]
    assert isinstance(review, catalog.NativeSavedReview)
    assert rows[0]["dataset"] == "ffac"
    bound = value["done"]["products_index"]
    assert review.index == dict(path=value["index"]["path"], sha256=bound["sha256"], size_bytes=bound["bytes"])
    assert review.current_machine_id == "Second"
    assert review.expected_run == value["run"]
    assert review.expected_calibration_sha256 == value["job"]["calibration_sha256"]
    assert review.collections == () and review.source_paths is None
    assert review.geometry_producer_sha256 is None
    assert value["registry"].calls == [(value["series_root"], "FF_AC")]
    assert value["snapshot_calls"] == [value["registration"]]
    assert {path: path.read_bytes() for path in original_files} == original_files
    assert set(value["root"].parent.rglob("*")) == original_tree
    basename = PureWindowsPath(value["completion"]["calibration"]["path"]).name
    assert rows[0]["calibration_name"] == "Saved calibration: " + basename + " · Execution: Evening execution"
    with pytest.raises(ValueError, match="differs"):
        review.read_photo({})
    # A receiving host must await explicit original collection bindings. The
    # catalogue may not impersonate Main or infer aliases for indexed arrays.
    with pytest.raises(StoredProductCollectionRequired):
        review.read_frame(0)


def test_real_registry_uses_literal_manifest_key_and_selected_root(native_catalogue, monkeypatch):
    value = native_catalogue
    registry = registry_module.JobsUIRegistry(value["root"].parent / "ui-registry")
    def admit_only_local_fixture(path):
        # Exercise real registry hashing, persistence, descriptors and lookup;
        # no live UNC share or production registration is used by this test.
        assert Path(path) == value["root"]
        return path
    monkeypatch.setattr(registry_module, "require_unc", admit_only_local_fixture)
    registry.register_job(value["series_root"], "FF_AC", value["root"], label="Evening execution")
    literal_bucket = registry._file(value["series_root"], "FF_AC")
    alias_bucket = registry._file(value["series_root"], "ffac")
    assert literal_bucket != alias_bucket and literal_bucket.is_file() and not alias_bucket.exists()
    assert registry.jobs(value["series_root"], "ffac") == []
    original_registry = literal_bucket.read_bytes()
    value["registry"] = registry
    monkeypatch.setattr(registry, "known_jobs", _forbidden)
    rows, errors = _discover(value)
    assert errors == [] and len(rows) == 1
    assert isinstance(rows[0]["reader"], catalog.NativeSavedReview)
    assert rows[0]["dataset"] == "ffac"
    assert len(value["snapshot_calls"]) == 1
    assert value["snapshot_calls"][0]["series_label"] == "UI association with FF_AC"
    assert literal_bucket.read_bytes() == original_registry and not alias_bucket.exists()

    # Equal series spelling does not authorize discovering a different folder's
    # registry. A selected root with no explicit registrations stays empty.
    other_root = value["root"].parent / "another-series-root"
    other_root.mkdir()
    (other_root / "series.toml").write_text('schema_version = 1\nseries_id = "FF_AC"\n', encoding="utf-8")
    assert _discover(value, series_root=other_root) == ([], [])
    assert len(value["snapshot_calls"]) == 1
    assert literal_bucket.read_bytes() == original_registry


def test_existing_done_payload_opens_foreign_maps_and_photo_without_collecting(native_catalogue, monkeypatch):
    value = native_catalogue
    bundle = publication.publish_indexed_products(value["index"], value["root"] / "maps-publication",
        current_machine_id="Main", publish=["map"], file_writers_closed=True)
    done = dict(value["done"], payload_publication=job_queue._own_reference(value["root"], bundle["path"]))
    _write(value["root"] / "done" / (value["run"] + ".json"), done)
    monkeypatch.setattr(publication, "collect_products", _forbidden)
    rows, errors = _discover(value)
    assert errors == [] and len(rows) == 1
    review = rows[0]["reader"]
    assert review.current_machine_id == "Second" and len(review.collections) == 1
    stored = review.read_frame(0)
    image, evidence = review.read_photo(stored)
    assert image.shape == (3, 3, 4)
    assert evidence["geometry_producer_sha256"] == stored["photograph"]["geometry_producer_sha256"]


def test_explicit_collection_bindings_survive_catalogue_refresh_for_exact_index(native_catalogue):
    value = native_catalogue
    _, _, products = publication._read_index(value["index"])
    selected = [key for key, row in products.items() if row["kind"] == "map" and "/storage_" in key]
    collection = publication.collect_products(value["index"], selected, value["root"] / "opened-collection",
        current_machine_id="Second", source_paths={key: products[key]["local_path"] for key in selected},
        file_writers_closed=True)
    rows, errors = _discover(value, extra_collections={value["index"]["sha256"]: [collection]})
    assert errors == [] and len(rows) == 1
    assert rows[0]["reader"].read_frame(0)["case_id"] == rows[0]["reader"].case_ids[0]
    other, errors = _discover(value, extra_collections={"0" * 64: [collection]})
    assert errors == [] and other[0]["reader"].collections == ()
    with pytest.raises(StoredProductCollectionRequired):
        other[0]["reader"].read_frame(0)


@pytest.mark.parametrize("raw", [
    None,
    'schema_version = 2\nseries_id = "FF_AC"\n',
    'schema_version = true\nseries_id = "FF_AC"\n',
    'series_id = "FF_AC"\n',
    'schema_version = 1\n',
    'schema_version = 1\nseries_id = 123\n',
    'schema_version = 1\nseries_id = ""\n',
    'schema_version = 1\nseries_id = "FF_UM"\n',
    'schema_version = 1\nseries_id = "FF-AC"\n',
    'schema_version = [invalid TOML\n',
])
def test_invalid_or_different_series_manifest_stops_before_registry(registered_job, raw):
    value = registered_job
    manifest = value["series_root"] / "series.toml"
    if raw is None:
        manifest.unlink()
    else:
        manifest.write_text(raw, encoding="utf-8")
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1
    assert errors[0].startswith("Registered native reviews unavailable:")
    assert value["registry"].calls == [] and value["snapshot_calls"] == []


def test_series_manifest_keeps_original_jobs_metadata_size_limit(registered_job):
    value = registered_job
    (value["series_root"] / "series.toml").write_bytes(b"#" + b"x" * (4 * 1024 * 1024))
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1 and "too large for Jobs UI metadata" in errors[0]
    assert value["registry"].calls == [] and value["snapshot_calls"] == []


def test_cancelled_lookup_never_reads_the_selected_manifest(registered_job, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", _forbidden)
    assert _discover(registered_job, cancelled=lambda: True) == ([], [])
    assert registered_job["registry"].calls == []


@pytest.mark.parametrize("root", [None, ""])
def test_missing_selected_root_cannot_fall_back_to_working_directory(registered_job, monkeypatch, root):
    reads = []
    def attempted(path):
        reads.append(path)
        raise AssertionError("Missing selection attempted a working-directory read")
    monkeypatch.setattr(Path, "read_bytes", attempted)
    rows, errors = _discover(registered_job, series_root=root)
    assert rows == [] and len(errors) == 1 and "explicit selected series root" in errors[0]
    assert reads == [] and registered_job["registry"].calls == []


@pytest.mark.parametrize(("calibration", "top_name", "expected"), [
    ({"path": r"C:\freezes\chosen.json"}, None, "Saved calibration: chosen.json"),
    ({"path": "/freezes/chosen.json", "display_name": "  Native saved choice  "}, None, "Native saved choice"),
    ({"name": "Frozen choice"}, None, "Frozen choice"),
    ({}, "Saved choice", "Saved choice"),
    ({"name": " "}, None, "Saved calibration"),
])
def test_calibration_label_keeps_execution_context_separate(calibration, top_name, expected):
    completion = dict(calibration=calibration, calibration_name=top_name)
    assert catalog._label({"name": "Calibration-looking job title"}, completion) == (
        expected + " · Execution: Calibration-looking job title")
    assert catalog._label({}, completion) == expected


@pytest.mark.parametrize("state", ["complete", "Working", "Metadata error", "Complete (partial)", None])
def test_only_exact_complete_rows_are_considered(registered_job, monkeypatch, state):
    value = registered_job
    value["snapshot"]["runs"][0]["state"] = state
    monkeypatch.setattr(catalog, "_bound_owned", _forbidden)
    monkeypatch.setattr(catalog, "NativeSavedReview", _forbidden)
    assert _discover(value) == ([], [])


@pytest.mark.parametrize("actual_series", ["FF_UM", "ffac-extra", "ff-ac", None, 123])
def test_ui_association_cannot_override_actual_job_series(registered_job, actual_series):
    value = registered_job
    value["job"]["series_id"] = actual_series
    _refresh_job(value)
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1 and "series_id differs" in errors[0]
    assert value["snapshot_calls"] == []


@pytest.mark.parametrize("pipeline", ["native-phase-statistics", "n1-calibration-refit", "series-setup", "lattice-phase-n1"])
def test_unsupported_job_protocol_is_skipped_before_snapshot(registered_job, pipeline):
    value = registered_job
    value["job"]["pipeline"] = pipeline
    value["job"].pop("series_id")  # Other protocols need not share native schema.
    _refresh_job(value)
    assert _discover(value) == ([], [])
    assert value["snapshot_calls"] == []


@pytest.mark.parametrize("pipeline", ["native-phase-statistics", "n1-calibration-refit", "series-setup", "lattice-phase-n1"])
def test_unsupported_complete_task_protocol_never_opens_done(registered_job, monkeypatch, pipeline):
    value = registered_job
    _write(value["root"] / "results" / value["run"] / "task.json", dict(pipeline=pipeline))
    monkeypatch.setattr(catalog, "_bound_owned", _forbidden)
    assert _discover(value) == ([], [])


@pytest.mark.parametrize("field", ["run", "calibration_sha256"])
def test_complete_task_must_retain_job_run_and_freeze(registered_job, monkeypatch, field):
    value = registered_job
    path = value["root"] / "results" / value["run"] / "task.json"
    task = json.loads(path.read_bytes())
    task[field] = "another-run" if field == "run" else "0" * 64
    _write(path, task)
    monkeypatch.setattr(catalog, "_bound_owned", _forbidden)
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1 and "job/run/calibration" in errors[0]


def test_legacy_job_is_skipped_and_changed_registration_is_rejected(registered_job):
    value = registered_job
    value["job"]["format"] = "legacy-job"
    _refresh_job(value)
    assert _discover(value) == ([], [])
    value["job"]["format"] = "darsia-alias-job-v1"
    _write(value["root"] / "job.json", value["job"])
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1
    assert value["snapshot_calls"] == []


@pytest.mark.parametrize("fault", ["job_identity", "run_order", "missing_runs", "errors_type"])
def test_malformed_or_stale_snapshot_is_reported(registered_job, fault):
    value = registered_job
    if fault == "job_identity":
        value["snapshot"]["job_facts"]["series_id"] = "FF_UM"
    elif fault == "run_order":
        value["snapshot"]["runs"] = [dict(run="another-run", state="Complete")]
    elif fault == "missing_runs":
        value["snapshot"].pop("runs")
    else:
        value["snapshot"]["errors"] = None
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1


@pytest.mark.parametrize("fault", ["sha256", "bytes", "escape", "attempt_id", "completion_location"])
def test_original_done_bindings_are_required(native_catalogue, fault):
    value = native_catalogue
    done = deepcopy(value["done"])
    if fault == "sha256":
        done["products_index"]["sha256"] = "0" * 64
    elif fault == "bytes":
        done["products_index"]["bytes"] += 1
    elif fault == "escape":
        done["products_index"]["relative_path"] = "../unregistered/products_index.json"
    elif fault == "attempt_id":
        done["attempt_id"] = "b" * 32
    else:
        done["completed"]["relative_path"] = "publication/completed.json"
    _write(value["root"] / "done" / (value["run"] + ".json"), done)
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1


def test_bound_index_from_a_different_job_is_not_adopted(native_catalogue):
    value = native_catalogue
    path = Path(value["index"]["path"])
    document = json.loads(path.read_bytes())
    document["job"]["sha256"] = "0" * 64
    _write(path, document)
    done = deepcopy(value["done"])
    done["products_index"] = job_queue._own_reference(value["root"], path)
    _write(value["root"] / "done" / (value["run"] + ".json"), done)
    rows, errors = _discover(value)
    assert rows == [] and len(errors) == 1 and "original registered job/run/freeze" in errors[0]


def test_bad_registration_does_not_hide_an_independent_complete_review(native_catalogue):
    value = native_catalogue
    bad = deepcopy(value["registration"])
    bad["name"], bad["job"]["sha256"] = "Broken registration", "0" * 64
    value["registry"].registrations = [None, bad, value["registration"]]
    rows, errors = _discover(value)
    assert len(rows) == 1 and len(errors) == 2
    assert "explicit metadata record" in errors[0]
    assert errors[1].startswith("Broken registration:")


def test_native_reader_rejection_is_reported_without_an_attachment(native_catalogue, monkeypatch):
    def rejected(*args, **kwargs):
        raise ValueError("Original native saved contract rejected")
    monkeypatch.setattr(catalog, "NativeSavedReview", rejected)
    rows, errors = _discover(native_catalogue)
    assert rows == [] and len(errors) == 1
    assert errors[0].endswith("Original native saved contract rejected")


@pytest.mark.parametrize("machine", [None, "", "../Main", "Main with spaces"])
def test_receiving_machine_is_explicit_and_validated_before_registry(registered_job, machine):
    value = registered_job
    rows, errors = _discover(value, current_machine_id=machine)
    assert rows == [] and len(errors) == 1
    assert value["registry"].calls == []


def test_cancellation_stops_before_registry_or_before_a_complete_run(registered_job, monkeypatch):
    value = registered_job
    assert _discover(value, cancelled=lambda: True) == ([], [])
    assert value["registry"].calls == []
    monkeypatch.setattr(catalog, "_bound_owned", _forbidden)
    assert _discover(value, cancelled=lambda: bool(value["snapshot_calls"])) == ([], [])
    assert len(value["snapshot_calls"]) == 1


def test_cancellation_after_metadata_resolution_does_not_construct_reader(native_catalogue, monkeypatch):
    original, resolved = catalog._bound_owned, []
    def tracked(root, reference):
        bound = original(root, reference)
        resolved.append(bound)
        return bound
    monkeypatch.setattr(catalog, "_bound_owned", tracked)
    monkeypatch.setattr(catalog, "NativeSavedReview", _forbidden)
    assert _discover(native_catalogue, cancelled=lambda: len(resolved) == 2) == ([], [])
    assert len(resolved) == 2


def test_registry_errors_are_returned_without_follow_on_reads(registered_job, monkeypatch):
    def unavailable(*args):
        raise OSError("Registered share unavailable")
    monkeypatch.setattr(registered_job["registry"], "jobs", unavailable)
    rows, errors = _discover(registered_job)
    assert rows == [] and errors == ["Registered native reviews unavailable: Registered share unavailable"]
    assert registered_job["snapshot_calls"] == []
