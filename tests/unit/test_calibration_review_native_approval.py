"""Native approval boundaries against original saved metadata.

The orchestration specimen uses genuine tiny N1 saved products, shared metadata,
the original single-model-origin proof and the original append-only writer.
Its Jobs terminal snapshot and series lock are explicit test seams: it does not
qualify a real queue execution or a campaign calibration/model selection. Real
series-lock refusal is exercised separately with the original lock API.
"""
from copy import deepcopy
import datetime as dt
import json
from pathlib import Path

import pytest

from darsia.fluidflower import calibration_approvals as records
from darsia.fluidflower import calibration_review_native_approval as approval
from darsia.fluidflower import calibration_review_native_catalog as catalogue
from darsia.fluidflower import phase_model_library
from darsia.fluidflower import series_calibration_approval
from darsia.fluidflower import storage_n1_saved_products as saved_n1
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia.fluidflower.pipeline import job_queue, products_publication
from darsia.fluidflower.pipeline.job_inputs import CATALOG_FORMAT, InputCatalog, catalog_order_key
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from .test_calibration_review_native import phase1_data, tiny_config
from .test_calibration_review_native_catalog import registered_job, native_catalogue, _write
from .test_calibration_review_native_phase import native_phase_attachment
from .test_series_calibration_approval import bound_products
from .test_storage_n1_saved_products import _inputs, _run, _index


_PRODUCER_MODEL = "Z:/foreign-producer/phase-model-source"


def _model_consumers(phase):
    """Explicit original consumer names, including placement and projection."""
    bundles = phase["model_bundles"]
    return [
        ("phase.model.native.co2_presence", bundles["native"]["documents"]["co2_presence"], "models/generic/co2_presence_signal.json"),
        ("phase.model.native.gas_within_co2", bundles["native"]["documents"]["gas_within_co2"], "models/generic/gas_within_co2_signal.json"),
        ("phase.model.edge.co2_presence", bundles["edge"]["documents"]["co2_presence"], "models/generic/co2_presence_signal.json"),
        ("phase.model.edge.gas_within_co2", bundles["edge"]["documents"]["gas_within_co2"], "models/generic/gas_within_co2_signal.json"),
        ("phase.placement.pixel_model", bundles["placement"]["pixel_model"], "placement/pixel.json"),
        ("phase.placement.regional_model", bundles["placement"]["regional_model"], "placement/regional.json"),
        ("phase.placement.projected.pixel", bundles["placement"]["projected_models"]["pixel"], "placement/projected_pixel.json"),
        ("phase.placement.projected.regional", bundles["placement"]["projected_models"]["regional"], "placement/projected_regional.json"),
    ]


@pytest.fixture
def native_attachment(tmp_path, tiny_config, phase1_data):
    # These saved metadata assets are not trained models; _origin verifies their
    # exact location and bytes without invoking inference or fabricating output.
    model = tmp_path / "models" / "phase-model-source"
    _write(model / "summary.json", dict(model_id=model.name))
    heads = {key: _write(model / "models/generic" / (key + "_signal.json"), dict(coefficients=[1.]))
             for key in ("co2_presence", "gas_within_co2")}
    def original(bound):
        return dict(bound, path=_PRODUCER_MODEL + "/" + Path(bound["path"]).relative_to(model).as_posix())
    heads = {key: original(bound) for key, bound in heads.items()}
    placement = {key: original(_write(model / "placement" / (key + ".json"), dict(model=key)))
                 for key in ("pixel", "regional", "projected_pixel", "projected_regional")}
    bundle = dict(documents=heads, requirements=dict(expected_sha256={key: value["sha256"] for key, value in heads.items()}))
    phase = _write(tmp_path / "phase_configuration.json", dict(model_bundles=dict(
        native=deepcopy(bundle), edge=deepcopy(bundle), placement=dict(
            pixel_model=placement["pixel"], regional_model=placement["regional"],
            projected_models=dict(pixel=placement["projected_pixel"], regional=placement["projected_regional"])))))
    source = deepcopy(phase1_data)
    source["phase"]["configuration"] = phase
    config = _inputs(tmp_path, tiny_config, source)
    completion = _run(tmp_path, config)
    return config, completion, _index(tmp_path, completion)


def _displayed_review(case):
    # The UI displays the admitted catalogue reader, whose index is bound from
    # the original done record; the publisher's return descriptor is distinct.
    attachments, errors = catalogue.discover_native_reviews(
        case["registry"], case["series_root"], "ffac", current_machine_id="Main")
    assert not errors and len(attachments) == 1
    assert attachments[0]["registration"] == case["registration"]
    return attachments[0]["reader"]


@pytest.fixture
def approval_case(native_catalogue, monkeypatch):
    case = native_catalogue
    case["review"] = _displayed_review(case)
    case["model"] = case["root"].parent / "models" / "phase-model-source"
    calibration = case["completion"]["calibration"]
    case["locked"] = dict(locked=True, binding=dict(active="phase_model", chemistry_model="titration",
        model_version=str(case["model"]), mass_calibration=calibration["path"], calibration=deepcopy(calibration),
        phase_model=dict(root=str(case["model"]), identity=phase_model_library.source_identity(case["model"]))))
    def lock(root, *, verify=True):
        assert Path(root) == case["series_root"] and verify is True
        return deepcopy(case["locked"])
    monkeypatch.setattr(approval, "read_approval", lock)
    source_root = case["root"].parent / "phase-source-job"
    source_job = dict(format="darsia-alias-job-v1", pipeline="native-phase-statistics",
                     series_id="FF_AC", expected_runs=[case["run"]])
    source_registration = dict(root=str(source_root), job=_write(source_root / "job.json", source_job))
    config = descriptor(case["root"].parent / "phase1_configuration.json")
    phase = descriptor(case["root"].parent / "phase_configuration.json")
    phase_document = json.loads(Path(phase["path"]).read_bytes())
    root_aliases = dict(series=str(case["root"].parent))
    consumers = _model_consumers(phase_document)
    catalog = InputCatalog(root_aliases, aliases=[dict(logical=bound, physical=descriptor(case["model"] / relative))
                                                 for _use, bound, relative in consumers])
    configuration_reference = catalog.add(config, "configuration", "run.configuration")
    catalog.doc(phase, "configuration", "phase.configuration")
    for use, bound, _relative in consumers:
        catalog.doc(bound, "model", use)
    rows = list(catalog.entries.values())
    rows.sort(key=lambda row: catalog_order_key(row["reference"]))
    task = dict(run=case["run"], configuration=configuration_reference, input_catalog=dict(
        format=CATALOG_FORMAT, complete=True, run=case["run"], configuration=configuration_reference, inputs=rows))
    _write(source_root / "results" / case["run"] / "task.json", task)
    native_manifest = json.loads(Path(case["completion"]["manifest"]["path"]).read_bytes())
    source_completion = _write(source_root / "results" / case["run"] / "completed.json",
        dict(complete=True, run=case["run"], worker_references_closed=True,
             phase1_completion=native_manifest["storage"]["phase1"]["completion"]))
    index = products_publication.publish_run_metadata(source_root / "publication",
        job=source_registration["job"], run=case["run"], attempt_id="a" * 32,
        machine_id="Main", completed=source_completion, products=[], metadata_product_ids=[],
        file_writers_closed=True, all_final_products_listed=True)
    _write(source_root / "done" / (case["run"] + ".json"), dict(attempt_id="a" * 32,
        completed=job_queue._own_reference(source_root, source_completion["path"]),
        products_index=job_queue._own_reference(source_root, index["path"])))
    source_snapshot = dict(job_facts=source_job, errors=[], runs=[dict(run=case["run"], state="Complete")])
    def snapshot(_reader, registration):
        return deepcopy(source_snapshot if registration["root"] == str(source_root) else case["snapshot"])
    monkeypatch.setattr(JobSnapshotReader, "__call__", snapshot)
    stores = case["root"].parent / "stores.toml"
    stores.write_text('[machine]\nid = "Main"\n[stores.FF_AC]\nseries = '
        + json.dumps(str(case["root"].parent)) + '\nresults = '
        + json.dumps(str(case["root"].parent / "receiver-results")) + '\n', encoding="utf-8")
    case.update(source_registrations=[source_registration], stores_path=stores,
                source_root=source_root, source_snapshot=source_snapshot)
    return case


def _prepare(case):
    return approval.prepare_native_run_approval(case["series_root"], "ffac", case["run"],
        case["review"], case["registration"], source_registrations=case["source_registrations"],
        stores_path=case["stores_path"])


def _approve(case, expected, **kwargs):
    return approval.approve_native_run(case["series_root"], "ffac", case["run"],
        case["review"], case["registration"], expected_binding=expected,
        retry=kwargs.pop("retry", lambda action: action()), source_registrations=case["source_registrations"],
        stores_path=case["stores_path"], **kwargs)


def test_prepare_uses_original_sources_without_frames_or_writes(approval_case, monkeypatch):
    case = approval_case
    displayed_index = case["review"].index
    assert displayed_index == dict(path=case["index"]["path"], sha256=case["index"]["sha256"],
                                   size_bytes=case["index"]["bytes"])
    before = {path: path.read_bytes() for path in case["root"].parent.rglob("*") if path.is_file()}
    monkeypatch.setattr(saved_n1, "read_n1_stored_frame", lambda *a, **k: pytest.fail("Approval loaded mass arrays"))
    binding = _prepare(case)
    evidence = binding["native_evidence"]
    assert evidence["index"] == case["review"].index
    assert evidence["attempt_id"] == case["done"]["attempt_id"]
    assert evidence["series_id"] == "FF_AC" and evidence["run"] == case["run"]
    assert binding["source_access"]["model_root"] == str(case["model"].resolve())
    assert len(evidence["source_model"]["model_assets"]) == 8
    assert all(asset["path"].startswith(_PRODUCER_MODEL) for asset in evidence["source_model"]["model_assets"])
    assert evidence["series_approval_binding"] == case["locked"]["binding"]
    assert set(evidence["phase1"]) == {"configuration", "completion", "code_commit"}
    access_keys = {"stores", "physical_configuration", "physical_phase_configuration",
                   "model_root", "mapped_phase", "physical_model_assets"}
    assert set(binding["source_access"]) == access_keys
    assert not access_keys.intersection(evidence["source_model"])
    assert binding["calibration"]["sha256"] == case["review"].expected_calibration_sha256
    assert not (case["series_root"] / records.APPROVALS_PATH).exists()
    assert {path: path.read_bytes() for path in before} == before
    binding["native_evidence"]["index"]["sha256"] = "0" * 64
    assert case["review"].index == displayed_index


def test_native_append_retains_exact_evidence_and_original_reviewer_policy(approval_case):
    case = approval_case
    expected = _prepare(case)
    when = dt.datetime(2026, 10, 23, 2, tzinfo=dt.timezone.utc)
    result = _approve(case, expected, reviewer=" Reviewer A ", now=when)
    assert result["format"] == records.APPROVAL_FORMAT
    assert result["native_evidence"] == expected["native_evidence"]
    assert result["result"] == case["index"]["path"]
    assert result["reviewer"] == "Reviewer A" and result["recorded_utc"] == when.isoformat()
    existing = records.read_run_approvals(case["series_root"])
    assert records.run_approval(existing, case["run"], expected["calibration"]["sha256"],
                                native_evidence=expected["native_evidence"]) == result


def test_all_model_consumers_map_without_producer_probes_or_legacy_aliases(approval_case, monkeypatch):
    case = approval_case
    from darsia.fluidflower import review_storage
    original_phase = (case["root"].parent / "phase_configuration.json").read_bytes()
    original_stat, original_open = Path.stat, Path.open
    def admit(path):
        assert not str(path).replace("\\", "/").startswith(_PRODUCER_MODEL), "Producer model path was probed"
        return Path(path)
    def stat(path, *args, **kwargs):
        return original_stat(admit(path), *args, **kwargs)
    def open_file(path, *args, **kwargs):
        return original_open(admit(path), *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(review_storage, "resolve_review_path", admit)
    expected = _prepare(case)
    source = expected["native_evidence"]["source_model"]
    assert {row["use"] for row in source["model_consumers"]} == {
        use for use, _bound, _relative in _model_consumers(json.loads(original_phase))}
    assert all(row["reference"]["role"] == "model" for row in source["model_consumers"])
    assert all(Path(asset["path"]).is_relative_to(case["model"])
               for asset in expected["source_access"]["physical_model_assets"])
    assert (case["root"].parent / "phase_configuration.json").read_bytes() == original_phase
    result = _approve(case, expected)
    assert result["native_evidence"] == expected["native_evidence"]
    assert "source_access" not in result


@pytest.mark.parametrize("use", ["phase.model.native.co2_presence", "phase.model.edge.gas_within_co2",
    "phase.placement.pixel_model", "phase.placement.regional_model",
    "phase.placement.projected.pixel", "phase.placement.projected.regional"])
@pytest.mark.parametrize("problem", ["sha", "size", "role", "consumer", "origin"])
def test_model_consumer_mismatch_never_creates_an_approval(approval_case, use, problem):
    case = approval_case
    task_path = case["source_root"] / "results" / case["run"] / "task.json"
    task = json.loads(task_path.read_bytes())
    rows = task["input_catalog"]["inputs"]
    row = next(row for row in rows if use in row["uses"])
    reference = row["reference"]
    if problem == "sha":
        reference["sha256"] = "0" * 64
    elif problem == "size":
        reference["bytes"] += 1
    elif problem == "role":
        reference["role"] = "configuration"
    elif problem == "consumer":
        row["uses"][row["uses"].index(use)] = "not-the-original-model-consumer"
    else:
        original = case["root"].parent / reference["relative_path"]
        external = case["root"].parent / "identical-outside-selected-model.json"
        external.write_bytes(original.read_bytes())
        reference["relative_path"] = external.name
    rows.sort(key=lambda row: catalog_order_key(row["reference"]))
    _write(task_path, task)
    with pytest.raises(ValueError, match="input catalogue|input consumer|multiple model origins"):
        _prepare(case)
    assert not (case["series_root"] / records.APPROVALS_PATH).exists()


def test_model_path_and_sha_only_declarations_use_catalogued_size(approval_case):
    case = approval_case
    from darsia.fluidflower.store_map import read_store_map
    from darsia.fluidflower.raw_production_review import _origin
    task = json.loads((case["source_root"] / "results" / case["run"] / "task.json").read_bytes())
    phase = json.loads((case["root"].parent / "phase_configuration.json").read_bytes())
    for _use, bound, _relative in _model_consumers(phase):
        bound.pop("size_bytes")
    before = deepcopy(phase)
    reader = approval._ApprovalSnapshotReader()
    mapped, original, consumers = approval._mapped_model_phase(
        task, phase, read_store_map(case["stores_path"]), "FF_AC", reader)
    model, _identity, physical = _origin(case["model"], mapped)
    reader.finish()
    assert model == case["model"] and phase == before
    assert len(original) == len(consumers) == len(physical) == 8
    assert all("size_bytes" not in bound for bound in original)
    assert all(type(bound["size_bytes"]) is int for bound in physical)


def test_undeclared_model_consumer_refuses_before_any_asset_probe(approval_case, monkeypatch):
    case = approval_case
    from darsia.fluidflower.store_map import read_store_map
    task = json.loads((case["source_root"] / "results" / case["run"] / "task.json").read_bytes())
    phase = json.loads((case["root"].parent / "phase_configuration.json").read_bytes())
    phase["model_bundles"]["unconsumed"] = dict(path=_PRODUCER_MODEL + "/extra.json", sha256="0" * 64)
    stores = read_store_map(case["stores_path"])
    monkeypatch.setattr(Path, "stat", lambda *_a, **_k: pytest.fail("Unmapped model caused a file probe"))
    with pytest.raises(ValueError, match="no original named input consumer"):
        approval._mapped_model_phase(task, phase, stores, "FF_AC", approval._ApprovalSnapshotReader())


def test_native_phase_approval_refuses_absent_original_source_contract(native_phase_attachment, tmp_path, monkeypatch):
    value = native_phase_attachment
    review = approval.NativeSavedReview(value["index"], current_machine_id="Main",
        expected_run=value["completion"]["run"],
        expected_calibration_sha256=value["completion"]["calibration"]["sha256"],
        expected_pipeline="native-phase-n1")
    assert review.pipeline == "native-phase-n1"
    series_root = tmp_path / "unopened-series"
    monkeypatch.setattr(approval, "_metadata_descriptor", lambda *_a, **_k: pytest.fail("Unsupported approval read source metadata"))
    monkeypatch.setattr(approval, "read_approval", lambda *_a, **_k: pytest.fail("Unsupported approval read series lock"))
    with pytest.raises(ValueError, match="no original N1 phase-one source approval contract"):
        approval.prepare_native_run_approval(series_root, "ffac", review.expected_run, review, {})
    assert not series_root.exists()


@pytest.mark.parametrize("source", ["job", "task"])
def test_approval_job_and_task_must_match_displayed_protocol(approval_case, source):
    case = approval_case
    if source == "job":
        case["job"]["pipeline"] = "native-phase-n1"
        case["registration"]["job"] = _write(case["root"] / "job.json", case["job"])
        case["snapshot"]["job_facts"] = deepcopy(case["job"])
    else:
        task_path = case["root"] / "results" / case["run"] / "task.json"
        task = json.loads(task_path.read_bytes())
        task["pipeline"] = "native-phase-n1"
        _write(task_path, task)
    with pytest.raises(ValueError, match="Original native job differs|Current native task or done record differs"):
        _prepare(case)
    assert not (case["series_root"] / records.APPROVALS_PATH).exists()


def test_mapped_model_bytes_are_rechecked_immediately_before_append(approval_case):
    case = approval_case
    expected = _prepare(case)
    def delayed_retry(action):
        path = case["model"] / "models/generic/co2_presence_signal.json"
        original = path.read_bytes()
        changed = original.replace(b"1.0", b"2.0")
        assert changed != original and len(changed) == len(original)
        path.write_bytes(changed)
        return action()
    with pytest.raises(ValueError, match="Metadata bytes differ"):
        _approve(case, expected, retry=delayed_retry)
    assert not (case["series_root"] / records.APPROVALS_PATH).exists()


def test_legacy_and_other_native_result_cannot_certify_this_publication(tmp_path):
    calibration = dict(reference="frozen.json", sha256="a" * 64)
    retry = lambda action: action()
    evidence = dict(format=approval.EVIDENCE_FORMAT, index=dict(path="same-index.json", sha256="b" * 64))
    records.approve_run(tmp_path, "run", calibration, result="same-index.json", retry=retry)
    assert records.run_approval(records.read_run_approvals(tmp_path), "run", "a" * 64,
                                native_evidence=evidence) is None
    first = records.approve_run(tmp_path, "run", calibration, result="same-index.json",
                                native_evidence=evidence, retry=retry)
    changed = deepcopy(evidence)
    changed["index"]["sha256"] = "c" * 64
    records.approve_run(tmp_path, "run", calibration, result="same-index.json",
                        native_evidence=changed, retry=retry)
    assert records.run_approval(records.read_run_approvals(tmp_path), "run", "a" * 64,
                                native_evidence=evidence) == first
    assert records.run_approval(records.read_run_approvals(tmp_path), "run", "a" * 64)["native_evidence"] == changed


@pytest.mark.parametrize("drift", ["series", "calibration", "model", "done", "shared_metadata",
                                  "source_task", "source_done", "source_complete", "stores"])
def test_changed_review_sources_refuse_without_creating_a_record(approval_case, drift):
    case = approval_case
    expected = _prepare(case)
    if drift == "series":
        target = case["series_root"] / "series.toml"
    elif drift == "calibration":
        case["locked"]["binding"]["calibration"]["sha256"] = "0" * 64
        target = None
    elif drift == "model":
        target = case["model"] / "models/generic/co2_presence_signal.json"
    elif drift == "done":
        target = case["root"] / "done" / (case["run"] + ".json")
    elif drift == "source_task":
        target = case["source_root"] / "results" / case["run"] / "task.json"
    elif drift == "source_done":
        target = case["source_root"] / "done" / (case["run"] + ".json")
    elif drift == "source_complete":
        case["source_snapshot"]["runs"][0]["state"] = "Publishing"
        target = None
    elif drift == "stores":
        target = case["stores_path"]
    else:
        index_root, _document, products = approval.publication._read_index(case["index"])
        target = Path(index_root) / products["manifest.json"]["metadata_copy"]["relative_path"]
    if target is not None:
        target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises((ValueError, OSError)):
        _approve(case, expected)
    assert not (case["series_root"] / records.APPROVALS_PATH).exists()


def test_stale_action_is_checked_before_sources_and_again_before_each_append(approval_case):
    case = approval_case
    expected = _prepare(case)
    with pytest.raises(ValueError, match="displayed native review changed"):
        _approve(case, expected, action_guard=lambda: False)
    calls = []
    def action_current():
        calls.append(True)
        return len(calls) < 3  # First source check passes; append-time guard fails.
    with pytest.raises(ValueError, match="displayed native review changed"):
        _approve(case, expected, action_guard=action_current)
    assert len(calls) == 3
    assert not (case["series_root"] / records.APPROVALS_PATH).exists()


def test_retry_cannot_append_after_captured_sources_change(approval_case):
    case = approval_case
    expected = _prepare(case)
    def delayed_retry(action):
        path = case["series_root"] / "series.toml"
        path.write_bytes(path.read_bytes() + b"\n# changed while awaiting append\n")
        return action()
    with pytest.raises(ValueError, match="changed since it was shown"):
        _approve(case, expected, retry=delayed_retry)
    assert not (case["series_root"] / records.APPROVALS_PATH).exists()


def test_original_writer_runs_guard_at_every_retry_and_detaches_evidence(tmp_path):
    evidence = dict(format=approval.EVIDENCE_FORMAT, values=[1])
    calls = []
    def guard():
        calls.append(True)
        if len(calls) == 1:
            raise OSError("Transient guard read")
    def retry(action):
        try:
            action()
        except OSError:
            return action()
    result = records.approve_run(tmp_path, "run", dict(reference="frozen", sha256="a" * 64),
        native_evidence=evidence, before_append=guard, retry=retry)
    evidence["values"].append(2)
    assert calls == [True, True] and result["native_evidence"]["values"] == [1]
    assert len(records.read_run_approvals(tmp_path)) == 1


@pytest.mark.parametrize("problem", ["unlocked", "missing_registration", "wrong_run", "wrong_series", "incomplete", "error"])
def test_missing_or_inconsistent_current_evidence_never_enables_approval(approval_case, problem):
    case = approval_case
    run, series, registration = case["run"], "ffac", case["registration"]
    if problem == "unlocked":
        case["locked"]["locked"] = False
    elif problem == "missing_registration":
        registration = None
    elif problem == "wrong_run":
        run = "another-run"
    elif problem == "wrong_series":
        series = "another-series"
    elif problem == "incomplete":
        case["snapshot"]["runs"][0]["state"] = "Publishing"
    else:
        case["snapshot"]["errors"] = ["Original owner changed"]
    with pytest.raises(ValueError):
        approval.prepare_native_run_approval(case["series_root"], series, run, case["review"], registration)


def test_original_series_lock_is_required_even_when_native_freeze_is_readable(native_catalogue):
    case = native_catalogue
    review = _displayed_review(case)
    with pytest.raises(ValueError, match="Use and lock"):
        approval.prepare_native_run_approval(case["series_root"], "ffac", case["run"], review, case["registration"])
    # Catalogue admission covers BOM input; the original model-library path
    # reader used below requires plain UTF-8 TOML. Retain the same series data.
    manifest = case["series_root"] / "series.toml"
    manifest.write_bytes(manifest.read_bytes().removeprefix(b"\xef\xbb\xbf"))
    choice = bound_products(case["series_root"])
    series_calibration_approval.approve_series_method(choice.root, choice.selection)
    with pytest.raises(ValueError, match="locked calibration"):
        approval.prepare_native_run_approval(case["series_root"], "ffac", case["run"], review, case["registration"])


def test_original_model_origin_refuses_an_identical_external_head(approval_case):
    case = approval_case
    # Re-sealing the source configuration would invalidate N1's old descriptor.
    # Exercise the original origin boundary directly with the declared source.
    phase = _prepare(case)["source_access"]["mapped_phase"]
    head = phase["model_bundles"]["native"]["documents"]["co2_presence"]
    external = case["root"].parent / "same_head_elsewhere.json"
    external.write_bytes(Path(head["path"]).read_bytes())
    head.update(descriptor(external))
    from darsia.fluidflower.raw_production_review import _origin
    with pytest.raises(ValueError, match="multiple model origins"):
        _origin(case["model"], phase)


def test_source_reader_finish_detects_same_stat_content_edit(tmp_path):
    import os
    path = tmp_path / "metadata.json"
    original = _write(path, {"value": 1})
    reader = approval._ApprovalSnapshotReader()
    assert reader.read(path, original) == {"value": 1}
    info = path.stat()
    path.write_bytes(path.read_bytes().replace(b"1", b"2"))
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(ValueError, match="Metadata bytes differ"):
        reader.finish()


def test_source_reader_normalizes_original_descriptor_byte_count_vocabularies(tmp_path):
    path = tmp_path / "metadata.json"
    bound = _write(path, dict(value=1))
    reader = approval._ApprovalSnapshotReader()
    assert reader.read(path, bound) == reader.read(path, dict(
        path=bound["path"], sha256=bound["sha256"], bytes=bound["size_bytes"]))
    assert len(reader.observed) == 1
    reader.finish()


def test_source_registration_is_required_before_any_source_store_or_literal_probe(approval_case, monkeypatch):
    case = approval_case
    from darsia.fluidflower import store_map
    monkeypatch.setattr(store_map, "read_store_map", lambda *a, **k: pytest.fail("No source registration may probe stores"))
    case["source_registrations"] = []
    with pytest.raises(ValueError, match="exact registered complete phase-one source"):
        _prepare(case)


def test_duplicate_exact_source_registration_collapses_but_incomplete_source_refuses(approval_case):
    case = approval_case
    expected = _prepare(case)
    case["source_registrations"] *= 2
    assert _prepare(case) == expected
    case["source_snapshot"]["runs"][0]["state"] = "Publishing"
    with pytest.raises(ValueError, match="exact registered complete phase-one source"):
        _prepare(case)


def test_unrelated_pending_source_needs_no_result_path_and_does_not_bind_progress(approval_case, monkeypatch):
    case = approval_case
    expected = _prepare(case)
    pending_root = case["root"].parent / "pending-phase-source"
    pending_job = dict(format="darsia-alias-job-v1", pipeline="native-phase-statistics",
                       series_id="FF_AC", expected_runs=[case["run"]])
    pending = dict(root=str(pending_root), job=_write(pending_root / "job.json", pending_job))
    original_snapshot = JobSnapshotReader.__call__
    def snapshot(reader, registration):
        if registration["root"] == str(pending_root):
            return dict(job_facts=pending_job, errors=[], runs=[dict(run=case["run"], state="Not queued")])
        return original_snapshot(reader, registration)
    monkeypatch.setattr(JobSnapshotReader, "__call__", snapshot)
    case["source_registrations"].insert(0, pending)
    assert not (pending_root / "results").exists()
    assert _prepare(case) == expected


@pytest.mark.parametrize("operation", ["initial", "expected", "finish"])
def test_metadata_limit_is_checked_before_unbounded_queue_reads(tmp_path, monkeypatch, operation):
    path = tmp_path / "metadata.json"
    bound = _write(path, dict(value=1))
    reader = approval._ApprovalSnapshotReader()
    if operation == "finish":
        reader.read(path, bound)
    monkeypatch.setattr(approval.publication, "MAX_METADATA_BYTES", 32)
    path.write_bytes(b" " * 33)
    monkeypatch.setattr(Path, "read_bytes", lambda *_a, **_k: pytest.fail("Unbounded metadata read"))
    with pytest.raises(ValueError, match="bounded read"):
        if operation == "finish":
            reader.finish()
        else:
            reader.read(path, bound if operation == "expected" else None)


def test_series_manifest_is_bounded_before_original_selection_read(approval_case, monkeypatch):
    case = approval_case
    monkeypatch.setattr(approval, "_SERIES_MANIFEST_LIMIT", 16)
    monkeypatch.setattr(approval, "_registered_series_id", lambda *_a: pytest.fail("Oversized series metadata read"))
    with pytest.raises(ValueError, match="bounded read"):
        _prepare(case)


def test_receiving_aliases_guard_action_but_do_not_change_persisted_identity(approval_case):
    case = approval_case
    expected = _prepare(case)
    stores_copy = case["stores_path"].with_name("other-stores.toml")
    stores_copy.write_bytes(case["stores_path"].read_bytes())
    case["stores_path"] = stores_copy
    remapped = _prepare(case)
    assert remapped["native_evidence"] == expected["native_evidence"]
    assert remapped["source_access"] != expected["source_access"]
    with pytest.raises(ValueError, match="changed since it was shown"):
        _approve(case, expected)


def test_named_source_mapping_never_opens_foreign_same_drive_literal(approval_case, monkeypatch):
    case = approval_case
    from darsia.fluidflower.store_map import read_store_map
    stores = read_store_map(case["stores_path"])
    source_task = json.loads((case["source_root"] / "results" / case["run"] / "task.json").read_bytes())
    physical = case["root"].parent / "phase_configuration.json"
    original = dict(descriptor(physical), path="Z:/foreign-producer/phase_configuration.json")
    stat = Path.stat
    def reject_literal(path, *args, **kwargs):
        assert str(path).replace("\\", "/") != original["path"], "Foreign source literal was probed"
        return stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", reject_literal)
    bound, document = approval._mapped_metadata(source_task, original, "phase.configuration", stores,
                                                "FF_AC", approval._ApprovalSnapshotReader())
    assert Path(bound["path"]) == physical and "model_bundles" in document


def test_run_configuration_consumer_cannot_substitute_an_identical_different_reference(approval_case):
    case = approval_case
    from darsia.fluidflower.store_map import read_store_map
    stores = read_store_map(case["stores_path"])
    task = json.loads((case["source_root"] / "results" / case["run"] / "task.json").read_bytes())
    original = descriptor(case["root"].parent / "phase1_configuration.json")
    declared = next(row for row in task["input_catalog"]["inputs"] if row["uses"] == ["run.configuration"])
    alternate = deepcopy(declared)
    alternate["reference"]["relative_path"] = "identical-config-at-another-path.json"
    declared["uses"] = ["retained.task.configuration"]
    task["input_catalog"]["inputs"].append(alternate)
    task["input_catalog"]["inputs"].sort(key=lambda row: catalog_order_key(row["reference"]))
    with pytest.raises(ValueError, match="run.configuration consumer differs"):
        approval._mapped_metadata(task, original, "run.configuration", stores,
                                  "FF_AC", approval._ApprovalSnapshotReader())


def test_missing_named_consumer_and_wrong_machine_refuse_before_source_reads(approval_case):
    case = approval_case
    task_path = case["source_root"] / "results" / case["run"] / "task.json"
    task = json.loads(task_path.read_bytes())
    for row in task["input_catalog"]["inputs"]:
        if row["uses"] == ["phase.configuration"]:
            row["uses"] = ["not-the-phase-configuration"]
    _write(task_path, task)
    with pytest.raises(ValueError, match="No unique original source input consumer"):
        _prepare(case)
    case["stores_path"].write_text(case["stores_path"].read_text().replace('id = "Main"', 'id = "Second"'))
    with pytest.raises(ValueError, match="another physical machine"):
        _prepare(case)
