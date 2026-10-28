"""The final typed Setup task uses its own ordinary queue-attempt namespace.

Only private metadata is claimed here. Geometry done files are scheduling
hints, not scientific completions; no image executor or RAW decoder runs.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import toml

from darsia.fluidflower import image_normalisation, job_filesystem
from darsia.fluidflower.pipeline import job_queue, publication_runtime, store_runtime
from darsia.fluidflower.pipeline import setup_image_preparation as subject
from .test_series_setup_jobs import setup_job, controls, _prepare  # noqa: F401
from .test_series_setup_inputs import _doc
from .test_series_setup_queue_runtime import _queue_context, SyntheticServerClock


def test_image_scope_adds_only_its_exact_standard_attempt_namespace(tmp_path):
    roots = [tmp_path / "job", tmp_path / "results/setup-job/setup-apply/.attempts",
             tmp_path / "results/setup-job/sample/.attempts"]
    plan = dict(output_scope=dict(metadata_roots=list(map(str, roots)), scratch_roots=[]))
    original = deepcopy(plan)
    expected = tmp_path / "results/setup-job/setup-image-preparation/.attempts"
    assert subject.image_output_scope(plan) == dict(
        metadata_roots=[*map(str, roots), str(expected)], scratch_roots=[])
    assert plan == original
    plan["output_scope"]["metadata_roots"].append(str(tmp_path / "results"))
    assert subject.image_output_scope(plan) == plan["output_scope"]


def test_actual_typed_final_claim_and_failed_retirement_with_separate_results(setup_job, tmp_path, tmp_path_factory, monkeypatch):
    # Keep the real admitted queue/writer/ledger seams used by other queue tests.
    context = _queue_context(tmp_path_factory.mktemp("image-service"), monkeypatch)
    case = setup_job
    source = Path(case.plan["series_manifest"]["path"])
    document = dict(schema_version=1, series_id="synthetic", method=dict(active="phase_model"),
        runs=dict(ids=case.plan["runs"]), roots=dict(series=".", private=str(case.root)))
    document["image_normalisation"] = image_normalisation.choice_record("synthetic", case.plan["runs"],
        "off", None, color_method="off")
    source.write_text(toml.dumps(document), encoding="utf-8")
    case.plan["series_manifest"] = job_queue.descriptor(source)
    case.plan["control_root"] = str(case.job_root / "setup-controls")
    results = tmp_path / "separate-results"
    scopes = [case.job_root, results / "setup-job/setup-apply/.attempts",
              results / "setup-job/sample/.attempts"]
    for path in scopes:
        path.mkdir(parents=True, exist_ok=True)
    case.plan["output_scope"] = dict(metadata_roots=list(map(str, scopes)), scratch_roots=[])
    for build in case.plan["build_plans"]:
        build["output_scope"] = deepcopy(case.plan["output_scope"])
        build["request_template"]["series_context"]["manifest"] = deepcopy(case.plan["series_manifest"])
        build["request_template"]["output"].update(maximum_output_bytes=8 * 1024**2,
                                                   maximum_file_bytes=2 * 1024**2)
    case.bound_plan = _doc(case.plan_path, case.plan)
    _prepare(case)  # Actually calls configure_setup_image_preparation_job.
    job = job_queue.read_json(case.job_root / "job.json")
    config = job_queue.read_json(case.job_root / "setup-controls/image-operation.json")
    image_attempts = results / "setup-job/setup-image-preparation/.attempts"
    assert config["output_scope"]["metadata_roots"] == [*map(str, scopes), str(image_attempts)]
    assert image_attempts.is_dir() and not list(image_attempts.iterdir())
    assert case.plan["output_scope"]["metadata_roots"] == list(map(str, scopes))

    # Bind the real produced catalogue to the ordinary local results mapping.
    runtime = context.runtime
    runtime.root, runtime.job, runtime.commit = case.job_root, job, job["commit"]
    runtime.stores.machine_id = config["target_machine"]
    runtime.service_id = config["target_machine"] + "-cpu"
    from darsia.fluidflower.pipeline.alias_job import _manifest
    _source, _manifest_value, roots = _manifest(config["series_manifest"])
    runtime.stores.roots = lambda _series: dict({key: str(value) for key, value in roots.items()},
                                               results=str(results))
    proof = job_filesystem.LocalFileAdmission(True, "synthetic-fixed-local", "private test observation",
        "Not a production filesystem qualification", str(results))
    monkeypatch.setattr(store_runtime, "inspect_results_root", lambda _root: proof)
    owner = dict(context.owner, machine=runtime.service_id)
    runtime.cache_adapter.owner = owner
    queue = job_queue.JobQueue(case.job_root, machine_id=runtime.stores.machine_id,
                               server_clock=SyntheticServerClock())
    publisher = publication_runtime.make_publication_factory(case.job_root, job, runtime.stores.machine_id,
        producer_origin_for_attempt=lambda _attempt: pytest.fail("No scientific result is published"))

    def claim_final():
        with runtime.capacity_claim(owner) as candidate:
            claim = queue.claim(runtime.service_id, runtime.backend, owner,
                availability=candidate.availability, output_factory=runtime.output_factory,
                publication_factory=publisher, claim_controls=candidate.claim_controls,
                exclude_runs=("setup-apply",))
            candidate.bind(claim)
        return claim

    assert claim_final() is None  # The final task waits for every Build.
    pending = case.job_root / "queue" / (subject.IMAGE_RUN + ".json")
    assert pending.is_file(), list(case.job_root.rglob("*quarantine*"))
    for run in case.plan["runs"]:
        (case.job_root / "done" / (run + ".json")).write_text("{}")
    assert runtime.availability(job_queue.read_json(pending)) is not None
    claim = claim_final()
    assert claim is not None, runtime.capacity_refusals
    assert claim.task["run"] == subject.IMAGE_RUN
    assert claim.output == image_attempts / claim.token
    assert claim.output.is_dir()
    assert job_queue.read_json(claim.lease / "owner.json")["actual_owner"] == owner
    helper = runtime.setup_admissions[claim.token]
    assert {domain.path for domain in helper.admission._domains} == {case.job_root, image_attempts}
    with runtime.output_scope(claim):
        helper.sink.write_bytes(claim.output / "private-metadata.json", b"{}")
        with pytest.raises(ValueError, match="outside"):
            helper.sink.write_bytes(scopes[-1] / "another-run.json", b"{}")
        queue.fail(claim, "Synthetic image producer failure after a genuine bounded claim", owner)
    runtime.release(claim, retain_capacity=True)
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in helper.admission.bound_reservations)
    from darsia.fluidflower.pipeline.checkpoint import retirement_document
    from darsia.fluidflower.pipeline.series_setup_worker_admission import retirement_control_writer
    from darsia.fluidflower.pipeline.capacity import release_retired_capacity
    retired = retirement_document(claim.task, claim.output, [deepcopy(owner)],
        dict(owner, pid=999, process_create_time="999.125"))
    owned = job_queue.read_json(claim.lease / "owner.json")
    writer = retirement_control_writer(case.job_root, claim.task, runtime.stores, claim.output, owned)
    with writer.activate():
        job_queue.write_immutable(claim.output / "retirement.json", retired)
    writer.close()
    reserved = Path(helper.admission.bound_reservations[0]["path"]) / "reserved.json"
    original_reservation = reserved.read_bytes()
    changed = job_queue.read_json(reserved)
    changed["estimate"]["roots"] = [item for item in changed["estimate"]["roots"]
                                    if item["path"] != str(case.job_root)]
    try:
        reserved.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="exact operation controls"):
            release_retired_capacity(case.job_root, claim.task, runtime.stores, retired)
        assert all(not (Path(row["path"]) / "released.json").exists()
                   for row in helper.admission.bound_reservations)
    finally:
        reserved.write_bytes(original_reservation)
    released = release_retired_capacity(case.job_root, claim.task, runtime.stores, retired)
    assert set(released) == {row["path"] for row in helper.admission.bound_reservations}
    assert released and all((Path(path) / "released.json").is_file() for path in released)
    assert (case.job_root / "failed" / (subject.IMAGE_RUN + ".json")).is_file()
    assert (claim.output / "private-metadata.json").read_bytes() == b"{}"


def test_image_namespace_cannot_overlap_a_protected_acquisition(setup_job):
    case = setup_job
    apply = case.root / "separate-results/setup-job/setup-apply/.attempts"
    image_attempts = apply.parent.parent / subject.IMAGE_RUN / ".attempts"
    case.plan["output_scope"]["metadata_roots"].append(str(apply))
    source = Path(case.plan["series_manifest"]["path"])
    document = dict(schema_version=1, series_id="synthetic", method=dict(active="phase_model"),
        runs=dict(ids=case.plan["runs"]), roots=dict(series=".", private=str(case.root)))
    document["image_normalisation"] = image_normalisation.choice_record("synthetic", case.plan["runs"],
        "off", None, color_method="off")
    source.write_text(toml.dumps(document), encoding="utf-8")
    case.plan["series_manifest"] = job_queue.descriptor(source)
    for build in case.plan["build_plans"]:
        build["request_template"]["series_context"]["manifest"] = deepcopy(case.plan["series_manifest"])
        build["request_template"]["output"]["forbidden_raw_roots"].append(str(image_attempts))
    case.bound_plan = _doc(case.plan_path, case.plan)
    with pytest.raises(ValueError, match="protected RAW"):
        _prepare(case)
    assert not image_attempts.exists()
