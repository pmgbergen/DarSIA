"""Synthetic queue metadata only; no executor, scientific payload or network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import job_queue as jobs
from darsia.fluidflower.pipeline import products_publication, server_clock
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput


class FakeServerClock:
    """Explicit server-time fixture; client wall clock has no authority."""
    def __init__(self):
        self.now = 10_000_000_000
        self.timestamps = {}

    def stamp(self, path):
        return self.timestamps.setdefault(str(path), self.now)

    def now_ns(self):
        return self.now

    def stale(self, path, timeout_seconds=600):
        return self.now - self.stamp(path) >= timeout_seconds * 1_000_000_000


def fixture(tmp_path, monkeypatch, *, ready=True, local_results_directory=None):
    clock = FakeServerClock()
    monkeypatch.setattr(server_clock, "file_mtime_ns", clock.stamp)
    queue = jobs.JobQueue(tmp_path / "shared", server_clock=clock,
                          clock=lambda: -999999999999, machine_id="Main")
    job = dict(format="darsia-alias-job-v1", execution_ready=ready,
               runtime_support="darsia-role-io-runtime-v1",
               expected_runs=["ac52"], series_id="Synthetic")
    jobs.write_immutable(queue.root / "job.json", job)
    raw = b'{"synthetic":true}'
    source = tmp_path / "input.json"
    source.write_bytes(raw)
    reference = dict(root_alias="series", relative_path="input.json",
                     sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), role="configuration")
    task = dict(format="darsia-alias-task-v1", format_version=1, legacy_paths=False,
                execution_ready=ready, run="ac52", job_id="synthetic-job", commit="a" * 40,
                calibration_sha256="b" * 64, configuration=reference, series=reference,
                series_manifest=reference, ordered_case_ids=["ac52_f1"], backend="cpu",
                precision="float32", input_catalog=dict(format="darsia-role-input-catalog-v1",
                    complete=True, run="ac52", configuration=reference,
                    runtime_support="darsia-role-io-runtime-v1",
                    inputs=[dict(reference=reference, uses=["run.configuration", "run.execution_series", "run.series_manifest"])]))
    queue.enqueue(task)
    events = []

    def availability(value):
        assert (queue.root / "queue" / "ac52.json").exists()
        assert not list((queue.root / "claimed").glob("*/*.json"))
        events.append("available_before_rename")
        return tuple(ResolvedInput(deepcopy(item["reference"]), source, tuple(item["uses"]))
                     for item in value["input_catalog"]["inputs"])

    def output_factory(value, token):
        events.append("owned_output_after_rename")
        result_base = tmp_path if local_results_directory is None else local_results_directory
        root = result_base / "local-results"
        output = root / value["job_id"] / value["run"] / ".attempts" / token
        proof = job_filesystem.LocalFileAdmission(True, "synthetic-local", "fixture only",
                                                   "No actual filesystem qualification claimed", str(result_base))
        return jobs.LocalOutputBinding("Main", root, output, proof)

    def publication_factory(claim, candidate, success):
        assert success is True
        events.append("shared_copy_before_fence")
        bundle = claim.lease / "artifacts"
        index = products_publication.publish_run_metadata(bundle,
            job=jobs.descriptor(queue.root / "job.json"), run=claim.task["run"],
            attempt_id=claim.token, machine_id="Main", completed=candidate,
            products=[], metadata_product_ids=[], file_writers_closed=True,
            all_final_products_listed=True)
        return dict(candidate=jobs._own_reference(claim.lease, bundle / "completed.json"),
                    products_index=jobs._own_reference(claim.lease, Path(index["path"])))

    def publish_progress(claim, event):
        record = event["publication"]
        original = record["files"]["commit.json"]
        source = Path(record["nas_directory"]) / "commit.json"
        target = claim.lease / "progress_artifacts" / event["case_id"] / "commit.json"
        target.parent.mkdir(parents=True)
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == original["sha256"]
        target.write_bytes(raw)
        return jobs._own_reference(claim.lease, target)

    publication_factory.publish_progress = publish_progress
    callbacks = dict(availability=availability, output_factory=output_factory,
                     publication_factory=publication_factory)
    return queue, clock, callbacks, events


def claim_fixture(queue, clock, callbacks):
    claim = queue.claim("Main", "cpu", {"pid": 123}, **callbacks)
    clock.stamp(claim.lease / "owner.json")
    return claim


def completion():
    return dict(complete=True, run="ac52", worker_references_closed=True,
                records=[dict(nas_directory="C:/actual-main/old-attempt/frame1")],
                precision="float32", scalar=23.859055832362177)


def test_alias_failed_publication_decision_keeps_success_bundle_and_original_failure_without_rewrite(tmp_path, monkeypatch):
    queue, clock, callbacks, events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    decisions = []
    def blocked(*args, **kwargs):
        decisions.append(args[3])
        raise PermissionError("Original publication decision refused")
    monkeypatch.setattr(jobs, "decide", blocked)  # The ownership transition fails.
    with pytest.raises(PermissionError, match="Original publication decision refused"):
        queue.complete(claim, completion())
    publication = claim.lease / "publication.json"
    assert not jobs.decision_path(queue.root, "ac52", claim.token).exists()
    assert not (queue.root / "publishing/ac52").exists()
    before = {p.relative_to(claim.lease): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in claim.lease.rglob("*") if p.is_file()}
    factory_calls = events.count("shared_copy_before_fence")
    queue.fail(claim, "Original publication decision refused; do not replace success publication", {})
    assert decisions == ["publish"] and events.count("shared_copy_before_fence") == factory_calls
    assert {p: ((claim.lease / p).read_bytes(), (claim.lease / p).stat().st_mtime_ns)
            for p in before} == before
    assert jobs.read_json(publication)["success"] is True
    assert "Original publication decision refused" in jobs.read_json(claim.output / "failed.json")["traceback"]
    assert not (queue.root / "done/ac52.json").exists() and not (queue.root / "failed/ac52.json").exists()
    assert not jobs.decision_path(queue.root, "ac52", claim.token).exists()
    assert not (queue.root / "publishing/ac52").exists()


def test_existing_foreign_success_publication_is_not_a_failure_bypass(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    monkeypatch.setattr(jobs, "decide",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("decision refused")))
    with pytest.raises(PermissionError): queue.complete(claim, completion())
    path = claim.lease / "publication.json"
    document = jobs.read_json(path); document["attempt_id"] = "different-attempt"
    path.write_text(json.dumps(document))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="another attempt"):
        queue.fail(claim, "original failure", {})
    assert path.read_bytes() == before and not (queue.root / "failed/ac52.json").exists()


def test_pending_j2_and_missing_runtime_callbacks_never_claim(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch, ready=False)
    with pytest.raises(RuntimeError, match="not explicitly ready"):
        queue.claim("Main", "cpu", {}, **callbacks)
    assert (queue.root / "queue/ac52.json").exists()
    assert not list((queue.root / "claimed").glob("*/*.json"))


def test_ready_metadata_without_concrete_runtime_bindings_is_not_boolean_bypassed(tmp_path, monkeypatch):
    queue, _clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="requires availability"):
        queue.claim("Main", "cpu", {})
    with pytest.raises(ValueError, match="boolean admission"):
        queue.claim("Main", "cpu", {}, **dict(callbacks, availability=lambda _: True))
    assert queue.claim("Main", "cpu", {}, **dict(callbacks, availability=lambda _: None)) is None
    assert (queue.root / "queue/ac52.json").exists()


def test_claim_uses_positive_local_output_and_portable_owner_metadata(tmp_path, monkeypatch):
    queue, clock, callbacks, events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    assert events[:2] == ["available_before_rename", "owned_output_after_rename"]
    assert claim.output == claim.local_output.path
    assert len(claim.input_rows) == 1
    owner = jobs.read_json(claim.lease / "owner.json")
    assert "output_path" not in owner and "claimed_path" not in owner
    assert owner["output"] == dict(machine_id="Main", root_alias="results",
        relative_path="synthetic-job/ac52/.attempts/" + claim.token)
    assert owner["claimed"]["relative_path"] == "claimed/main/ac52.json"
    assert str(claim.output) not in json.dumps(owner)


def test_arbitrary_local_path_without_admission_is_refused_before_science(tmp_path, monkeypatch):
    queue, _clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="positive configured result-root"):
        queue.claim("Main", "cpu", {}, **dict(callbacks, output_factory=lambda *_: tmp_path / "arbitrary"))
    assert not (tmp_path / "arbitrary").exists()
    assert not list((queue.root / "done").glob("*.json"))


def test_changed_generation_is_not_combined_with_old_input_rows(tmp_path, monkeypatch):
    queue, _clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    original_rename = jobs.rename
    changed = [False]
    def change_before_rename(source, destination):
        if source.parent.name == "queue" and not changed[0]:
            value = jobs.read_json(source)
            value["attempt_id"] = "f" * 32
            source.write_text(json.dumps(value), encoding="utf-8")
            changed[0] = True
        original_rename(source, destination)
    monkeypatch.setattr(jobs, "rename", change_before_rename)
    assert queue.claim("Main", "cpu", {}, **callbacks) is None
    assert jobs.read_json(queue.root / "queue/ac52.json")["attempt_id"] == "f" * 32
    assert not list((queue.root / "leases").iterdir())


def committed_event(claim, *, case="ac52_f1"):
    directory = claim.output / "frames" / case
    directory.mkdir(parents=True)
    raw = (json.dumps(dict(complete=True, frame_id=case, manifest_sha256="a" * 64,
                          files={"result.json": {"sha256": "b" * 64, "size_bytes": 3}}),
                      indent=2) + "\r\n").encode()
    (directory / "commit.json").write_bytes(raw)
    record = dict(frame_id=case, nas_directory=str(directory), manifest_sha256="a" * 64,
        files={"result.json": {"sha256": "b" * 64, "size_bytes": 3},
               "commit.json": {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}})
    return dict(state="frame_completed", run="ac52", case_id=case, frame_index=1,
                complete=True, compute_seconds=12.0, publication=record)


def test_shared_progress_binds_original_commit_without_needing_producer_local_read(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    event = committed_event(claim)
    queue.progress(claim, event)
    shared = jobs.read_json(claim.lease / "progress_00000000.json")
    assert shared["event"] == event  # Exact original event, including passive true origin.
    bound = shared["published_frame"]
    copied = jobs._own_asset(claim.lease, bound)
    expected = Path(event["publication"]["nas_directory"]) / "commit.json"
    assert copied.read_bytes() == expected.read_bytes()
    expected.rename(expected.with_name("producer_closed_elsewhere.json"))
    assert jobs.read_json(jobs._own_asset(claim.lease, bound))["frame_id"] == "ac52_f1"
    assert jobs.read_json(claim.output / "progress_00000000.json")["event"] == event


def test_server_file_time_alone_controls_alias_expiry_without_foreign_output_read(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    owner_path = claim.lease / "owner.json"
    owner = jobs.read_json(owner_path)
    owner["claimed_unix_ns"] = -1  # Embedded client/provenance value is ignored.
    owner["output_path"] = "C:/foreign-machine/must-not-open"
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    original_glob = Path.glob
    def no_foreign(path, pattern):
        assert "foreign-machine" not in str(path)
        return original_glob(path, pattern)
    monkeypatch.setattr(Path, "glob", no_foreign)
    clock.now += 599 * 1_000_000_000
    assert queue.reclaim_stale() == []
    clock.now += 1_000_000_000
    records = queue.reclaim_stale()
    assert len(records) == 1 and records[0]["previous_attempt"] == claim.token
    assert "interrupted_claim_before_science" not in records[0]


def test_old_queued_mtime_never_expires_new_ownerless_claim_early(tmp_path, monkeypatch):
    queue, clock, _callbacks, _events = fixture(tmp_path, monkeypatch)
    pending = queue.root / "queue/ac52.json"
    task = jobs.read_json(pending)
    claimed = queue.root / "claimed/main/ac52.json"
    claimed.parent.mkdir()
    jobs.rename(pending, claimed)
    clock.timestamps[str(claimed)] = 1
    clock.now = 1_000_000_000_000
    assert queue.reclaim_stale() == []
    marker = queue.root / "claim_observations" / ("ac52." + task["attempt_id"] + ".json")
    assert marker.exists()
    clock.now += 599 * 1_000_000_000
    assert queue.reclaim_stale() == []
    clock.now += 1_000_000_000
    assert queue.reclaim_stale()[0]["interrupted_claim_before_science"] is True


def test_input_hash_rejection_archives_reason_requeues_and_excludes_same_worker(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    record = queue.reject_before_science(claim, "InputHashMismatch: original SHA differs", {"pid":123})
    assert record["interrupted_claim_before_science"] is True
    assert record["preserve_prior_resume_lineage"] is True
    assert record["prior_resume_lineage"] is None
    assert record["reason"] == "InputHashMismatch: original SHA differs"
    retry = jobs.read_json(queue.root / "queue/ac52.json")
    assert retry["attempt_id"] != claim.token
    assert not list((queue.root / "failed").glob("*.json"))
    assert not list((queue.root / "done").glob("*.json"))
    assert queue.claim("Main", "cpu", {}, exclude_runs={record["run"]}, **callbacks) is None


def test_pre_science_rejection_cannot_waive_existing_output_or_progress(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    (claim.output / "original_partial.json").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="output/progress"):
        queue.reject_before_science(claim, "not allowed after output", {})
    queue.assert_owner(claim)
    assert (claim.output / "original_partial.json").read_bytes() == b"preserve"


def test_shared_publication_survives_lease_move_and_foreign_local_source_absence(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    original_finish = queue._finish_publication
    monkeypatch.setattr(queue, "_finish_publication", lambda directory: None)  # Simulated crash after original rename fence.
    queue.complete(claim, completion())
    candidate = claim.output / "candidate_completed.json"
    original_raw = candidate.read_bytes()
    candidate.unlink()  # Explicit synthetic producer-local source is unavailable.
    monkeypatch.setattr(queue, "_finish_publication", original_finish)
    queue.recover_publications()
    done = jobs.read_json(queue.root / "done/ac52.json")
    assert done["format"] == "darsia-alias-done-v1"
    assert set(done["completed"]) == {"relative_path", "sha256", "bytes"}
    assert (queue.root / done["completed"]["relative_path"]).read_bytes() == original_raw
    assert queue.aggregate()["complete"] is True
    assert queue.aggregate()["results"]["ac52"]["records"][0]["nas_directory"] == "C:/actual-main/old-attempt/frame1"


def test_publication_source_descriptor_cannot_escape_bundle_or_bind_another_completion(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    original = callbacks["publication_factory"]
    def bad(claim, bound, success):
        result = original(claim, bound, success)
        result["candidate"] = dict(result["candidate"], relative_path="../foreign/completed.json")
        return result
    bad.publish_progress = original.publish_progress
    claim = claim_fixture(queue, clock, dict(callbacks, publication_factory=bad))
    with pytest.raises(ValueError, match="inside its bundle"):
        queue.complete(claim, completion())
    assert not (queue.root / "done/ac52.json").exists()
    queue.assert_owner(claim)


def test_publication_does_not_relabel_open_references_as_completion(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    with pytest.raises(ValueError, match="still open"):
        queue.complete(claim, dict(completion(), worker_references_closed=False))
    assert not (claim.output / "candidate_completed.json").exists()


def test_unavailable_prefix_skips_before_claim_without_weakening_original_guards(tmp_path, monkeypatch):
    queue, _clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    unavailable = dict(callbacks, availability=lambda task: None)
    assert queue.claim("Main", "cpu", {}, **unavailable) is None
    assert (queue.root / "queue/ac52.json").exists()
    assert not list((queue.root / "claimed").glob("*/*.json"))


def test_pre_science_middle_generation_retains_bound_original_scientific_lineage(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    first = claim_fixture(queue, clock, callbacks)
    queue.progress(first, committed_event(first))
    clock.stamp(first.lease / "progress_00000000.json")
    clock.now += 600 * 1_000_000_000
    original_record = queue.reclaim_stale()[0]
    second = claim_fixture(queue, clock, callbacks)
    assert second.token == original_record["next_attempt"]
    rejected = queue.reject_before_science(second, "InputHashMismatch before any new science", {"pid":123})
    assert rejected["preserve_prior_resume_lineage"] is True
    prior = rejected["prior_resume_lineage"]
    path = queue.root / prior["relative_path"]
    assert path == queue.root / "reclaimed" / "ac52" / first.token / "reclaimed.json"
    assert jobs._own_asset(queue.root, prior) == path.resolve()
    assert jobs.read_json(path)["previous_attempt"] == first.token
    assert (first.output / "progress_00000000.json").exists()
    # This test binds the chain only. Actual prefix/retirement admission remains
    # the original checkpoint verifier plus A7's explicit alias path adapter.


def test_physical_machine_binding_is_distinct_from_service_claim_identity(tmp_path, monkeypatch):
    queue, _clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = queue.claim("Main-cpu", "cpu", {"pid": 123}, **callbacks)
    owner = jobs.read_json(claim.lease / "owner.json")
    assert claim.machine == "main-cpu"
    assert owner["machine"] == "main-cpu"
    assert owner["claimed"]["relative_path"] == "claimed/main-cpu/ac52.json"
    assert owner["output"]["machine_id"] == "Main"
    assert claim.local_output.machine_id == "Main"


def test_alias_without_explicit_physical_machine_or_progress_publisher_does_not_rename(tmp_path, monkeypatch):
    queue, _clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    queue._machine_id = None
    with pytest.raises(RuntimeError, match="physical store machine_id"):
        queue.claim("Main-cpu", "cpu", {}, **callbacks)
    assert (queue.root / "queue/ac52.json").exists()
    queue._machine_id = "main"
    def only_terminal(*args):
        raise AssertionError("Must not publish or launch")
    with pytest.raises(RuntimeError, match="frame-commit publication"):
        queue.claim("Main-cpu", "cpu", {}, **dict(callbacks, publication_factory=only_terminal))
    assert (queue.root / "queue/ac52.json").exists()


def test_unverified_frame_copy_never_becomes_shared_progress(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    event = committed_event(claim)
    original = claim.publication_factory.publish_progress
    def corrupt(claim, event):
        reference = original(claim, event)
        reference["sha256"] = "0" * 64
        return reference
    claim.publication_factory.publish_progress = corrupt
    with pytest.raises(ValueError, match="metadata changed"):
        queue.progress(claim, event)
    assert not list(claim.lease.glob("progress_*.json"))
    assert (claim.output / "progress_00000000.json").exists()


def test_nonframe_event_does_not_publish_or_claim_frame_progress(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    claim = claim_fixture(queue, clock, callbacks)
    def forbidden(*_):
        raise AssertionError("No frame completed")
    claim.publication_factory.publish_progress = forbidden
    event = dict(state="frame_started", run="ac52", case_id="ac52_f1", frame_index=1)
    queue.progress(claim, event)
    shared = jobs.read_json(claim.lease / "progress_00000000.json")
    assert shared["event"] == event and "published_frame" not in shared


@pytest.mark.parametrize("source", ["job", "catalogue"])
def test_bool_ready_without_exact_role_io_runtime_marker_never_claims(tmp_path, monkeypatch, source):
    queue, _clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    if source == "job":
        path = queue.root / "job.json"
        value = jobs.read_json(path); value.pop("runtime_support")
    else:
        path = queue.root / "queue/ac52.json"
        value = jobs.read_json(path); value["input_catalog"]["runtime_support"] = "J3 pending"
    path.write_bytes(json.dumps(value).encode())
    with pytest.raises(RuntimeError, match="not explicitly ready"):
        queue.claim("Main-cpu", "cpu", {}, **callbacks)
    assert (queue.root / "queue/ac52.json").exists()
    assert not list((queue.root / "claimed").glob("*/*.json"))


def payload_callback(original, *, kind="film", tamper=False):
    def publish(claim, candidate, success):
        source = claim.output / "final-film.mp4"
        source.write_bytes(b"synthetic final film")
        descriptor = jobs.descriptor(source)
        product = products_publication.product("final-film.mp4", machine_id="Main",
            local_path=descriptor["path"], sha256=descriptor["sha256"],
            bytes=descriptor["size_bytes"], kind=kind)
        bundle = claim.lease / "artifacts"
        index = products_publication.publish_run_metadata(bundle,
            job=jobs.descriptor(claim.lease.parents[1] / "job.json"),
            run=claim.task["run"], attempt_id=claim.token, machine_id="Main",
            completed=candidate, products=[product], metadata_product_ids=[],
            file_writers_closed=True, all_final_products_listed=True)
        payload = products_publication.publish_indexed_products(index, bundle / "default_products",
            current_machine_id="Main", publish=[kind], file_writers_closed=True)
        if tamper:
            path = Path(payload["path"])
            value = jobs.read_json(path); value["products_index"]["sha256"] = "0" * 64
            path.write_bytes(json.dumps(value).encode())
        return dict(candidate=jobs._own_reference(claim.lease, bundle / "completed.json"),
            products_index=jobs._own_reference(claim.lease, Path(index["path"])),
            payload_publication=jobs._own_reference(claim.lease, Path(payload["path"])))
    publish.publish_progress = original.publish_progress
    return publish


def test_default_payload_receipt_survives_lease_rename_and_is_job_relative_in_done(tmp_path, monkeypatch):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    callback = payload_callback(callbacks["publication_factory"])
    claim = claim_fixture(queue, clock, dict(callbacks, publication_factory=callback))
    queue.complete(claim, completion())
    done = jobs.read_json(queue.root / "done/ac52.json")
    path = jobs._own_asset(queue.root, done["payload_publication"])
    # The published bundle stays in its lease; publishing/ac52 is only its marker.
    assert jobs.publication_directory(queue.root, "ac52") == claim.lease
    assert path == (claim.lease / "artifacts/default_products/publication.json").resolve()
    assert done["payload_publication"]["relative_path"] == (
        "leases/ac52." + claim.token + "/artifacts/default_products/publication.json")
    receipt = jobs.read_json(path)
    assert receipt["products_index"]["path"].startswith(str(claim.lease))
    assert receipt["products_index"]["sha256"] == done["products_index"]["sha256"]
    # The done record stays job-relative when the whole shared root is renamed.
    moved = tmp_path / "renamed-shared-root"
    queue.root.rename(moved)
    assert not claim.lease.exists()  # Old path is passive provenance, not opened.
    assert jobs._own_asset(moved, done["payload_publication"]) == (
        moved / done["payload_publication"]["relative_path"]).resolve()
    assert jobs.read_json(jobs._own_asset(moved, done["payload_publication"])) == receipt


@pytest.mark.parametrize("mode", ["index_mismatch", "unrequested_kind"])
def test_unbound_or_unrequested_payload_receipt_cannot_pass_terminal_fence(tmp_path, monkeypatch, mode):
    queue, clock, callbacks, _events = fixture(tmp_path, monkeypatch)
    callback = payload_callback(callbacks["publication_factory"],
        kind="map" if mode == "unrequested_kind" else "film", tamper=mode == "index_mismatch")
    claim = claim_fixture(queue, clock, dict(callbacks, publication_factory=callback))
    with pytest.raises(ValueError, match="bound to this original index|absent matching products"):
        queue.complete(claim, completion())
    assert not (queue.root / "done/ac52.json").exists()
    queue.assert_owner(claim)


@pytest.fixture
def result_space_cache_boundary():
    """Only these result-space callbacks double the separate cache protocol.

    No accepted production policy, namespace, reserve or measured cache profile
    is represented by this result-only fixture.
    """
    def make(expected):
        class CacheBoundary:
            def __init__(self):
                self.prepares = 0
            def prepare(self, task, estimate):
                assert task["run"] == "ac52" and estimate is expected
                self.prepares += 1
            def acquire(self):
                pass
            def check(self, task, estimate, other):
                assert task["run"] == "ac52" and estimate is expected and other == 0
                return dict(TEST_ONLY_SEPARATE_CACHE_BOUNDARY=True)
            def bind(self, claim):
                pytest.fail("This result availability fixture must not bind a cache allocation")
            def unlock(self):
                pass
        return CacheBoundary()
    return make


@pytest.mark.parametrize("shortfall",[0,1])
def test_capacity_check_precedes_atomic_claim_and_refuses_without_result(
    tmp_path, monkeypatch, shortfall, result_space_cache_boundary
):
    from types import SimpleNamespace
    from contextlib import nullcontext
    from darsia.fluidflower.pipeline import capacity
    queue,clock,callbacks,events=fixture(tmp_path,monkeypatch)
    value=capacity.SpaceEstimate("ac52",1,100,100,0,0,30,100,{},"a"*40)
    free={"value":130-shortfall}
    ledger=SimpleNamespace(reserve=0,fresh_volume=lambda:dict(free_bytes=free["value"]),remaining=lambda:0,locked=nullcontext)
    runtime=SimpleNamespace(root=queue.root,stores=None,job=dict(series_id="Synthetic"),
        availability=callbacks["availability"],capacity_refusals={},capacities={})
    original_owner=dict(machine="main-cpu",pid=123,process_create_time="12345.125")
    check=capacity.ClaimCapacity(runtime,original_owner);check.ledger=ledger
    check.cache=result_space_cache_boundary(value)
    monkeypatch.setattr(capacity,"configuration_for_task",lambda *a: {})
    monkeypatch.setattr(capacity,"load_estimate",lambda *a,**k:value)
    claim=queue.claim("Main","cpu",original_owner,**dict(callbacks,availability=check.availability))
    if shortfall:
        assert claim is None and (queue.root/"queue/ac52.json").exists()
        assert events==["available_before_rename"]
        assert not (tmp_path/"local-results").exists()
        assert not list((queue.root/"claimed").glob("*/*.json"))
    else:
        assert claim is not None and events[:2]==["available_before_rename","owned_output_after_rename"]
        assert check.estimates[claim.token] is value


def test_capacity_callback_rechecks_free_space_before_each_unclaimed_run(
    tmp_path, monkeypatch, result_space_cache_boundary
):
    from types import SimpleNamespace
    from contextlib import nullcontext
    from darsia.fluidflower.pipeline import capacity
    queue,clock,callbacks,events=fixture(tmp_path,monkeypatch)
    value=capacity.SpaceEstimate("ac52",1,100,100,0,0,30,100,{},"a"*40)
    spaces=iter([129,130])
    ledger=SimpleNamespace(reserve=0,fresh_volume=lambda:dict(free_bytes=next(spaces)),remaining=lambda:0,locked=nullcontext)
    runtime=SimpleNamespace(root=queue.root,stores=None,job=dict(series_id="Synthetic"),
        availability=callbacks["availability"],capacity_refusals={},capacities={})
    owner=dict(machine="main-cpu",pid=123,process_create_time="12345.125")
    check=capacity.ClaimCapacity(runtime,owner);check.ledger=ledger
    check.cache=result_space_cache_boundary(value)
    monkeypatch.setattr(capacity,"configuration_for_task",lambda *a:{})
    monkeypatch.setattr(capacity,"load_estimate",lambda *a,**k:value)
    assert queue.claim("Main","cpu",owner,**dict(callbacks,availability=check.availability)) is None
    assert queue.claim("Main","cpu",owner,**dict(callbacks,availability=check.availability)) is not None
    assert check.cache.prepares == 2
