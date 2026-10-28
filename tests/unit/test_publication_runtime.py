"""Synthetic explicit metadata inventories only; no science, worker or network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import publication_runtime as runtime
from darsia.fluidflower.pipeline import products_publication as publication


def asset(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + "\r\n").encode()
    path.write_bytes(raw)
    return asset(path)


def frame(output, case, *, film=False, export_kind=None):
    manifest = write(output / "manifest.json", {"run": "ac52", "precision": "float32"})
    stage = output / "preparation" / case
    labels = write(stage / "phase_labels.png", b"synthetic undecoded label bytes\x00\xff")
    native = write(stage / "native_geometry.npz", b"synthetic undecoded map bytes")
    phase = {"phase_labels.png": labels, "native_geometry.npz": native}
    if film:
        phase["preview.mp4"] = write(stage / "preview.mp4", b"synthetic preview\x00")
    if export_kind is not None:
        exported = write(stage / "producer_named_export.bin", b"existing explicit lossless bytes")
        exported["kind"] = export_kind
        phase["producer_named_export.bin"] = exported
    result = write(output / "frames" / case / "result.json", {
        "prediction": {"pipeline": {"complete": True, "case_id": case, "files": phase}},
        "mass": 0.125,
    })
    stats = write(output / "frames" / case / "native_statistics.json.gz", b"synthetic closed statistics")
    files = {Path(item["path"]).name: {k: v for k, v in item.items() if k != "path"}
             for item in (result, stats)}
    commit = write(output / "frames" / case / "commit.json", {
        "complete": True, "frame_id": case, "commit_id": "c" * 32,
        "manifest_sha256": manifest["sha256"], "settings": {"precision": "float32"},
        "products": {"precision": "float32", "backend": "cpu"}, "files": files,
    })
    record = {"frame_id": case, "nas_directory": str(output / "frames" / case),
              "manifest_sha256": manifest["sha256"],
              "files": dict(files, **{"commit.json": {k: v for k, v in commit.items() if k != "path"}})}
    registration = write(output / "ops" / (case + "-registration.json"), {
        "frame_id": case, "commit": commit, "files": files,
    })
    receipt = write(output / "ops" / (case + "-publication.json"), record)
    producer = {"case_id": case, "attempt_directory": str(output), "manifest": manifest,
                "commit": commit, "frame_registration": registration,
                "publication": receipt, "result": result}
    return record, producer, phase


def fixture(tmp_path, *, prefix=False, film=False, publish=None, export_kind=None, case_ids=None):
    root = tmp_path / "shared"
    job = {"format": "darsia-alias-job-v1", "job_id": "synthetic-job"}
    if publish is not None:
        job["publish"] = publish
    write(root / "job.json", job)
    token = "a" * 32
    output = tmp_path / "local" / "synthetic-job" / "ac52" / ".attempts" / token
    cases = list(case_ids) if case_ids is not None else (["ac52_f1", "ac52_f2"] if prefix else ["ac52_f1"])
    task = {"run": "ac52", "attempt_id": token, "ordered_case_ids": cases,
            "commit": "d" * 40, "calibration_sha256": "e" * 64}
    claim = SimpleNamespace(task=task, token=token, output=output,
                            lease=root / "leases" / ("ac52." + token))
    claim.lease.mkdir(parents=True)
    records, producers, phases, proofs = [], [], [], {}
    if prefix:
        old = tmp_path / "old-explicitly-accessible" / ("b" * 32)
        record, producer, phase = frame(old, cases[0])
        records.append(record); producers.append(producer); phases.append(phase)
        retirement = write(old / "retirement.json", {
            "complete": True, "all_absent": True, "worker_os_references_closed": True,
            "run": "ac52", "attempt_id": old.name, "attempt_directory": str(old),
            "machine_id": "Second",
        })
        proofs[str(old)] = {"machine_id": "Second", "proof_descriptor": retirement,
                            "accessible_root": str(old)}
    record, producer, phase = frame(output, cases[-1], film=film, export_kind=export_kind)
    records.append(record); producers.append(producer); phases.append(phase)
    table = write(output / "mass_table.csv", b"case,mass\r\nac52_f1,0.125\r\n")
    result = {"complete": True, "worker_references_closed": True, "run": "ac52",
              "attempt_id": token, "task": task, "commit": task["commit"],
              "calibration_sha256": task["calibration_sha256"], "case_count": len(cases),
              "ordered_case_ids": cases, "records": records, "frame_producers": producers,
              "manifest": asset(output / "manifest.json"), "table": table, "precision": "float32"}
    raw = b"\xef\xbb\xbf" + (json.dumps(result, ensure_ascii=False, indent=2) + "\r\n").encode()
    candidate = write(output / "candidate_completed.json", raw)
    def origin(attempt):
        return deepcopy(proofs[str(Path(attempt))])
    factory = runtime.make_publication_factory(root, job, "Main", producer_origin_for_attempt=origin)
    return SimpleNamespace(root=root, job=job, claim=claim, result=result, raw=raw,
        candidate=candidate, factory=factory, proofs=proofs, phases=phases)


def bound(claim, reference):
    return {"path": str(claim.lease.joinpath(*reference["relative_path"].split("/"))),
            "sha256": reference["sha256"], "bytes": reference["bytes"]}


def index_of(data, prepared):
    descriptor = bound(data.claim, prepared["products_index"])
    return descriptor, json.loads(Path(descriptor["path"]).read_bytes())


def test_original_main_completion_bytes_and_true_producer_metadata_are_unchanged(tmp_path):
    data = fixture(tmp_path)
    prepared = data.factory(data.claim, data.candidate, True)
    shared = Path(bound(data.claim, prepared["candidate"])["path"])
    assert shared.read_bytes() == data.raw == Path(data.candidate["path"]).read_bytes()
    index, document = index_of(data, prepared)
    assert publication.read_published_metadata(index) == data.raw
    assert document["file_closure"]["os_retirement_asserted"] is False
    assert all(row["machine_id"] == "Main" for row in document["products"])
    assert "payload_publication" not in prepared
    assert not (data.claim.lease / "artifacts/default_products").exists()


def test_explicit_native_inventory_exposes_phase_labels_without_scanning_or_loading_maps(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    original_file = publication._file
    maps = {row["path"] for row in data.phases[0].values()}
    def no_payload(path):
        assert str(path) not in maps, "Indexing must not open a map payload"
        return original_file(path)
    def no_walk(*_args, **_kwargs):
        raise AssertionError("No directory inventory scan in publication adapter")
    monkeypatch.setattr(publication, "_file", no_payload)
    monkeypatch.setattr(Path, "iterdir", no_walk)
    monkeypatch.setattr(Path, "glob", no_walk)
    prepared = data.factory(data.claim, data.candidate, True)
    _index, document = index_of(data, prepared)
    rows = [row for row in document["products"] if row["local_path"] in maps]
    assert len(rows) == 2 and all(row["kind"] == "map" for row in rows)
    assert all("metadata_copy" not in row for row in rows)
    assert any(row["local_path"].endswith("phase_labels.png") for row in rows)


def test_prefix_machine_and_exact_retirement_evidence_are_preserved_in_shared_index(tmp_path):
    data = fixture(tmp_path, prefix=True)
    prepared = data.factory(data.claim, data.candidate, True)
    index, document = index_of(data, prepared)
    prefix = data.result["frame_producers"][0]["attempt_directory"]
    prefix_rows = [row for row in document["products"] if row["local_path"].startswith(prefix)]
    assert prefix_rows and all(row["machine_id"] == "Second" for row in prefix_rows)
    current_rows = [row for row in document["products"] if row["local_path"].startswith(str(data.claim.output))]
    assert current_rows and all(row["machine_id"] == "Main" for row in current_rows)
    proof = next(row for row in prefix_rows if row["local_path"].endswith("retirement.json"))
    original = Path(proof["local_path"]).read_bytes()
    assert publication.read_published_metadata(index, proof["product_id"]) == original


@pytest.mark.parametrize("field,value", [("machine_id", "Wrong"), ("attempt_id", "f" * 32),
    ("all_absent", False), ("worker_os_references_closed", False), ("run", "ac21")])
def test_prefix_retirement_identity_or_closure_mismatch_is_not_relabelled(tmp_path, field, value):
    data = fixture(tmp_path, prefix=True)
    proof = next(iter(data.proofs.values()))
    path = Path(proof["proof_descriptor"]["path"])
    document = json.loads(path.read_bytes()); document[field] = value
    proof["proof_descriptor"] = write(path, document)
    with pytest.raises(ValueError, match="retired prefix proof differs"):
        data.factory(data.claim, data.candidate, True)
    assert not (data.claim.lease / "artifacts").exists()


def test_missing_prefix_origin_proof_cannot_fall_back_to_current_machine(tmp_path):
    data = fixture(tmp_path, prefix=True)
    data.proofs.clear()
    with pytest.raises(KeyError):
        data.factory(data.claim, data.candidate, True)
    assert not (data.claim.lease / "artifacts").exists()


@pytest.mark.parametrize("field,value", [("run", "ac21"), ("attempt_id", "f" * 32),
    ("commit", "f" * 40), ("calibration_sha256", "f" * 64)])
def test_completion_must_belong_to_exact_original_claim(tmp_path, field, value):
    data = fixture(tmp_path)
    altered = dict(data.result, **{field: value})
    candidate = write(Path(data.candidate["path"]), altered)
    with pytest.raises(ValueError, match="another claim|identity differs"):
        data.factory(data.claim, candidate, True)
    assert not (data.claim.lease / "artifacts").exists()


def test_exact_order_and_original_stage_case_binding_cannot_be_repaired_by_adapter(tmp_path):
    data = fixture(tmp_path)
    bad = deepcopy(data.result)
    bad["frame_producers"][0]["case_id"] = "ac21_f1"
    candidate = write(Path(data.candidate["path"]), bad)
    with pytest.raises(ValueError, match="exact case order"):
        data.factory(data.claim, candidate, True)
    assert not (data.claim.lease / "artifacts").exists()


def test_candidate_sha_failure_does_not_start_shared_publication(tmp_path):
    data = fixture(tmp_path)
    Path(data.candidate["path"]).write_bytes(data.raw + b" ")
    with pytest.raises(ValueError, match="size differs"):
        data.factory(data.claim, data.candidate, True)
    assert not (data.claim.lease / "artifacts").exists()


def test_default_product_publication_copies_only_existing_matching_kinds_after_index(tmp_path):
    data = fixture(tmp_path, film=True)
    prepared = data.factory(data.claim, data.candidate, True)
    index, document = index_of(data, prepared)
    original_index = Path(index["path"]).read_bytes()
    receipt_path = Path(bound(data.claim, prepared["payload_publication"])["path"])
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["products_index"]["sha256"] == index["sha256"]
    assert receipt["products_index"]["bytes"] == index["bytes"]
    assert len(receipt["copies"]) == 1
    assert receipt["copies"][0]["product_id"].endswith("preview.mp4")
    assert (receipt_path.parent / receipt["copies"][0]["relative_path"]).read_bytes() == b"synthetic preview\x00"
    assert Path(index["path"]).read_bytes() == original_index
    assert any(row["kind"] == "map" and "metadata_copy" not in row for row in document["products"])


def test_explicit_map_publication_is_requested_by_job_not_automatic(tmp_path):
    data = fixture(tmp_path, publish=["map"])
    prepared = data.factory(data.claim, data.candidate, True)
    receipt_path = Path(bound(data.claim, prepared["payload_publication"])["path"])
    receipt = json.loads(receipt_path.read_bytes())
    assert len(receipt["copies"]) == 2
    assert all(row["product_id"].endswith(("phase_labels.png", "native_geometry.npz")) for row in receipt["copies"])


def test_relative_bundle_references_survive_atomic_lease_move_without_original_candidate(tmp_path):
    data = fixture(tmp_path, film=True)
    prepared = data.factory(data.claim, data.candidate, True)
    moved = data.root / "publishing/ac52"
    moved.parent.mkdir()
    data.claim.lease.rename(moved)
    Path(data.candidate["path"]).unlink()
    data.claim.lease = moved
    index = bound(data.claim, prepared["products_index"])
    receipt = bound(data.claim, prepared["payload_publication"])
    assert publication.read_published_metadata(index) == data.raw
    collected = publication.collect_products(index,
        [json.loads(Path(receipt["path"]).read_bytes())["copies"][0]["product_id"]],
        tmp_path / "collected", current_machine_id="Other", payload_publications=[receipt],
        file_writers_closed=True)
    assert json.loads(Path(collected["path"]).read_bytes())["copies"]


def test_original_failure_callback_retains_verbatim_traceback_and_raw_bytes(tmp_path):
    data = fixture(tmp_path)
    failure = {"complete": False, "run": "ac52", "attempt_id": data.claim.token,
               "traceback": "Traceback (most recent call last):\nValueError: original Ω\n"}
    raw = b"\xef\xbb\xbf" + (json.dumps(failure, ensure_ascii=False, indent=4) + "\r\n").encode()
    source = write(data.claim.output / "failed.json", raw)
    prepared = data.factory(data.claim, source, False)
    assert set(prepared) == {"candidate"}
    assert Path(bound(data.claim, prepared["candidate"])["path"]).read_bytes() == raw
    assert not (data.claim.lease / "artifacts/products_index.json").exists()


def test_original_frame_commit_is_copied_as_bytes_without_new_synthetic_commit(tmp_path):
    data = fixture(tmp_path)
    record = data.result["records"][0]
    event = {"state": "frame_completed", "complete": True, "run": "ac52",
             "case_id": record["frame_id"], "publication": record}
    original_path = Path(record["nas_directory"]) / "commit.json"
    raw = original_path.read_bytes()
    reference = data.factory.publish_progress(data.claim, event)
    assert Path(bound(data.claim, reference)["path"]).read_bytes() == raw
    assert reference["sha256"] == record["files"]["commit.json"]["sha256"]
    original_path.rename(original_path.with_name("not-accessible-as-commit.json"))
    assert json.loads(Path(bound(data.claim, reference)["path"]).read_bytes())["frame_id"] == record["frame_id"]


def test_frame_progress_rejects_mismatched_commit_before_shared_copy(tmp_path):
    data = fixture(tmp_path)
    record = deepcopy(data.result["records"][0])
    record["manifest_sha256"] = "0" * 64
    event = {"state": "frame_completed", "complete": True, "run": "ac52",
             "case_id": record["frame_id"], "publication": record}
    with pytest.raises(ValueError, match="exact publication record"):
        data.factory.publish_progress(data.claim, event)
    assert not (data.claim.lease / "progress_artifacts").exists()


def test_duplicate_progress_does_not_overwrite_first_shared_commit(tmp_path):
    data = fixture(tmp_path)
    record = data.result["records"][0]
    event = {"state": "frame_completed", "complete": True, "run": "ac52",
             "case_id": record["frame_id"], "publication": record}
    reference = data.factory.publish_progress(data.claim, event)
    raw = Path(bound(data.claim, reference)["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        data.factory.publish_progress(data.claim, event)
    assert Path(bound(data.claim, reference)["path"]).read_bytes() == raw


def test_existing_explicit_lossless_export_role_is_preserved_without_filename_guess(tmp_path):
    data = fixture(tmp_path, export_kind="lossless_export")
    prepared = data.factory(data.claim, data.candidate, True)
    _index, document = index_of(data, prepared)
    exports = [row for row in document["products"] if row["kind"] == "lossless_export"]
    assert len(exports) == 1 and exports[0]["local_path"].endswith("producer_named_export.bin")
    receipt = json.loads(Path(bound(data.claim, prepared["payload_publication"])["path"]).read_bytes())
    assert [row["product_id"] for row in receipt["copies"]] == [exports[0]["product_id"]]
    maps = [row for row in document["products"] if row["local_path"].endswith("native_geometry.npz")]
    assert maps and all(row["kind"] == "map" and "metadata_copy" not in row for row in maps)


@pytest.mark.parametrize("invalid", ["Invalid kind", "", 42])
def test_invalid_explicit_product_kind_is_rejected_without_publishing(tmp_path, invalid):
    data = fixture(tmp_path, export_kind=invalid)
    with pytest.raises(ValueError, match="declared lowercase product kind"):
        data.factory(data.claim, data.candidate, True)
    assert not (data.claim.lease / "artifacts").exists()


@pytest.mark.parametrize("stage", ["metadata_copy", "default_product_copy"])
def test_partial_success_publication_cannot_block_original_failure_bytes(tmp_path, monkeypatch, stage):
    data = fixture(tmp_path, film=True)
    if stage == "metadata_copy":
        original = publication._copy
        calls = []
        def fail_second(source, expected, target):
            calls.append(str(target))
            if len(calls) == 2:
                raise OSError("Original publication failure: synthetic metadata copy")
            return original(source, expected, target)
        monkeypatch.setattr(publication, "_copy", fail_second)
    else:
        def fail_payload(*args, **kwargs):
            raise OSError("Original publication failure: synthetic explicit payload copy")
        monkeypatch.setattr(runtime, "publish_indexed_products", fail_payload)
    with pytest.raises(OSError, match="Original publication failure"):
        data.factory(data.claim, data.candidate, True)
    partial = data.claim.lease / "artifacts"
    assert partial.exists()
    original_partial = {str(path.relative_to(partial)): path.read_bytes()
                        for path in partial.rglob("*") if path.is_file()}
    failure = {"complete": False, "run": "ac52", "attempt_id": data.claim.token,
               "traceback": "Traceback (most recent call last):\nOSError: original copy failure\n"}
    raw = (json.dumps(failure, indent=4) + "\r\n").encode()
    candidate = write(data.claim.output / "failed.json", raw)
    prepared = data.factory(data.claim, candidate, False)
    assert prepared["candidate"]["relative_path"] == "failure_artifacts/failed.json"
    assert Path(bound(data.claim, prepared["candidate"])["path"]).read_bytes() == raw
    assert {str(path.relative_to(partial)): path.read_bytes()
            for path in partial.rglob("*") if path.is_file()} == original_partial
    assert Path(data.candidate["path"]).read_bytes() == data.raw


def test_mixed_case_selected_ids_are_preserved_in_index_and_original_completion(tmp_path):
    case = "Ac52_F1"
    data = fixture(tmp_path, case_ids=[case])
    task_before = deepcopy(data.claim.task)
    prepared = data.factory(data.claim, data.candidate, True)
    index, document = index_of(data, prepared)
    frame_ids = [row["product_id"] for row in document["products"] if row["product_id"].startswith("frames/")]
    assert frame_ids and all(row.startswith("frames/"+case+"/") for row in frame_ids)
    assert "frames/"+case+"/commit.json" in frame_ids
    assert not any(row.startswith("frames/ac52_f1/") for row in frame_ids)
    assert publication.read_published_metadata(index) == data.raw
    assert data.claim.task == task_before
    assert data.result["ordered_case_ids"] == [case]
    assert data.result["records"][0]["frame_id"] == case
