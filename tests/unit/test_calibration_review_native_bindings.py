"""Unrun binding tests using genuine tiny N1 outputs and original publication.

Future authorized tests create their own temporary saved fixture and existing
payload publication before testing passive attachment. No queue/science work
is performed by the attachment helpers. Mutated receipts below deliberately
test metadata rejection; they are not assertions of valid copied products.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.calibration_review_native import NativeSavedReview
from darsia.fluidflower.calibration_review_native_bindings import collect_bindings, photo_geometry
from darsia.fluidflower import storage_n1_saved_products as saved_n1
from darsia.fluidflower.storage_indexed_products import indexed_asset_resolver
from darsia.fluidflower.storage_saved_products import StoredProductCollectionRequired
from darsia.fluidflower.pipeline import job_queue, products_publication as publication
from .test_calibration_review_native import GEOMETRY, native_attachment, phase1_data, tiny_config


def _forbidden(*args, **kwargs):
    raise AssertionError("Passive binding attempted payload access, discovery or copying")


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return job_queue.descriptor(path)


@pytest.fixture
def native_binding(tmp_path, native_attachment):
    _config, completed, index = native_attachment
    _root, document, rows = publication._read_index(index)
    payload = publication.publish_indexed_products(index, tmp_path / "published-payload",
        current_machine_id="Main", publish=["map"], file_writers_closed=True)
    done = dict(format="darsia-alias-done-v1", run=document["run"], attempt_id=document["attempt_id"],
        products_index=job_queue._own_reference(tmp_path, index["path"]),
        payload_publication=job_queue._own_reference(tmp_path, payload["path"]))
    return dict(root=tmp_path, completion=completed, index=index, rows=rows, payload=payload, done=done)


def _bindings(value, **options):
    return collect_bindings(value["index"], value["root"], value["done"],
        current_machine_id=options.pop("current_machine_id", "Second"), **options)


def _review(value, bindings):
    return NativeSavedReview(value["index"], current_machine_id="Second", collections=bindings,
        expected_run=value["completion"]["run"],
        expected_calibration_sha256=value["completion"]["calibration"]["sha256"])


def test_existing_done_publication_is_admitted_without_payload_io(native_binding, monkeypatch):
    value = native_binding
    original = deepcopy((value["done"], value["index"], value["payload"]))
    original_tree = set(value["root"].rglob("*"))
    original_files = {path: path.read_bytes() for path in original_tree if path.is_file()}
    original_stat, original_open = Path.stat, Path.open
    def metadata_stat(path, *args, **kwargs):
        if path.suffix.lower() in {".png", ".npz", ".npy", ".jpg"}:
            return _forbidden()
        return original_stat(path, *args, **kwargs)
    def metadata_open(path, *args, **kwargs):
        if path.suffix.lower() in {".png", ".npz", ".npy", ".jpg"}:
            return _forbidden()
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "stat", metadata_stat)
        guard.setattr(Path, "open", metadata_open)
        guard.setattr(Path, "rglob", _forbidden)
        guard.setattr(Path, "glob", _forbidden)
        guard.setattr(publication, "collect_products", _forbidden)
        guard.setattr(publication, "publish_indexed_products", _forbidden)
        guard.setattr(job_queue, "JobQueue", _forbidden)
        bindings = _bindings(value)
    assert isinstance(bindings, tuple) and len(bindings) == 1
    assert bindings[0] == dict(path=value["payload"]["path"], sha256=value["payload"]["sha256"],
                               size_bytes=value["payload"]["bytes"])
    bindings[0]["sha256"] = "0" * 64
    assert (value["done"], value["index"], value["payload"]) == original
    assert set(value["root"].rglob("*")) == original_tree
    assert {path: path.read_bytes() for path in original_files} == original_files


def test_existing_publication_enables_original_foreign_frame_and_photo_readers(native_binding):
    value = native_binding
    bindings = _bindings(value)
    review = _review(value, bindings)
    stored = review.read_frame(0)
    assert review.current_machine_id == "Second" and review.source_paths is None
    digest = photo_geometry(stored)
    assert digest == GEOMETRY
    resolve = indexed_asset_resolver(value["index"], current_machine_id="Second", collections=bindings)
    rgba, evidence = saved_n1.read_n1_original_photo(stored, resolve_asset=resolve,
                                                   geometry_producer_sha256=digest)
    assert str(rgba.dtype) == "uint8" and rgba.shape[-1] == 4 and not rgba.flags.writeable
    assert evidence["geometry_producer_sha256"] == GEOMETRY
    assert not stored["arrays"]["cell_mass_total_g"].flags.writeable


def test_no_publication_keeps_foreign_payloads_unavailable_without_inference(native_binding):
    value = native_binding
    value["done"].pop("payload_publication")
    assert _bindings(value) == ()
    with pytest.raises(StoredProductCollectionRequired):
        _review(value, ()).read_frame(0)
    explicit = _bindings(value, collections=[value["payload"]])
    assert len(explicit) == 1
    assert _review(value, explicit).read_frame(0)["producer_kind"] == "n1-mass-only"


def test_original_collection_receipt_is_explicit_only_and_retains_producer_identity(native_binding, monkeypatch):
    value = native_binding
    # Test setup only: the actual original collector produces a small receipt
    # from the already-published tiny fixture. Attachment never runs it.
    keys = [key for key, row in value["rows"].items() if row["kind"] == "map"]
    collection = publication.collect_products(value["index"], keys, value["root"] / "existing-collection",
        current_machine_id="Second", payload_publications=[value["payload"]], file_writers_closed=True)
    monkeypatch.setattr(publication, "collect_products", _forbidden)
    value["done"]["payload_publication"] = job_queue._own_reference(value["root"], collection["path"])
    with pytest.raises(ValueError, match="publication receipt"):
        _bindings(value)
    value["done"].pop("payload_publication")
    bound = _bindings(value, collections=[collection])
    assert len(bound) == 1 and bound[0]["sha256"] == collection["sha256"]
    review = _review(value, bound)
    assert review.current_machine_id == "Second" and review.source_paths is None
    assert review.read_frame(0)["producer_kind"] == "n1-mass-only"


@pytest.mark.parametrize("fault", ["format", "run", "attempt", "index", "payload_sha", "payload_escape"])
def test_done_must_bind_its_exact_original_index_and_publication(native_binding, fault):
    value = native_binding
    if fault == "format":
        value["done"]["format"] = "legacy-done"
    elif fault == "run":
        value["done"]["run"] = "other-run"
    elif fault == "attempt":
        value["done"]["attempt_id"] = "b" * 32
    elif fault == "index":
        value["done"]["products_index"]["sha256"] = "0" * 64
    elif fault == "payload_sha":
        value["done"]["payload_publication"]["sha256"] = "0" * 64
    else:
        value["done"]["payload_publication"]["relative_path"] = "../unregistered/publication.json"
    with pytest.raises(ValueError):
        _bindings(value)


@pytest.mark.parametrize("fault", ["format", "index", "run", "attempt", "closed", "copy_sha", "copy_bytes",
                                   "producer", "local_path", "relative_path", "unknown_product", "duplicate_copy"])
def test_explicit_receipt_semantics_use_original_resolver(native_binding, fault):
    value = native_binding
    receipt = json.loads(Path(value["payload"]["path"]).read_bytes())
    value["done"].pop("payload_publication")
    if fault == "format":
        receipt["format"] = "fluidflower-phase-review-collection-v1"
    elif fault == "index":
        receipt["products_index"]["sha256"] = "0" * 64
    elif fault == "run":
        receipt["run"] = "other-run"
    elif fault == "attempt":
        receipt["original_attempt_id"] = "b" * 32
    elif fault == "closed":
        receipt["file_writers_closed"] = False
    elif fault == "copy_sha":
        receipt["copies"][0]["sha256"] = "0" * 64
    elif fault == "copy_bytes":
        receipt["copies"][0]["bytes"] += 1
    elif fault == "producer":
        receipt["copies"][0]["machine_id"] = "Second"
    elif fault == "local_path":
        receipt["copies"][0]["local_path"] += ".different"
    elif fault == "relative_path":
        receipt["copies"][0]["relative_path"] = "../outside.npz"
    elif fault == "unknown_product":
        receipt["copies"][0]["product_id"] = "unregistered/product.npz"
    else:
        receipt["copies"].append(deepcopy(receipt["copies"][0]))
    bound = _write(value["root"] / "invalid-receipt.json", receipt)
    with pytest.raises(ValueError):
        _bindings(value, collections=[bound])


def test_identical_binding_coalesces_but_other_overlapping_receipts_fail(native_binding):
    value = native_binding
    assert _bindings(value, collections=[value["payload"]]) == _bindings(value)
    document = json.loads(Path(value["payload"]["path"]).read_bytes())
    other = _write(Path(value["payload"]["path"]).with_name("another-publication.json"), document)
    with pytest.raises(ValueError, match="conflicting"):
        _bindings(value, collections=[other])


def test_changed_receipt_bytes_are_not_adopted(native_binding):
    value = native_binding
    path = Path(value["payload"]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        _bindings(value)


@pytest.mark.parametrize("collections", [None, "some-directory", [{"path": "/unbound/collection.json"}]])
def test_collection_inputs_require_explicit_immutable_descriptors(native_binding, collections):
    with pytest.raises(ValueError):
        _bindings(native_binding, collections=collections)


@pytest.mark.parametrize("machine", [None, "", "../Main"])
def test_machine_identity_is_explicit(native_binding, machine):
    with pytest.raises(ValueError):
        _bindings(native_binding, current_machine_id=machine)


def test_photo_geometry_is_saved_per_frame_metadata_without_io(native_binding, monkeypatch):
    value = native_binding
    stored = _review(value, _bindings(value)).read_frame(0)
    before_document = deepcopy(stored["storage_document"])
    before_arrays = {key: array.tobytes() for key, array in stored["arrays"].items()}
    with monkeypatch.context() as guard:
        guard.setattr(Path, "read_bytes", _forbidden)
        guard.setattr(Path, "open", _forbidden)
        guard.setattr(saved_n1, "read_n1_stored_frame", _forbidden)
        assert photo_geometry(stored) == photo_geometry(stored, expected=GEOMETRY) == GEOMETRY
    assert stored["storage_document"] == before_document
    assert {key: array.tobytes() for key, array in stored["arrays"].items()} == before_arrays
    with pytest.raises(ValueError, match="original producer request"):
        photo_geometry(stored, expected="0" * 64)


@pytest.mark.parametrize("fault", ["reader_format", "producer", "document_run", "source", "changed_science",
                                   "missing_digest", "invalid_digest", "photo_copy", "pending"])
def test_photo_geometry_rejects_changed_or_unavailable_identity(native_binding, fault):
    value = native_binding
    stored = _review(value, _bindings(value)).read_frame(0)
    if fault == "reader_format":
        stored["format"] = "studio-delivery"
    elif fault == "producer":
        stored["producer_kind"] = "native-phase"
    elif fault == "document_run":
        stored["storage_document"]["run"] = "other-run"
    elif fault == "source":
        stored["source"]["source_sha256"] = "0" * 64
    elif fault == "changed_science":
        stored["storage_document"]["existing_native_products_unchanged"] = 2
    elif fault == "missing_digest":
        for photo in (stored["photograph"], stored["storage_document"]["photograph"]):
            photo.pop("geometry_producer_sha256")
    elif fault == "invalid_digest":
        for photo in (stored["photograph"], stored["storage_document"]["photograph"]):
            photo["geometry_producer_sha256"] = "not-a-digest"
    elif fault == "photo_copy":
        stored["photograph"]["geometry_producer_sha256"] = "0" * 64
    else:
        stored["photograph"] = stored["storage_document"]["photograph"] = dict(state="pending", reason="not_bound")
    with pytest.raises(FileNotFoundError if fault == "pending" else ValueError):
        photo_geometry(stored)


def test_photo_digest_helper_does_not_replace_original_receipt_validation(native_binding):
    value = native_binding
    bindings = _bindings(value)
    stored = _review(value, bindings).read_frame(0)
    for photo in (stored["photograph"], stored["storage_document"]["photograph"]):
        photo["geometry_producer_sha256"] = "0" * 64
    digest = photo_geometry(stored)
    resolve = indexed_asset_resolver(value["index"], current_machine_id="Second", collections=bindings)
    with pytest.raises(ValueError, match="identity/pre-light"):
        saved_n1.read_n1_original_photo(stored, resolve_asset=resolve, geometry_producer_sha256=digest)
