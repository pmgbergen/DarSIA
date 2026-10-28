"""Prepared, unrun tests for existing receipts and serialized Qt attachment."""
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QThread

from darsia.fluidflower import calibration_review_native_collection_ui as collection_ui
from darsia.fluidflower.calibration_review_native import NativeSavedReview
from darsia.fluidflower.pipeline import products_publication as publication
from darsia.fluidflower.storage_saved_products import StoredProductCollectionRequired
from ..test_calibration_review_native_bindings import (
    native_binding, native_attachment, phase1_data, tiny_config,
)
from ..test_calibration_review_native import GEOMETRY
from ..test_calibration_review_native_phase import native_phase_attachment


def _forbidden(*args, **kwargs):
    raise AssertionError("Existing-receipt attachment attempted copying or payload access")


def _reader(value, **options):
    return NativeSavedReview(value["index"], current_machine_id="Second",
        expected_run=value["completion"]["run"],
        expected_calibration_sha256=value["completion"]["calibration"]["sha256"],
        geometry_producer_sha256=GEOMETRY, **options)


def test_selected_original_receipt_constructs_replacement_without_live_mutation(native_binding, monkeypatch):
    value = native_binding
    metadata = publication.publish_indexed_products(value["index"], value["root"] / "published-manifest",
        current_machine_id="Main", publish=["manifest"], file_writers_closed=True)
    selected_id = next(key for key, row in value["rows"].items() if row["kind"] == "map")
    aliases = {selected_id: value["rows"][selected_id]["local_path"]}
    old = _reader(value, collections=[metadata], source_paths=aliases)
    original_bindings = old.collections
    original_tree = set(value["root"].rglob("*"))
    original_files = {path: path.read_bytes() for path in original_tree if path.is_file()}
    descriptor_calls, original_descriptor, original_open = [], collection_ui._selected_receipt, Path.open
    def descriptor(path):
        descriptor_calls.append(path)
        return original_descriptor(path)
    def metadata_only(path, *args, **kwargs):
        if path.suffix.lower() in {".png", ".npz", ".npy", ".jpg"}:
            return _forbidden()
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(collection_ui, "_selected_receipt", descriptor)
        guard.setattr(Path, "open", metadata_only)
        guard.setattr(Path, "glob", _forbidden)
        guard.setattr(Path, "rglob", _forbidden)
        guard.setattr(publication, "collect_products", _forbidden)
        guard.setattr(publication, "publish_indexed_products", _forbidden)
        replacement = collection_ui.attach_existing_collection(old, value["payload"]["path"])
    assert replacement is not old and old.collections == original_bindings
    for key in ("index", "current_machine_id", "expected_run", "expected_calibration_sha256",
                "geometry_producer_sha256", "source_paths", "case_ids", "pipeline", "photographs"):
        assert getattr(replacement, key) == getattr(old, key)
    assert len(replacement.collections) == 2
    assert replacement.source_paths == aliases
    assert descriptor_calls == [Path(value["payload"]["path"])]
    assert set(value["root"].rglob("*")) == original_tree
    assert {path: path.read_bytes() for path in original_files} == original_files
    assert replacement.read_frame(0)["producer_kind"] == "n1-mass-only"


def test_selected_receipt_keeps_receiving_host_and_never_changes_old_availability(native_binding):
    value = native_binding
    old = _reader(value)
    replacement = collection_ui.attach_existing_collection(old, value["payload"]["path"])
    assert old.current_machine_id == replacement.current_machine_id == "Second"
    assert old.collections == () and old.source_paths is None and replacement.source_paths is None
    with pytest.raises(StoredProductCollectionRequired):
        old.read_frame(0)
    assert replacement.read_frame(0)["arrays"]["cell_mass_total_g"].flags.writeable is False


def test_collection_retains_native_phase_protocol_and_explicit_photo_bindings(
        native_phase_attachment, tmp_path, monkeypatch):
    value = native_phase_attachment
    _root, _document, rows = publication._read_index(value["index"])
    selected = [key for key in rows if key.startswith(("storage/", "photos/"))]
    payload = publication.collect_products(value["index"], selected, tmp_path / "received-native-phase",
        current_machine_id="Main", file_writers_closed=True)
    old = NativeSavedReview(value["index"], current_machine_id="Second",
        expected_run=value["completion"]["run"],
        expected_calibration_sha256=value["completion"]["calibration"]["sha256"],
        expected_pipeline="native-phase-n1", photographs=value["photographs"],
        geometry_producer_sha256="f" * 64)
    original_open = Path.open
    def metadata_only(path, *args, **kwargs):
        if path.suffix.lower() in {".png", ".npz", ".npy", ".jpg"}:
            return _forbidden()
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", metadata_only)
        guard.setattr(publication, "collect_products", _forbidden)
        guard.setattr(publication, "publish_indexed_products", _forbidden)
        replacement = collection_ui.attach_existing_collection(old, payload["path"])
    assert old.collections == ()
    assert replacement.collections == (dict(path=payload["path"], sha256=payload["sha256"],
                                           size_bytes=payload["bytes"]),)
    assert replacement.pipeline == old.pipeline == "native-phase-n1"
    assert replacement.photographs == old.photographs == value["photographs"]
    assert replacement.geometry_producer_sha256 == old.geometry_producer_sha256 == "f" * 64
    with pytest.raises(StoredProductCollectionRequired):
        old.read_frame(0)
    stored = replacement.read_frame(0)
    rgba, _receipt = replacement.read_photo(stored)
    assert rgba.shape == (2, 2, 4)
    assert "producer_kind" not in stored


@pytest.mark.parametrize("fault", ["format", "index", "producer"])
def test_selected_receipt_must_authenticate_against_original_index(native_binding, fault):
    value = native_binding
    document = json.loads(Path(value["payload"]["path"]).read_bytes())
    if fault == "format":
        document["format"] = "fluidflower-phase-review-collection-v1"
    elif fault == "index":
        document["products_index"]["sha256"] = "0" * 64
    else:
        document["copies"][0]["machine_id"] = "Second"
    selected = value["root"] / "selected-receipt.json"
    selected.write_text(json.dumps(document), encoding="utf-8")
    old = _reader(value)
    with pytest.raises(ValueError):
        collection_ui.attach_existing_collection(old, selected)
    assert old.collections == ()


def test_identical_receipt_coalesces_but_distinct_overlapping_receipt_is_rejected(native_binding):
    value = native_binding
    first = collection_ui.attach_existing_collection(_reader(value), value["payload"]["path"])
    same = collection_ui.attach_existing_collection(first, value["payload"]["path"])
    assert same is not first and same.collections == first.collections
    duplicate = Path(value["payload"]["path"]).with_name("collection.json")
    duplicate.write_bytes(Path(value["payload"]["path"]).read_bytes())
    with pytest.raises(ValueError, match="conflicting"):
        collection_ui.attach_existing_collection(first, duplicate)
    assert len(first.collections) == 1


def test_oversized_receipt_is_rejected_before_descriptor_hashing(native_binding, monkeypatch):
    value = native_binding
    old = _reader(value)
    selected = value["root"] / "oversized.json"
    selected.write_bytes(b" " * (publication.MAX_INDEX_BYTES + 1))
    monkeypatch.setattr(collection_ui.hashlib, "sha256", _forbidden)
    with pytest.raises(ValueError, match="size limit"):
        collection_ui.attach_existing_collection(old, selected)


def test_receipt_growth_between_stat_and_open_keeps_a_hard_read_limit(native_binding, monkeypatch):
    value = native_binding
    old = _reader(value)
    selected = Path(value["payload"]["path"])
    original_open, reads = Path.open, []
    class ObservedStream:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def read(self, limit):
            reads.append(limit)
            return self.stream.read(limit)
    def grew_before_open(path, *args, **kwargs):
        if path == selected and args == ("rb",):
            with original_open(path, "ab") as append:
                append.write(b" " * (publication.MAX_INDEX_BYTES + 1))
            return ObservedStream(original_open(path, *args, **kwargs))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", grew_before_open)
    monkeypatch.setattr(collection_ui.hashlib, "sha256", _forbidden)
    with pytest.raises(ValueError, match="size limit"):
        collection_ui.attach_existing_collection(old, selected)
    assert reads == [publication.MAX_INDEX_BYTES + 1]
    assert old.collections == ()


@pytest.mark.parametrize("path", [None, "", "relative/collection.json"])
def test_selected_receipt_needs_explicit_absolute_path(native_binding, monkeypatch, path):
    old = _reader(native_binding)
    monkeypatch.setattr(collection_ui, "_selected_receipt", _forbidden)
    with pytest.raises(ValueError, match="absolute"):
        collection_ui.attach_existing_collection(old, path)


def test_receipt_changed_after_selection_descriptor_is_rejected(native_binding, monkeypatch):
    value = native_binding
    old, original = _reader(value), collection_ui._selected_receipt
    def changed(path):
        bound = original(path)
        path.write_bytes(path.read_bytes() + b"\n")
        return bound
    monkeypatch.setattr(collection_ui, "_selected_receipt", changed)
    with pytest.raises(ValueError):
        collection_ui.attach_existing_collection(old, value["payload"]["path"])
    assert old.collections == ()


class _Signal:
    def connect(self, *args):
        self.connection = args


class _ManualAttachment:
    def __init__(self, generation, context, reader, path, parent):
        self.generation, self.context, self.reader, self.path = generation, context, reader, path
        self.finished = _Signal()
        self.replacement, self.error = None, ""
        self.started = self.interruptions = self.waits = self.deleted = 0

    def start(self): self.started += 1
    def requestInterruption(self): self.interruptions += 1
    def wait(self): self.waits += 1
    def deleteLater(self): self.deleted += 1


@pytest.fixture
def controller(qtbot, monkeypatch):
    workers = []
    def factory(*args):
        worker = _ManualAttachment(*args)
        workers.append(worker)
        return worker
    monkeypatch.setattr(collection_ui, "_Attachment", factory)
    instance = collection_ui.NativeCollectionLookup()
    yield instance, workers
    instance.stop()
    instance.deleteLater()


def test_controller_coalesces_requests_and_clears_busy_before_ready(controller):
    lookup, workers = controller
    delivered = []
    lookup.ready.connect(lambda *args: delivered.append((args, lookup.busy())))
    lookup.request(("root-a", "old-reader"), "old", "first.json")
    lookup.request(("root-b", "middle-reader"), "middle", "second.json")
    lookup.request(("root-c", "latest-reader"), "latest", "third.json")
    assert lookup.busy() and len(workers) == 1 and workers[0].interruptions == 2
    workers[0].replacement = "stale replacement"
    lookup._finish(workers[0])
    assert delivered == [] and len(workers) == 2 and lookup.busy()
    assert workers[1].context == ("root-c", "latest-reader") and workers[1].path == "third.json"
    workers[1].replacement = "current replacement"
    lookup._finish(workers[1])
    assert delivered == [((("root-c", "latest-reader"), "current replacement", ""), False)]
    assert not lookup.busy() and all(worker.deleted == 1 for worker in workers)


def test_stop_waits_discards_pending_and_ignores_late_finished(controller):
    lookup, workers = controller
    delivered = []
    lookup.ready.connect(lambda *args: delivered.append(args))
    lookup.request("old", object(), "old.json")
    lookup.request("pending", object(), "pending.json")
    worker = workers[0]
    lookup.stop()
    assert worker.waits == 1 and worker.deleted == 1 and not lookup.busy()
    lookup._finish(worker)
    lookup._finish(None)
    assert delivered == [] and len(workers) == 1
    lookup.request("new", object(), "new.json")
    lookup._start()
    assert delivered == [] and len(workers) == 1 and not lookup.busy()


def test_ready_callback_can_request_again_without_starting_two_workers(controller):
    lookup, workers = controller
    seen = []
    def ready(context, replacement, error):
        seen.append((context, lookup.busy()))
        if context == "first":
            lookup.request("second", object(), "second.json")
    lookup.ready.connect(ready)
    lookup.request("first", object(), "first.json")
    workers[0].replacement = object()
    lookup._finish(workers[0])
    assert seen == [("first", False)] and len(workers) == 2 and lookup.busy()
    assert workers[1].started == 1


@pytest.mark.parametrize("failure", [False, True])
def test_real_worker_delivers_only_on_gui_thread_after_retirement(qtbot, monkeypatch, failure):
    app_thread = QCoreApplication.instance().thread()
    work_on_gui, received = [], []
    replacement = object()
    def attach(*args):
        work_on_gui.append(QThread.currentThread() is app_thread)
        if failure:
            raise ValueError("Receipt does not match")
        return replacement
    monkeypatch.setattr(collection_ui, "attach_existing_collection", attach)
    lookup = collection_ui.NativeCollectionLookup()
    def ready(context, reader, error):
        received.append((context, reader, error, QThread.currentThread() is app_thread, lookup.busy()))
    lookup.ready.connect(ready)
    try:
        lookup.request("captured-context", object(), "explicit-selection.json")
        qtbot.waitUntil(lambda: bool(received), timeout=5000)
        assert work_on_gui == [False]
        assert received == [("captured-context", None if failure else replacement,
                             "Receipt does not match" if failure else "", True, False)]
    finally:
        lookup.stop()
        lookup.deleteLater()
