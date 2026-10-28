"""IPP policy boundaries use the executing thread, never historical products."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import cv2
import pytest

import darsia
from darsia import _opencv_runtime as runtime


@pytest.fixture
def fresh_thread_policy(monkeypatch):
    previous = bool(cv2.ipp.useIPP())
    monkeypatch.setattr(runtime, "_thread", threading.local())
    yield
    cv2.ipp.setUseIPP(previous)


def test_already_initialized_cv2_is_disabled_by_actual_code_in_child():
    environment = os.environ.copy()
    environment.pop("OPENCV_IPP", None)
    code = """import cv2, json
before = bool(cv2.ipp.useIPP())
assert before, 'This locked test runtime must expose an initially enabled IPP build'
import darsia
from darsia._opencv_runtime import require_ipp_off
print(json.dumps(dict(before=before, after=require_ipp_off('already-imported'))))
"""
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code],
        cwd=Path(darsia.__file__).parent.parent, env=environment,
        capture_output=True, text=True, check=True, timeout=60)
    observed = json.loads(result.stdout)
    assert observed["before"] is True
    assert observed["after"]["use_ipp"] is False
    assert observed["after"]["process_environment_opencv_ipp"] == "disabled"
    assert observed["after"]["observation_scope"] == "executing-thread-at-boundary"


def test_real_python_thread_has_its_own_initialization(fresh_thread_policy, monkeypatch):
    main = runtime.scientific_thread("main")
    records, failures = [], []
    setters = []
    actual_setter = cv2.ipp.setUseIPP

    def setter(value):
        setters.append((threading.get_ident(), value))
        actual_setter(value)

    monkeypatch.setattr(cv2.ipp, "setUseIPP", setter)

    def child():
        try:
            before = runtime.scientific_thread("child:before")
            records.append(runtime.completed_frame(before, "child:after"))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=child)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive() and not failures
    assert len(records) == 1
    assert records[0]["before"]["thread_id"] != main["thread_id"]
    assert records[0]["before"]["use_ipp"] is False
    assert records[0]["after"]["use_ipp"] is False
    assert setters == [(records[0]["before"]["thread_id"], False)]
    assert cv2.ipp.useIPP() is False


def test_reactivation_is_refused_before_commit(fresh_thread_policy, tmp_path, monkeypatch):
    runtime.scientific_thread("start")
    committed = tmp_path / "commit.json"
    with pytest.raises(RuntimeError, match="IPP policy violated"):
        with runtime.scientific_frame("frame"):
            monkeypatch.setattr(cv2.ipp, "useIPP", lambda: True)
        committed.write_text("must not publish", encoding="utf-8")
    assert not committed.exists()
    # A second entry cannot silently repair unexpected reactivation either.
    with pytest.raises(RuntimeError, match="IPP policy violated"):
        runtime.scientific_thread("next-entry")


def test_primary_science_exception_survives_reactivation(fresh_thread_policy, monkeypatch):
    class ScientificFailure(Exception):
        pass
    runtime.scientific_thread("start")
    original = ScientificFailure("original scientific traceback")
    with pytest.raises(ScientificFailure) as caught:
        with runtime.scientific_frame("frame"):
            monkeypatch.setattr(cv2.ipp, "useIPP", lambda: True)
            raise original
    assert caught.value is original


def test_observation_requires_same_executing_thread(fresh_thread_policy):
    before = runtime.scientific_thread("before")
    different = dict(before, thread_id=before["thread_id"] + 1)
    with pytest.raises(RuntimeError, match="changed its executing thread"):
        runtime.completed_frame(different, "after")


@pytest.mark.parametrize("field,value", [("use_ipp", True), ("policy", "old")])
def test_saved_observation_does_not_accept_enabled_or_other_policy(
        fresh_thread_policy, field, value):
    before = runtime.scientific_thread("before")
    record = runtime.completed_frame(before, "after")
    record["after"][field] = value
    with pytest.raises(ValueError, match="observation"):
        runtime.validate_observation(record)


def test_saved_observation_is_not_relabelled_or_mutated(fresh_thread_policy):
    before = runtime.scientific_thread("before")
    record = runtime.completed_frame(before, "after")
    original = deepcopy(record)
    runtime.validate_observation(record)
    assert record == original
    assert "no retroactive" in record["saved_inputs"]


@pytest.mark.parametrize("field", ["pid", "thread_id", "native_thread_id", "opencv_version"])
def test_saved_observation_requires_actual_observer_fields(fresh_thread_policy, field):
    record = runtime.scientific_thread("before")
    record.pop(field)
    with pytest.raises(ValueError, match="observation"):
        runtime.validate_observation(record)


def test_direct_native_entry_refuses_before_raw_read(fresh_thread_policy, monkeypatch):
    from darsia.fluidflower.native.source import SourceProcessor
    runtime.scientific_thread("start")
    monkeypatch.setattr(cv2.ipp, "useIPP", lambda: True)
    processor = object.__new__(SourceProcessor)
    # An uninitialized object would fail on its input bindings if reached.
    with pytest.raises(RuntimeError, match="IPP policy violated"):
        processor.correct(run="test", frame_id="frame_0", camera_jpeg=Path("no.jpg"),
                          camera_arw=Path("no.arw"), review_jpeg=Path("no-review.jpg"),
                          source_manifest_sha256="0" * 64, content_shape_hw=(1, 1),
                          content_box_xywh=(0, 0, 1, 1))


def test_publish_boundary_refuses_before_output_creation(fresh_thread_policy, tmp_path, monkeypatch):
    from darsia.fluidflower.mass.n1_pipeline import _publish_frame
    runtime.scientific_thread("start")
    monkeypatch.setattr(cv2.ipp, "useIPP", lambda: True)
    output = tmp_path / "not-created"
    with pytest.raises(RuntimeError, match="IPP policy violated"):
        _publish_frame(output, {}, {}, {}, {}, {}, "cpu", 0, 0.0)
    assert not output.exists()


def test_old_corrected_cache_is_preserved_and_not_read_as_new_policy(
        fresh_thread_policy, monkeypatch, tmp_path):
    from darsia.presets.workflows.utils.images import load_images_with_cache
    paths = [tmp_path / "source" / "a.jpg", tmp_path / "source" / "b.jpg"]
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"test source descriptor only")
    cache = tmp_path / "cache"
    cache.mkdir()
    old = cache / "a.npz"
    old.write_bytes(b"original IPP-unknown corrected cache")
    reads, hits = [], []

    class Image:
        def save(self, path):
            Path(path).write_bytes(b"new policy test image")

    def read(path):
        reads.append(path)
        return Image()

    monkeypatch.setattr(darsia, "imread", lambda path: hits.append(path) or Image())
    rig = SimpleNamespace(read_image=read)
    assert len(load_images_with_cache(rig, paths, True, cache)) == 2
    assert reads == paths and not hits
    assert old.read_bytes() == b"original IPP-unknown corrected cache"
    assert (cache / runtime.POLICY / "a.npz").is_file()
    assert len(load_images_with_cache(rig, paths, True, cache)) == 2
    assert reads == paths and len(hits) == 2
    assert all(runtime.POLICY in path.parts for path in hits)


def test_feature_cache_key_binds_stable_policy_not_observation(monkeypatch):
    from darsia.fluidflower.phase_training import trainer
    staged = SimpleNamespace(asset_manifest_path=None, asset_sha256="a" * 64)
    captured = []
    monkeypatch.setattr(trainer, "_canonical_hash", lambda value: captured.append(value) or "key")
    assert trainer._feature_cache_key(staged, [], pixels_per_region=2, texture_sigma=1) == "key"
    assert captured[0]["opencv_ipp_policy"] == runtime.POLICY
    assert "thread_id" not in captured[0] and "pid" not in captured[0]


def test_corrected_cache_is_not_saved_if_rig_reactivates_ipp(
        fresh_thread_policy, tmp_path, monkeypatch):
    from darsia.presets.workflows.utils.images import load_images_with_cache
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    saved = []

    def read(path):
        monkeypatch.setattr(cv2.ipp, "useIPP", lambda: True)
        return SimpleNamespace(save=lambda destination: saved.append(destination))

    with pytest.raises(RuntimeError, match="IPP policy violated"):
        load_images_with_cache(SimpleNamespace(read_image=read), paths, True, tmp_path / "cache")
    assert saved == []


def test_radiometry_does_not_publish_cache_after_ipp_reactivation(
        fresh_thread_policy, monkeypatch, tmp_path):
    from darsia.fluidflower.phase_training import radiometry
    from darsia.fluidflower import image_correction
    source = tmp_path / "source.jpg"
    source.write_bytes(b"test source only")
    adapter = object.__new__(radiometry.RunRadiometry)
    adapter.content_box = (0, 0, 1, 1)
    adapter.source_manifest = None
    adapter.frame_sources = {}
    adapter.upstream_correction_contract = None
    adapter.profile = {"sha256": "0" * 64}
    adapter.identity = {"opencv_ipp_policy": runtime.POLICY}
    adapter.series_id, adapter.run_id = "series", "run"
    adapter.baseline_sha256 = "1" * 64
    adapter.cache_root = tmp_path / "cache"
    adapter.roi = [[0, 1], [0, 1]]

    def prepare(*args, **kwargs):
        monkeypatch.setattr(cv2.ipp, "useIPP", lambda: True)
        return object(), {"synthetic": "not published"}

    monkeypatch.setattr(radiometry, "core", lambda: SimpleNamespace(
        digest=lambda value: "exact-test-key", prepare_image=prepare))
    monkeypatch.setattr(radiometry, "load_review_rgb", lambda *args: object())
    monkeypatch.setattr(image_correction, "_reference_colorchecker", lambda *args, **kwargs: (None, None))
    with pytest.raises(RuntimeError, match="IPP policy violated"):
        adapter.load(source, "frame_0")
    assert not adapter.cache_root.exists()


def test_old_feature_metadata_cannot_be_a_new_policy_cache_hit(
        fresh_thread_policy, monkeypatch, tmp_path):
    from darsia.fluidflower.phase_training import trainer
    output, metadata = tmp_path / "old.npz", tmp_path / "old.json"
    output.write_bytes(b"old generated feature bytes")
    metadata.write_text(json.dumps({"cache_key": "same-old-key"}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    task = SimpleNamespace(opencv_threads=1, output_path=output,
                           metadata_path=metadata, cache_key="same-old-key",
                           staged=SimpleNamespace(asset_manifest_path=manifest))
    original = (output.read_bytes(), metadata.read_bytes())
    with pytest.raises(ValueError, match="predates the IPP-off policy"):
        trainer._extract_run_features(task)
    assert (output.read_bytes(), metadata.read_bytes()) == original


@pytest.mark.parametrize("reader", ["seal", "publication-index"])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_phase1_completion_readers_accept_only_valid_current_observation(
        fresh_thread_policy, tmp_path, reader, enabled):
    from .test_phase1_worker_protocol import claimed_fixture
    from .test_staged_data import _rewrite_metadata
    from darsia.fluidflower.pipeline import phase1_protocol, staged_data
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    fixture = claimed_fixture(tmp_path)
    document = dict(fixture.closure["document"], opencv_ipp=runtime.scientific_thread("completion"))
    document["opencv_ipp"]["use_ipp"] = enabled
    bound = _rewrite_metadata(fixture.closure["completion"], document)
    fixture.result["phase1_completion"] = bound

    def read():
        if reader == "seal":
            return staged_data.verify_phase1_completion(bound, fixture.closure["configuration"],
                expected_commit="1" * 40, reader=DataReader())
        return phase1_protocol.extract_phase1_products(fixture.result, fixture.claim, "Main")

    if enabled:
        with pytest.raises(ValueError, match="observation"):
            read()
    else:
        assert read() is not None


def test_statistics_runtime_sidecar_binds_unchanged_statistics_product(
        fresh_thread_policy, tmp_path):
    from darsia.fluidflower.pipeline.phase_statistics import _save_statistics
    before = runtime.scientific_thread("statistics:before")
    observation = runtime.completed_frame(before, "statistics:after")
    extracted = dict(statistics={"actual_test_summary": 1.25}, sandmap={}, opencv_ipp=observation)
    product = _save_statistics(tmp_path, extracted)
    sidecar = json.loads((tmp_path / "native_statistics_runtime.json").read_bytes())
    assert sidecar["statistics"] == product
    assert sidecar["opencv_ipp"] == observation
    import gzip
    with gzip.open(product["path"], "rt", encoding="utf-8") as stream:
        assert json.load(stream) == extracted["statistics"]


@pytest.mark.parametrize("method", ["evaluate", "extract_statistics"])
@pytest.mark.parametrize("failure", ["entry-guard", "calculation"])
def test_pipeline_observation_is_cleared_before_failed_new_call(
        fresh_thread_policy, monkeypatch, method, failure):
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline
    context = object.__new__(N1Pipeline)
    context.last_opencv_ipp = {"old": "must not be published for this call"}
    context.statistics_only = False
    runtime.scientific_thread("test-start")

    class OriginalScientificFailure(Exception):
        pass

    original = OriginalScientificFailure("original calculation failure")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(context, "_prepare_statistics", fail)
    if failure == "entry-guard":
        monkeypatch.setattr(cv2.ipp, "useIPP", lambda: True)
        with pytest.raises(RuntimeError, match="IPP policy violated"):
            getattr(context, method)({})
    else:
        with pytest.raises(OriginalScientificFailure) as caught:
            getattr(context, method)({})
        assert caught.value is original
    assert context.last_opencv_ipp is None
