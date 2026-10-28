"""Native review integration against original tiny N1 saved-output fixtures.

These tests are source-only additions until an explicitly authorized test run.
Fixtures generate tiny genuine committed outputs; the attachment itself must
only consume saved metadata and the one selected stored frame.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower.calibration_review_native import NativeSavedReview
from darsia.fluidflower import storage_n1_saved_products as saved_n1
from darsia.fluidflower import storage_saved_products as saved
from darsia.fluidflower.pipeline import products_publication as publication
from .test_storage_n1_saved_products import (
    GEOMETRY, _index, _inputs, _run, phase1_data, tiny_config,
)


@pytest.fixture
def native_attachment(tmp_path, tiny_config, phase1_data):
    config = _inputs(tmp_path, tiny_config, phase1_data, photo=True)
    completion = _run(tmp_path, config)
    index = _index(tmp_path, completion)
    return config, completion, index


def _review(completion, index, **options):
    return NativeSavedReview(index, current_machine_id=options.pop("current_machine_id", "Main"),
        expected_run=options.pop("expected_run", completion["run"]),
        expected_calibration_sha256=options.pop("expected_calibration_sha256", completion["calibration"]["sha256"]),
        **options)


def test_open_admits_only_metadata_and_copies_explicit_bindings(native_attachment, monkeypatch):
    config, completion, index = native_attachment
    original_index = deepcopy(index)
    aliases = {}
    def forbidden(*args, **kwargs):
        raise AssertionError("Opening native review attempted to load a frame")
    monkeypatch.setattr(saved_n1, "read_n1_stored_frame", forbidden)
    review = _review(completion, index, current_machine_id="Second", source_paths=aliases)
    index["sha256"] = "0" * 64
    aliases["unrelated"] = "never-read"
    assert review.index == original_index
    assert review.source_paths == {}
    assert review.current_machine_id == "Second"
    assert review.expected_run == completion["run"]
    assert review.case_ids == tuple(completion["ordered_case_ids"])
    assert review.frame_metadata(1) == config["frames"][1]
    metadata = review.frame_metadata(1)
    metadata["source_binding"]["source_sha256"] = "0" * 64
    assert review.frame_metadata(1) == config["frames"][1]
    returned_index = review.index
    returned_index["sha256"] = "0" * 64
    assert review.index == original_index


def test_selected_frame_uses_original_reader_without_loading_other_frames(native_attachment, monkeypatch):
    _config, completion, index = native_attachment
    review = _review(completion, index)
    original, calls = saved_n1.read_n1_stored_frame, []
    def tracked(record, **kwargs):
        calls.append(record["frame_id"])
        return original(record, **kwargs)
    monkeypatch.setattr(saved_n1, "read_n1_stored_frame", tracked)
    stored = review.read_frame(1)
    assert calls == [review.case_ids[1]]
    assert stored["case_id"] == review.case_ids[1]
    assert stored["native_scalar_reference"]["elapsed_h"] == review.frame_metadata(1)["elapsed_h"]
    assert not stored["arrays"]["cell_mass_total_g"].flags.writeable
    assert all(Path(item["path"]).name not in {"products.npz", "native_geometry.npz"}
               for item in stored["loaded_assets"])


@pytest.mark.parametrize("changes", [
    {"expected_run": "another_run"},
    {"expected_calibration_sha256": "0" * 64},
    {"expected_calibration_sha256": "not-a-sha"},
    {"geometry_producer_sha256": "not-a-sha"},
])
def test_selected_identity_is_exact(native_attachment, changes):
    _config, completion, index = native_attachment
    with pytest.raises(ValueError):
        _review(completion, index, **changes)


@pytest.mark.parametrize("position", [-1, 2, True, 0.0, "0"])
def test_invalid_frame_index_does_not_wrap(native_attachment, position):
    _config, completion, index = native_attachment
    review = _review(completion, index)
    with pytest.raises(IndexError):
        review.frame_metadata(position)
    with pytest.raises(IndexError):
        review.read_frame(position)


def test_foreign_machine_requires_explicit_collection(native_attachment):
    _config, completion, index = native_attachment
    review = _review(completion, index, current_machine_id="Second")
    with pytest.raises(saved.StoredProductCollectionRequired):
        review.read_frame(0)


def test_explicit_collection_supports_frame_and_original_photo(native_attachment, tmp_path, monkeypatch):
    _config, completion, index = native_attachment
    _, _, rows = publication._read_index(index)
    wanted = [key for key, row in rows.items() if "/storage_" in key and row["kind"] == "map"]
    collection = publication.collect_products(index, wanted, tmp_path / "collected",
        current_machine_id="Second", file_writers_closed=True,
        source_paths={key: rows[key]["local_path"] for key in wanted})
    supplied = [deepcopy(collection)]
    review = _review(completion, index, current_machine_id="Second", collections=supplied,
                     geometry_producer_sha256=GEOMETRY)
    supplied[0]["sha256"] = "0" * 64
    assert review.collections == (collection,)
    original_read = Path.read_bytes
    producer = tmp_path / "mass"
    def guarded(path):
        assert not path.is_relative_to(producer), "Foreign producer payload was opened"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    stored = review.read_frame(0)
    rgba, evidence = review.read_photo(stored)
    assert rgba.shape == (3, 3, 4)
    assert evidence["geometry_producer_sha256"] == GEOMETRY


def test_photo_uses_original_per_frame_geometry_and_requires_same_attachment(native_attachment):
    _config, completion, index = native_attachment
    review = _review(completion, index)
    stored = review.read_frame(0)
    rgba, evidence = review.read_photo(stored)
    assert rgba.shape == (3, 3, 4)
    assert evidence["geometry_producer_sha256"] == GEOMETRY
    qualified = _review(completion, index, geometry_producer_sha256=GEOMETRY)
    wrong = dict(stored, run="another_run")
    with pytest.raises(ValueError, match="differs"):
        qualified.read_photo(wrong)
    wrong_geometry = _review(completion, index, geometry_producer_sha256="0" * 64)
    with pytest.raises(ValueError, match="geometry|identity"):
        wrong_geometry.read_photo(stored)


def test_completion_cannot_relabel_the_original_manifest_calibration(tiny_config, phase1_data, tmp_path, monkeypatch):
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data))
    completion["calibration"] = dict(completion["calibration"], sha256="0" * 64)
    index = _index(tmp_path, completion)
    def forbidden(*args, **kwargs):
        raise AssertionError("Contradictory metadata attempted a frame read")
    monkeypatch.setattr(saved_n1, "read_n1_stored_frame", forbidden)
    with pytest.raises(ValueError, match="manifest differs"):
        _review(completion, index)


@pytest.mark.parametrize("change", ["resume", "mixed_manifest"])
def test_unsupported_resumed_aggregates_fail_before_frames(tiny_config, phase1_data, tmp_path, monkeypatch, change):
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data))
    if change == "resume":
        completion["resumed_frame_count"] = 1
    else:
        completion["frame_producers"][0]["manifest"] = dict(
            completion["frame_producers"][0]["manifest"], sha256="0" * 64)
    index = _index(tmp_path, completion)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported aggregate attempted a frame read")
    monkeypatch.setattr(saved_n1, "read_n1_stored_frame", forbidden)
    with pytest.raises(ValueError, match="[Rr]esumed"):
        _review(completion, index)
