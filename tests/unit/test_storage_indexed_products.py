"""Tiny synthetic writer/index/collect/reader integration; no job or science."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import storage_indexed_products as indexed
from darsia.fluidflower import storage_outputs as output
from darsia.fluidflower import storage_saved_products as saved
from darsia.fluidflower.pipeline import products_publication as publication
from .test_storage_saved_products import _fixture


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True) + "\n").encode())
    return output._asset(path)


def _published(tmp_path, *, pending=False, extensive_inputs=None, alter_state=None):
    state = _fixture(tmp_path / "producer", pending=pending, extensive_inputs=extensive_inputs)
    if alter_state is not None:
        alter_state(state)
    manifest = output._asset(state["manifest_path"])
    completion = dict(complete=True, worker_references_closed=True, pipeline="native-phase-n1",
        run="ac52", frames=1, case_count=1, ordered_case_ids=["ac52_f1"],
        records=[state["record"]], manifest=manifest,
        configuration=state["manifest"]["configuration"], precision="float32",
        calibration=dict(path="explicit-unused-refit.json", sha256="c" * 64, size_bytes=1))
    completed = _write(state["root"] / "completed.json", completion)
    job = _write(tmp_path / "job.json", dict(format="synthetic-read-only-test-job"))
    rows, small = [], []
    def add(key, bound, kind):
        rows.append(publication.product(key, machine_id="Main", local_path=bound["path"],
            sha256=bound["sha256"], bytes=bound["size_bytes"], kind=kind))
        if kind in {"manifest", "frame_scalar", "commit_receipt", "operational_receipt"}:
            small.append(key)
    add("manifest.json", manifest, "manifest")
    add("frames/ac52_f1/commit.json", output._asset(state["directory"] / "commit.json"), "commit_receipt")
    add("frames/ac52_f1/result.json", output._asset(state["directory"] / "result.json"), "frame_scalar")
    # Real publication_runtime also lists result.json under a producer-inventory
    # logical ID. Resolver identities must support that exact duplicate asset.
    add("producer/result.json", output._asset(state["directory"] / "result.json"), "frame_scalar")
    add("storage/receipt.json", state["entry"]["receipt"], "operational_receipt")
    for bound in state["entry"]["files"]:
        add("storage/" + Path(bound["path"]).name, bound, bound.get("kind", "map"))
    # The original native archive is deliberately unavailable and must never be
    # opened by this storage reader, publication, or targeted collect.
    rows.append(publication.product("native/products.npz", machine_id="Main",
        local_path=str(state["directory"] / "products.npz"), sha256="e" * 64, bytes=999, kind="map"))
    index = publication.publish_run_metadata(tmp_path / "shared", job=job, run="ac52",
        attempt_id="a" * 32, machine_id="Main", completed=completed, products=rows,
        metadata_product_ids=small, file_writers_closed=True, all_final_products_listed=True)
    return state, index, completion


def _collect(tmp_path, index, *, include_before=True, include_extensive=True):
    _root, _document, rows = publication._read_index(index)
    keys = [key for key in rows if key.startswith("storage/") and rows[key]["kind"] == "map"
             and (include_before or not key.endswith(".png"))
             and (include_extensive or not key.endswith("storage_extensive_frame.npz"))]
    return publication.collect_products(index, keys, tmp_path / "collected",
        current_machine_id="Main", file_writers_closed=True)


def test_explicit_extensive_index_reader_streams_original_committed_payload(tmp_path):
    from darsia.fluidflower import storage_extensive_io as payload
    state, index, completion = _published(tmp_path, extensive_inputs=True)
    original = deepcopy(completion)
    run = payload.read_indexed_extensive_run(index, producer_kind="native-phase-n1", current_machine_id="Main")
    assert not isinstance(run["frames"], list)
    value = next(run["frames"])
    assert value["state"] == "synthetic_io_only"
    assert value["arrays"]["cell_mass_total_g"].sum(dtype=np.float64) == 9.
    assert run["physical_binding_admitted"] is run["producer_identity_admitted"] is False
    assert original == completion and not (state["directory"] / "products.npz").exists()
    with pytest.raises(StopIteration): next(run["frames"])


def test_unrequested_extensive_does_not_force_collect_but_explicit_reader_does(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_extensive_io as payload
    state, index, _ = _published(tmp_path, extensive_inputs=True)
    collection = _collect(tmp_path, index, include_extensive=False)
    original_stat = Path.stat
    def guard(path, *args, **kwargs):
        assert not str(path).startswith(str(state["root"])), "Foreign producer path was probed"
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", guard)
    old = indexed.read_indexed_stored_run(index, current_machine_id="Second", collections=[collection])
    assert old["frames"][0]["mass_totals_g"]["total"] == 9.
    run = payload.read_indexed_extensive_run(index, producer_kind="native-phase-n1",
        current_machine_id="Second", collections=[collection])
    with pytest.raises(saved.StoredProductCollectionRequired) as caught:
        next(run["frames"])
    assert Path(caught.value.descriptor["path"]).name == payload.ARCHIVE


def test_explicit_extensive_real_collect_uses_no_producer_or_native_fallback(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_extensive_io as payload
    state, index, _ = _published(tmp_path, extensive_inputs=True)
    collection = _collect(tmp_path, index)
    original_read = Path.read_bytes
    def guard(path):
        assert not str(path).startswith(str(state["root"])), "Producer payload fallback"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    run = payload.read_indexed_extensive_run(index, producer_kind="native-phase-n1",
        current_machine_id="Second", collections=[collection])
    assert next(run["frames"])["metadata"]["native_mass_totals_g"] == dict(total=9., aqueous=3., gas=3.)


def test_collected_extensive_archive_still_checks_original_SHA(tmp_path):
    from darsia.fluidflower import storage_extensive_io as payload
    _state, index, _ = _published(tmp_path, extensive_inputs=True)
    collection = _collect(tmp_path, index)
    path = Path(collection["path"]).parent / "products/storage/storage_extensive_frame.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    run = payload.read_indexed_extensive_run(index, producer_kind="native-phase-n1",
        current_machine_id="Second", collections=[collection])
    with pytest.raises(ValueError): next(run["frames"])


def test_resealed_spatial_mass_attack_is_rejected_after_real_foreign_collect(tmp_path):
    from darsia.fluidflower import storage_extensive_io as payload
    from .test_storage_saved_products import _reseal, _reseal_extensive_array_values
    def alter(state):
        def permute(arrays):
            original = arrays["cell_mass_total_g"].copy()
            arrays["cell_mass_total_g"][0, 0], arrays["cell_mass_total_g"][1, 1] = original[1, 1], original[0, 0]
            assert arrays["cell_mass_total_g"].sum(dtype=np.float64) == original.sum(dtype=np.float64)
        updated = _reseal_extensive_array_values(state["document"]["extensive"], permute)
        state["entry"]["extensive"] = deepcopy(state["document"]["extensive"])
        state["entry"]["files"] = [updated.get(Path(row["path"]).name, row) for row in state["entry"]["files"]]
        _reseal(state)
    _state, index, _ = _published(tmp_path, extensive_inputs=True, alter_state=alter)
    collection = _collect(tmp_path, index)
    run = payload.read_indexed_extensive_run(index, producer_kind="native-phase-n1",
        current_machine_id="Second", collections=[collection])
    with pytest.raises(ValueError, match="Extensive total cell masses differ from committed coarse arrays"):
        next(run["frames"])


def test_same_machine_reads_indexed_storage_with_original_completion_untouched(tmp_path):
    state, index, completion = _published(tmp_path)
    before = deepcopy(completion)
    result = indexed.read_indexed_stored_run(index, current_machine_id="Main")
    assert result["ordered_case_ids"] == ["ac52_f1"]
    assert result["frames"][0]["mass_totals_g"] == dict(total=9., aqueous=3., gas=3.)
    assert result["original_completion"] == before == completion
    assert not (state["directory"] / "products.npz").exists()


def test_foreign_machine_requires_collect_without_probing_producer_disk(tmp_path, monkeypatch):
    state, index, _ = _published(tmp_path)
    original_stat = Path.stat
    def guard(path, *args, **kwargs):
        assert not str(path).startswith(str(state["root"])), "Foreign producer path was probed"
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", guard)
    with pytest.raises(saved.StoredProductCollectionRequired) as caught:
        indexed.read_indexed_stored_run(index, current_machine_id="Second")
    assert caught.value.descriptor["path"].endswith("storage_stored_frame.npz")


def test_real_collect_receipt_reads_coarse_arrays_on_other_machine(tmp_path, monkeypatch):
    state, index, original = _published(tmp_path)
    collection = _collect(tmp_path, index)
    original_read = Path.read_bytes
    def guard(path):
        assert not str(path).startswith(str(state["root"])), "Producer payload fallback"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    result = indexed.read_indexed_stored_run(index, current_machine_id="Second",
        collections=[collection], load_before=True)
    frame = result["frames"][0]
    assert frame["mass_totals_g"] == dict(total=9., aqueous=3., gas=3.)
    np.testing.assert_array_equal(frame["before_native"][:1], np.array([[0., 1.]], np.float32))
    assert np.isnan(frame["before_native"][1, 1])
    assert result["original_completion"] == original


def test_optional_before_does_not_force_unrequested_native_png_collect(tmp_path):
    _state, index, _ = _published(tmp_path)
    collection = _collect(tmp_path, index, include_before=False)
    result = indexed.read_indexed_stored_run(index, current_machine_id="Second", collections=[collection])
    assert result["frames"][0]["before_native"] is None
    with pytest.raises(saved.StoredProductCollectionRequired):
        indexed.read_indexed_stored_run(index, current_machine_id="Second", collections=[collection], load_before=True)


def test_pending_before_survives_collect_without_inventing_a_score(tmp_path):
    _state, index, _ = _published(tmp_path, pending=True)
    collection = _collect(tmp_path, index)
    frame = indexed.read_indexed_stored_run(index, current_machine_id="Second",
        collections=[collection], load_before=True)["frames"][0]
    assert frame["before_rules"]["state"] == "pending" and frame["before_native"] is None
    assert "gas_probability_before_rules" not in frame["arrays"]


def test_changed_collected_payload_is_never_accepted_by_its_filename(tmp_path):
    _state, index, _ = _published(tmp_path)
    collection = _collect(tmp_path, index)
    path = Path(collection["path"]).parent / "products/storage/storage_stored_frame.npz"
    raw = bytearray(path.read_bytes()); raw[len(raw)//2] ^= 1; path.write_bytes(raw)
    with pytest.raises(ValueError, match="Stored product SHA"):
        indexed.read_indexed_stored_run(index, current_machine_id="Second", collections=[collection])


@pytest.mark.parametrize("field,value", [("run", "ac21"), ("original_attempt_id", "b" * 32),
    ("file_writers_closed", False), ("format", "unknown")])
def test_resealed_collection_must_match_original_index_identity(tmp_path, field, value):
    _state, index, _ = _published(tmp_path)
    collection = _collect(tmp_path, index)
    doc = json.loads(Path(collection["path"]).read_bytes()); doc[field] = value
    altered = _write(tmp_path / "altered.json", doc)
    with pytest.raises(ValueError, match="original products index"):
        indexed.indexed_asset_resolver(index, current_machine_id="Second", collections=[altered])


@pytest.mark.parametrize("field,value", [("sha256", "f" * 64), ("machine_id", "Other"),
    ("local_path", "C:/foreign/unrelated.npz"), ("relative_path", "../escape.npz"),
    ("product_id", "unlisted/product.npz")])
def test_collected_asset_keeps_full_original_producer_and_owned_path(tmp_path, field, value):
    _state, index, _ = _published(tmp_path)
    collection = _collect(tmp_path, index)
    doc = json.loads(Path(collection["path"]).read_bytes()); doc["copies"][0][field] = value
    altered = _write(tmp_path / "altered.json", doc)
    with pytest.raises(ValueError):
        indexed.indexed_asset_resolver(index, current_machine_id="Second", collections=[altered])


def test_unindexed_alias_or_asset_never_triggers_search_or_collection(tmp_path):
    _state, index, _ = _published(tmp_path)
    with pytest.raises(ValueError, match="indexed product IDs"):
        indexed.indexed_asset_resolver(index, current_machine_id="Second", source_paths={"unknown": "C:/whatever"})
    resolve = indexed.indexed_asset_resolver(index, current_machine_id="Main")
    with pytest.raises(ValueError, match="exact products-index closure"):
        resolve(dict(path="C:/unrelated/product.npz", sha256="f" * 64, size_bytes=1))


@pytest.mark.parametrize("adjacent", [False, True])
def test_real_N15_N16_writer_index_collect_and_foreign_extensive_reader(tmp_path, adjacent):
    from . import test_storage_n1_saved_products as fixture
    from darsia.fluidflower import storage_extensive_io as extensive_io
    producer = tmp_path / "producer"
    producer.mkdir()
    tiny = fixture.tiny_config.__wrapped__(producer)
    phase = fixture.phase1_data.__wrapped__()
    config = fixture._genuine_n16_config(producer, tiny, phase, adjacent=adjacent)
    completion = fixture._run(producer, config)
    index = fixture._index(tmp_path, completion)
    _, _, rows = publication._read_index(index)
    wanted = [key for key, row in rows.items() if "/storage_" in key and row["kind"] == "map"]
    collected = publication.collect_products(index, wanted, tmp_path / "collected", current_machine_id="Second",
        source_paths={key: row["local_path"] for key, row in rows.items() if key in wanted}, file_writers_closed=True)
    result = extensive_io.read_indexed_extensive_run(index, producer_kind="n1-mass-only",
        current_machine_id="Second", collections=[collected], qualification=extensive_io.GENUINE_QUALIFICATION)
    frames = list(result["frames"])
    assert result["ordered_case_ids"] == completion["ordered_case_ids"]
    assert len(frames) == 2
    for frame in frames:
        assert frame["state"] == "genuine_n15_n16_io"
        assert ("cell_U_g" in frame["arrays"]) is adjacent
        assert int(frame["arrays"]["hidden_pixel_count"].sum()) == 1
        assert all("/storage_" in str(bound["path"]).replace("\\", "/") for bound in frame["loaded_assets"])
