"""Local metadata/byte fixtures only; no worker, science, NAS or remote process."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import products_publication as publication


def descriptor(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return descriptor(path)


def as_product(product_id, bound, *, kind, machine="Main"):
    return publication.product(product_id, machine_id=machine, local_path=bound["path"],
                               sha256=bound["sha256"], bytes=bound["size_bytes"], kind=kind)


def fixture(tmp_path):
    producer = tmp_path / "producer"
    raw = (b'\xef\xbb\xbf{\r\n  "complete": true, "worker_references_closed": true,\r\n'
           b'  "run": "ac52", "precision": "float32",\r\n'
           b'  "records": [{"nas_directory": "C:/actual-producer/old-attempt/frame-1"}],\r\n'
           b'  "scientific_value": 23.859055832362177\r\n}\r\n')
    completion = write(producer / "completed.json", raw)
    scalar = write(producer / "frames" / "f1" / "result.json", b'{"mass": 0.125}\r\n')
    manifest = write(producer / "manifest.json", b'{"precision":"float32"}\n')
    table = write(producer / "mass_table.csv", b"case,mass\r\nf1,0.125\r\n")
    film = write(producer / "film.mp4", b"synthetic film bytes\x00\xff")
    export = write(producer / "export.npz", b"synthetic lossless bytes\x80")
    maps = write(producer / "maps.npz", b"synthetic map bytes\x00")
    job = write(tmp_path / "job.json", b'{"format":"darsia-alias-job-v1"}\n')
    products = [
        as_product("frames/f1/result.json", scalar, kind="frame_scalar"),
        as_product("manifest.json", manifest, kind="manifest"),
        as_product("mass_table.csv", table, kind="scalar_table"),
        as_product("film.mp4", film, kind="film"),
        as_product("export.npz", export, kind="lossless_export"),
        as_product("maps.npz", maps, kind="map"),
    ]
    return dict(completion=completion, completed_raw=raw, job=job, products=products,
                scalar=scalar, film=film, export=export, maps=maps)


def publish(tmp_path, data, **overrides):
    arguments = dict(job=data["job"], run="ac52", attempt_id="a" * 32, machine_id="Main",
                     completed=data["completion"], products=data["products"],
                     metadata_product_ids=["frames/f1/result.json", "manifest.json", "mass_table.csv"],
                     file_writers_closed=True, all_final_products_listed=True)
    arguments.update(overrides)
    return publication.publish_run_metadata(tmp_path / "shared-metadata", **arguments)


def test_raw_main_completion_and_producer_paths_are_preserved(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    assert publication.read_published_metadata(index) == data["completed_raw"]
    assert Path(data["completion"]["path"]).read_bytes() == data["completed_raw"]
    saved = json.loads(Path(index["path"]).read_bytes())
    assert saved["file_closure"]["os_retirement_asserted"] is False
    assert saved["products"][0]["local_path"] == data["completion"]["path"]
    assert publication.read_published_metadata(index, "frames/f1/result.json") == b'{"mass": 0.125}\r\n'
    assert json.loads(publication.read_published_metadata(index).decode("utf-8-sig"))["records"][0]["nas_directory"] == "C:/actual-producer/old-attempt/frame-1"


def test_indexing_does_not_open_large_maps_or_walk_preparation(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    opened = []
    original = publication._file

    def observe(path):
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(publication, "_file", observe)
    index = publish(tmp_path, data)
    assert data["maps"]["path"] not in opened
    assert data["film"]["path"] not in opened
    assert data["export"]["path"] not in opened
    saved = json.loads(Path(index["path"]).read_bytes())
    assert len(saved["products"]) == 7
    assert all(row["identity_basis"] == "original_producer_descriptor" for row in saved["products"])


def test_default_publish_only_films_and_lossless_export_and_never_rewrites_index(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    original_index = Path(index["path"]).read_bytes()
    bundle = publication.publish_indexed_products(index, tmp_path / "shared-large", current_machine_id="Main", file_writers_closed=True)
    saved = json.loads(Path(bundle["path"]).read_bytes())
    assert [row["product_id"] for row in saved["copies"]] == ["film.mp4", "export.npz"]
    assert not (tmp_path / "shared-large" / "products" / "maps.npz").exists()
    assert Path(index["path"]).read_bytes() == original_index
    assert (tmp_path / "shared-large" / "products" / "film.mp4").read_bytes() == Path(data["film"]["path"]).read_bytes()


def test_map_is_copied_only_when_explicitly_selected(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    bundle = publication.publish_indexed_products(index, tmp_path / "explicit-map", publish=["map"], current_machine_id="Main", file_writers_closed=True)
    assert [row["product_id"] for row in json.loads(Path(bundle["path"]).read_bytes())["copies"]] == ["maps.npz"]


def test_collect_reads_shared_publication_on_another_machine_and_keeps_origin(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    bundle = publication.publish_indexed_products(index, tmp_path / "shared-large", current_machine_id="Main", file_writers_closed=True)
    Path(data["film"]["path"]).unlink()  # Original is deliberately unavailable to the consumer.
    collection = publication.collect_products(index, ["film.mp4", "completed.json"], tmp_path / "collected", current_machine_id="Second", payload_publications=[bundle], file_writers_closed=True)
    result = json.loads(Path(collection["path"]).read_bytes())
    assert result["copies"][0]["machine_id"] == "Main"
    assert result["copies"][0]["local_path"] == data["film"]["path"]
    assert (tmp_path / "collected" / "products" / "completed.json").read_bytes() == data["completed_raw"]


def test_foreign_local_source_is_not_even_probed_without_explicit_accessible_copy(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    forbidden = data["maps"]["path"]
    original = publication._file

    def reject_foreign(path):
        assert str(path) != forbidden, "Foreign producer local path must not be inspected"
        return original(path)

    monkeypatch.setattr(publication, "_file", reject_foreign)
    with pytest.raises(publication.ProductUnavailable, match="another machine"):
        publication.collect_products(index, ["maps.npz"], tmp_path / "foreign", current_machine_id="Second", file_writers_closed=True)
    assert not (tmp_path / "foreign").exists()


def test_explicit_accessible_copy_is_hash_verified_and_origin_is_not_relabelled(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    accessible = tmp_path / "mounted-original-copy.npz"
    accessible.write_bytes(Path(data["maps"]["path"]).read_bytes())
    collection = publication.collect_products(index, ["maps.npz"], tmp_path / "collected", current_machine_id="Second", source_paths={"maps.npz": accessible}, file_writers_closed=True)
    row = json.loads(Path(collection["path"]).read_bytes())["copies"][0]
    assert row["machine_id"] == "Main"
    assert row["local_path"] == data["maps"]["path"]
    assert row["sha256"] == data["maps"]["sha256"]


def test_resumed_frame_metadata_keeps_old_machine_and_uses_only_explicit_copy(tmp_path):
    data = fixture(tmp_path)
    products = [dict(row) for row in data["products"]]
    products[0]["machine_id"] = "Second"
    products[0]["local_path"] = "Z:/original-second-attempt/frames/f1/result.json"
    with pytest.raises(publication.ProductUnavailable):
        publish(tmp_path, data, products=products)
    assert not (tmp_path / "shared-metadata").exists()
    index = publish(tmp_path, data, products=products, source_paths={"frames/f1/result.json": data["scalar"]["path"]})
    row = json.loads(Path(index["path"]).read_bytes())["products"][1]
    assert row["machine_id"] == "Second"
    assert row["local_path"].startswith("Z:/original-second-attempt/")
    assert publication.read_published_metadata(index, "frames/f1/result.json") == Path(data["scalar"]["path"]).read_bytes()


@pytest.mark.parametrize("field,value", [("file_writers_closed", False), ("all_final_products_listed", False)])
def test_no_publication_without_explicit_closure_and_complete_list(tmp_path, field, value):
    data = fixture(tmp_path)
    with pytest.raises(ValueError):
        publish(tmp_path, data, **{field: value})
    assert not (tmp_path / "shared-metadata").exists()


def test_false_reference_closure_is_not_misrepresented_as_completion(tmp_path):
    data = fixture(tmp_path)
    raw = b'{"complete":true,"run":"ac52","worker_references_closed":false}'
    data["completion"] = write(Path(data["completion"]["path"]), raw)
    with pytest.raises(ValueError, match="closed-reference"):
        publish(tmp_path, data)
    assert not (tmp_path / "shared-metadata").exists()


def test_changed_source_never_publishes_an_index_or_final_mismatched_file(tmp_path):
    data = fixture(tmp_path)
    Path(data["scalar"]["path"]).write_bytes(b'{"mass": 9.999}\r\n')
    with pytest.raises(ValueError, match="differs"):
        publish(tmp_path, data)
    assert not (tmp_path / "shared-metadata" / "products_index.json").exists()
    assert not (tmp_path / "shared-metadata" / "metadata" / "frames" / "f1" / "result.json").exists()
    assert Path(data["scalar"]["path"]).read_bytes() == b'{"mass": 9.999}\r\n'


def test_collect_mismatch_preserves_partial_and_never_reports_completion(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    Path(data["film"]["path"]).write_bytes(b"wrong same-length filmxxx")
    with pytest.raises(ValueError):
        publication.collect_products(index, ["film.mp4"], tmp_path / "target", current_machine_id="Main", file_writers_closed=True)
    assert not (tmp_path / "target" / "collection.json").exists()
    assert not (tmp_path / "target" / "products" / "film.mp4").exists()


def test_conflicting_target_is_never_overwritten(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    target = tmp_path / "target"
    target.mkdir()
    sentinel = write(target / "user-file.txt", b"unchanged")
    with pytest.raises(FileExistsError):
        publication.collect_products(index, ["completed.json"], target, current_machine_id="Main", file_writers_closed=True)
    assert descriptor(target / "user-file.txt") == sentinel
    with pytest.raises(FileExistsError):
        publish(tmp_path, data)


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "C:/drive", "a\\b", "a//b", "a/./b", "a/../b", "NUL.txt", "a:stream", "trailing."])
def test_product_path_traversal_drive_and_windows_ambiguity_are_rejected(tmp_path, bad):
    data = fixture(tmp_path)
    with pytest.raises(ValueError):
        as_product(bad, data["film"], kind="film")


def test_case_colliding_products_are_rejected_before_any_publication(tmp_path):
    data = fixture(tmp_path)
    products = data["products"] + [dict(data["products"][3], product_id="Film.mp4")]
    with pytest.raises(ValueError, match="Conflicting"):
        publish(tmp_path, data, products=products)
    assert not (tmp_path / "shared-metadata").exists()


def test_symlink_sources_and_target_ancestors_are_refused(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    link = tmp_path / "map-link.npz"
    directory_link = tmp_path / "target-link"
    try:
        link.symlink_to(Path(data["maps"]["path"]))
        directory_link.symlink_to(tmp_path / "producer", target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("This runtime does not permit local symlink fixtures")
    with pytest.raises(ValueError, match="Symlink/reparse"):
        publication.collect_products(index, ["maps.npz"], tmp_path / "target", current_machine_id="Main", source_paths={"maps.npz": link}, file_writers_closed=True)
    with pytest.raises(ValueError, match="Symlink/reparse"):
        publication.collect_products(index, ["maps.npz"], directory_link / "new", current_machine_id="Main", file_writers_closed=True)


def test_tampered_shared_metadata_never_falls_back_to_original(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    (tmp_path / "shared-metadata" / "completed.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Metadata"):
        publication.read_published_metadata(index)
    assert Path(data["completion"]["path"]).read_bytes() == data["completed_raw"]


def test_index_sha_change_and_payload_bundle_from_another_index_are_refused(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    bundle = publication.publish_indexed_products(index, tmp_path / "payload", current_machine_id="Main", file_writers_closed=True)
    # A relocated byte-identical index is intentionally the same identity.
    # Produce a genuinely different job/product index for this rejection case.
    other_root = tmp_path / "another-job"
    other_data = fixture(other_root)
    other = publish(other_root, other_data)
    assert other["sha256"] != index["sha256"]
    with pytest.raises(ValueError, match="differs"):
        publication.collect_products(other, ["film.mp4"], tmp_path / "target", current_machine_id="Second", payload_publications=[bundle], file_writers_closed=True)
    Path(index["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="Metadata"):
        publication.read_published_metadata(index)


def test_late_parent_retirement_is_separate_immutable_raw_receipt(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    index_before = Path(index["path"]).read_bytes()
    raw = b'{"complete":true,"retirement_basis":"owned_process_incarnations_absent"}\r\n'
    bound = write(tmp_path / "producer" / "retirement.json", raw)
    late = publication.publish_late_receipts(index, [as_product("retirement.json", bound, kind="operational_receipt")], tmp_path / "late", current_machine_id="Main", file_writers_closed=True)
    assert (tmp_path / "late" / "receipts" / "retirement.json").read_bytes() == raw
    assert Path(index["path"]).read_bytes() == index_before
    assert json.loads(Path(late["path"]).read_bytes())["retirement_claim_scope"].startswith("Only the unchanged")


def test_explicit_empty_publish_and_invalid_string_list(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    bundle = publication.publish_indexed_products(index, tmp_path / "empty", publish=[], current_machine_id="Main", file_writers_closed=True)
    assert json.loads(Path(bundle["path"]).read_bytes())["copies"] == []
    with pytest.raises(ValueError, match="explicit list"):
        publication.publish_indexed_products(index, tmp_path / "bad", publish="film", current_machine_id="Main", file_writers_closed=True)
    with pytest.raises(ValueError, match="explicit product-id"):
        publication.collect_products(index, "film.mp4", tmp_path / "bad-collect", current_machine_id="Main", file_writers_closed=True)


def test_actual_size_large_run_manifest_streams_without_raising_existing_reader_caps(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    size = 7_141_781  # Actual immutable Task1 AC52 run-manifest byte count.
    prefix, suffix = b'{"synthetic_metadata":"', b'"}\n'
    raw = prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix
    bound = write(tmp_path / "producer" / "manifest.json", raw)
    products = [dict(row) for row in data["products"]]
    products[1] = as_product("manifest.json", bound, kind="manifest")
    original_read_bytes = Path.read_bytes

    def no_manifest_buffer(path):
        assert path.name != "manifest.json", "Descriptor/copy must stream the large manifest"
        return original_read_bytes(path)

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", no_manifest_buffer)
        index = publish(tmp_path, data, products=products)
        shared = publication.published_metadata_descriptor(index, "manifest.json")
    assert shared["bytes"] == size
    assert shared["sha256"] == bound["sha256"]
    assert publication.read_published_metadata(index, "manifest.json") == raw
    assert publication.MAX_METADATA_BYTES == 128 * 1024 * 1024


def test_explicit_publication_metadata_limit_is_still_enforced_before_any_copy(tmp_path):
    data = fixture(tmp_path)
    products = [dict(row) for row in data["products"]]
    products[1]["bytes"] = publication.MAX_METADATA_BYTES + 1
    with pytest.raises(ValueError, match="Large/nonmetadata"):
        publish(tmp_path, data, products=products)
    assert not (tmp_path / "shared-metadata").exists()


def test_payload_publication_keeps_original_path_but_collects_after_shared_bundle_rename(tmp_path):
    data = fixture(tmp_path)
    lease = tmp_path / "lease"
    index = publication.publish_run_metadata(lease / "artifacts", job=data["job"],
        run="ac52", attempt_id="a" * 32, machine_id="Main", completed=data["completion"],
        products=data["products"], metadata_product_ids=["frames/f1/result.json"],
        file_writers_closed=True, all_final_products_listed=True)
    payload = publication.publish_indexed_products(index, lease / "artifacts/default_products",
        current_machine_id="Main", file_writers_closed=True)
    original_receipt = Path(payload["path"]).read_bytes()
    original_index = Path(index["path"]).read_bytes()
    moved = tmp_path / "publishing/ac52"
    moved.parent.mkdir()
    lease.rename(moved)
    moved_index = dict(index, path=str(moved / "artifacts/products_index.json"))
    moved_payload = dict(payload, path=str(moved / "artifacts/default_products/publication.json"))
    # Same immutable metadata, including its original now-inaccessible path.
    assert Path(moved_index["path"]).read_bytes() == original_index
    assert Path(moved_payload["path"]).read_bytes() == original_receipt
    assert not Path(index["path"]).exists()
    for item in (data["film"], data["export"]):
        Path(item["path"]).unlink()  # Explicit synthetic original is unavailable.
    result = publication.collect_products(moved_index, ["film.mp4", "export.npz"],
        tmp_path / "collected", current_machine_id="Another-machine",
        payload_publications=[moved_payload], file_writers_closed=True)
    saved = json.loads(Path(result["path"]).read_bytes())
    assert len(saved["copies"]) == 2
    assert (tmp_path / "collected/products/film.mp4").read_bytes() == b"synthetic film bytes\x00\xff"


def test_moved_payload_receipt_cannot_bind_an_index_with_different_bytes(tmp_path):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    payload = publication.publish_indexed_products(index, tmp_path / "payload",
        current_machine_id="Main", file_writers_closed=True)
    forged = dict(index, sha256="0" * 64)
    with pytest.raises(ValueError, match="Metadata bytes differ"):
        publication.collect_products(forged, ["film.mp4"], tmp_path / "collected",
            current_machine_id="Other", payload_publications=[payload], file_writers_closed=True)
    assert not (tmp_path / "collected").exists()
