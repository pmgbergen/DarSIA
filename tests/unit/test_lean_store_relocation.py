"""Real Lean output read through independent, exact original-asset locators.

These tests run the existing synthetic Full/Lean producer fixture. They do not
qualify a machine, a campaign, a queue, or scientific accuracy on real data.
"""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from darsia.fluidflower import storage_profile as lean
from darsia.fluidflower import storage_indexed_products as indexed
from darsia.fluidflower import storage_saved_products as saved
from darsia.fluidflower.mass import n1_pipeline as n1
from darsia.fluidflower.pipeline import products_publication as publication
from darsia.fluidflower.pipeline import publication_runtime as runtime
from darsia.fluidflower.pipeline.job_inputs import InputCatalog

from .test_lean_storage import synthetic, _full_and_lean, _frames, _archive, _same_bits  # noqa: F401


@pytest.fixture
def produced(synthetic, tmp_path):
    full, outcome = _full_and_lean(tmp_path, synthetic)
    frames = list(_frames(outcome))
    expected = [_archive(root / "native_geometry.npz") for root, _ in _frames(full)]
    assert all(result["precision"] == "float32" for _, result in frames)
    assert all(result["static"]["storage"]["written"] == [] for _, result in frames)
    return SimpleNamespace(outcome=outcome, frames=frames, expected=expected,
        store=outcome["lean_storage"]["store"],
        original_roots=(tmp_path / "lean", tmp_path / "synthetic"))


def _copy(bound, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(Path(bound["path"]).read_bytes())
    return dict(logical=deepcopy(bound), physical=n1.asset(destination))


def _retained_aliases(produced, destination):
    """Keep external regeneration inputs separate from published final products."""
    aliases, seen = [], set()
    for _, result in produced.frames:
        inputs = result["static"]["storage"]["regeneration"]["inputs"]
        for bound in inputs.values():
            if bound["path"] not in seen:
                seen.add(bound["path"])
                aliases.append(_copy(bound, destination / (str(len(aliases)) + ".npz")))
    return aliases


def _relocated(produced, tmp_path):
    aliases = [
        _copy(produced.store["index"], tmp_path / "metadata-copy" / "index.json"),
        _copy(produced.store["archive"], tmp_path / "payload-copy" / "archive.npz"),
        *_retained_aliases(produced, tmp_path / "retained-inputs"),
    ]
    assert not (Path(aliases[0]["physical"]["path"]).parent / lean.STORE_ARCHIVE).exists()
    return aliases


def _forbid_originals(monkeypatch, roots):
    """A foreign read must neither open nor stat the producer's old locations."""
    prefixes = [str(path).replace("\\", "/").casefold().rstrip("/") for path in roots]
    original_open, original_stat = Path.open, Path.stat

    def check(path):
        text = str(path).replace("\\", "/").casefold()
        assert not any(text == prefix or text.startswith(prefix + "/") for prefix in prefixes), \
            "Readback probed an original producer/input path: " + str(path)

    def opened(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    def stated(path, *args, **kwargs):
        check(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(Path, "stat", stated)


def _assert_static(produced, reader):
    for (root, result), expected in zip(produced.frames, produced.expected):
        actual = lean.read_static_fields(result, root, reader=reader)
        assert set(actual) == set(expected)
        for name, value in expected.items():
            assert _same_bits(actual[name], value), name


def test_independently_relocated_store_and_retained_inputs_match_full(produced, tmp_path, monkeypatch):
    aliases = _relocated(produced, tmp_path)
    before = deepcopy(produced.frames)
    index_bytes = Path(aliases[0]["physical"]["path"]).read_bytes()
    catalog = InputCatalog({"copies": str(tmp_path)}, aliases)
    document = catalog.doc(produced.store["index"], "configuration", "lean.static.index")
    archive = catalog.add(produced.store["archive"], "rig_model", "lean.static.archive")
    assert archive["relative_path"] == "payload-copy/archive.npz"
    assert document["archive"]["path"] == lean.STORE_ARCHIVE
    assert document["sources"] == json.loads(index_bytes)["sources"]
    _forbid_originals(monkeypatch, produced.original_roots)
    _assert_static(produced, n1.DataReader(aliases))
    assert produced.frames == before
    assert Path(aliases[0]["physical"]["path"]).read_bytes() == index_bytes


@pytest.mark.parametrize("original", [
    r"Z:\producer\attempt\lean_run_static.json",
    r"\\producer\results\attempt\lean_run_static.json",
    "/foreign/producer/attempt/lean_run_static.json",
])
def test_foreign_logical_path_syntax_is_resolved_before_local_io(produced, tmp_path, monkeypatch, original):
    aliases = _relocated(produced, tmp_path)
    logical_index = dict(produced.store["index"], path=original)
    parent = PureWindowsPath(original) if "\\" in original else PurePosixPath(original)
    logical_archive = dict(produced.store["archive"], path=str(parent.parent / lean.STORE_ARCHIVE))
    aliases[0]["logical"], aliases[1]["logical"] = logical_index, logical_archive
    expected = _archive(produced.store["archive"]["path"])
    _forbid_originals(monkeypatch, (*produced.original_roots, parent.parent))
    store = lean.load_store(logical_index, reader=n1.DataReader(aliases))
    assert store.index == logical_index
    for name, value in expected.items():
        assert _same_bits(store.arrays[name], value), name


@pytest.mark.parametrize("role", ["index", "archive", "area", "classes"])
def test_relocated_store_and_retained_input_bytes_remain_verified(produced, tmp_path, role):
    aliases = _relocated(produced, tmp_path)
    logical = (produced.store[role] if role in produced.store else
        produced.frames[0][1]["static"]["storage"]["regeneration"]["inputs"][role])
    row = next(row for row in aliases if row["logical"]["path"] == logical["path"])
    target = Path(row["physical"]["path"])
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 1  # Same size: the digest, not just stat, must reject it.
    target.write_bytes(raw)
    with pytest.raises(ValueError, match="Data SHA or identity"):
        _assert_static(produced, n1.DataReader(aliases))


@pytest.mark.parametrize("field", ["sha256", "size_bytes"])
def test_original_reader_and_catalog_refuse_alias_identity_changes(produced, tmp_path, field):
    aliases = _relocated(produced, tmp_path)
    changed = deepcopy(aliases)
    changed[1]["physical"][field] = "0" * 64 if field == "sha256" else changed[1]["physical"][field] + 1
    with pytest.raises(ValueError, match="alias changes"):
        n1.DataReader(changed)
    with pytest.raises(ValueError, match="alias changes"):
        InputCatalog({"copies": str(tmp_path)}, changed)


@pytest.mark.parametrize("alteration", ["archive_name", "archive_sha", "archive_size", "array_role"])
def test_resealed_index_cannot_change_archive_or_array_binding(produced, tmp_path, alteration):
    aliases = _relocated(produced, tmp_path)
    document = json.loads(Path(aliases[0]["physical"]["path"]).read_bytes())
    if alteration == "archive_name":
        document["archive"]["path"] = "another.npz"
    elif alteration == "archive_sha":
        document["archive"]["sha256"] = "0" * 64
    elif alteration == "archive_size":
        document["archive"]["size_bytes"] += 1
    else:
        document["arrays"]["physical_facies"] = deepcopy(document["arrays"]["physical_domain"])
    replacement = n1._write_json(tmp_path / "resealed-index.json", document)
    logical = dict(replacement, path=produced.store["index"]["path"])
    aliases[0] = dict(logical=logical, physical=replacement)
    with pytest.raises(ValueError):
        lean.load_store(logical, reader=n1.DataReader(aliases))


def _published(produced, tmp_path):
    """Publish actual producer inventories; retained RAW-derived inputs stay inputs."""
    rows, metadata = [], []

    def add(key, bound, kind):
        rows.append(publication.product(key, machine_id="Main", local_path=bound["path"],
            sha256=bound["sha256"], bytes=bound["size_bytes"], kind=kind))
        if kind in runtime._METADATA:
            metadata.append(key)

    add("manifest.json", produced.outcome["manifest"], "manifest")
    if produced.outcome["resume_checkpoint"] is not None:
        add("resume.json", produced.outcome["resume_checkpoint"], "operational_receipt")
    for record in produced.outcome["records"]:
        for leaf, bound in record["files"].items():
            add("frames/" + record["frame_id"] + "/" + leaf,
                dict(bound, path=str(Path(record["nas_directory"]) / leaf)), runtime._kind(leaf))
    for producer in produced.outcome["frame_producers"]:
        for field in ("manifest", "commit", "frame_registration", "publication", "result"):
            add("producer/" + producer["case_id"] + "/" + field + ".json",
                producer[field], "operational_receipt")
    add("run-static/" + lean.STORE_INDEX, produced.store["index"], "operational_receipt")
    add("run-static/" + lean.STORE_ARCHIVE, produced.store["archive"], "map")
    completed = n1._write_json(tmp_path / "completed.json", produced.outcome)
    job = n1._write_json(tmp_path / "publication-identity.json", dict(synthetic_unit=True))
    index = publication.publish_run_metadata(tmp_path / "published-metadata", job=job,
        run=produced.outcome["run"], attempt_id="a" * 32, machine_id="Main", completed=completed,
        products=rows, metadata_product_ids=metadata, file_writers_closed=True, all_final_products_listed=True)
    payloads = publication.publish_indexed_products(index, tmp_path / "published-maps",
        current_machine_id="Main", publish=["map"], file_writers_closed=True)
    return index, payloads


def _published_aliases(produced, index, payloads):
    resolve = indexed.indexed_asset_resolver(index, current_machine_id="Second", collections=[payloads])
    return [dict(logical=deepcopy(bound), physical=dict(bound, path=str(resolve(bound))))
            for bound in produced.store.values()]


def test_foreign_reader_uses_real_separate_metadata_and_map_publications(produced, tmp_path, monkeypatch):
    index, payloads = _published(produced, tmp_path)
    aliases = _published_aliases(produced, index, payloads)
    assert Path(aliases[0]["physical"]["path"]).parent != Path(aliases[1]["physical"]["path"]).parent
    aliases += _retained_aliases(produced, tmp_path / "retained-inputs")
    _forbid_originals(monkeypatch, produced.original_roots)
    _assert_static(produced, n1.DataReader(aliases))


def test_foreign_unpublished_archive_requires_collection_without_producer_probe(produced, tmp_path, monkeypatch):
    index, _ = _published(produced, tmp_path)
    _forbid_originals(monkeypatch, produced.original_roots)
    resolve = indexed.indexed_asset_resolver(index, current_machine_id="Second")
    assert Path(resolve(produced.store["index"])).is_file()
    with pytest.raises(saved.StoredProductCollectionRequired) as caught:
        resolve(produced.store["archive"])
    assert caught.value.descriptor == produced.store["archive"]


@pytest.mark.parametrize("role", ["index", "archive"])
def test_foreign_published_store_bytes_are_still_verified(produced, tmp_path, role):
    index, payloads = _published(produced, tmp_path)
    aliases = _published_aliases(produced, index, payloads)
    row = next(row for row in aliases if row["logical"] == produced.store[role])
    target = Path(row["physical"]["path"])
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Data size differs"):
        lean.load_store(produced.store["index"], reader=n1.DataReader(aliases))


@pytest.mark.parametrize("field,value", [
    ("sha256", "f" * 64), ("machine_id", "Other"),
    ("local_path", "Z:/another/lean_run_static.npz"),
    ("product_id", "run-static/" + lean.STORE_INDEX),
])
def test_foreign_publication_cannot_rebind_archive_role(produced, tmp_path, field, value):
    index, payloads = _published(produced, tmp_path)
    document = json.loads(Path(payloads["path"]).read_bytes())
    copy = next(row for row in document["copies"] if row["product_id"] == "run-static/" + lean.STORE_ARCHIVE)
    copy[field] = value
    changed = n1._write_json(tmp_path / "forged-publication.json", document)
    with pytest.raises(ValueError):
        indexed.indexed_asset_resolver(index, current_machine_id="Second", collections=[changed])
