"""Synthetic local metadata tests; no claim, worker, job or real FS probe."""
from pathlib import Path

import pytest

from darsia.fluidflower.job_filesystem import LocalFilesystemFacts

from darsia.fluidflower.store_map import (
    StoreMapError, default_store_map_path, input_reference, manifest_roots,
    manifest_root_proposal, parse_store_map, propose_manifest_roots, read_store_map, require_real_containment, save_store_map_text,
    resolve_input_reference, root_availability, validate_input_reference,
)


@pytest.fixture
def stores(tmp_path):
    path = tmp_path / "local_settings" / "stores.toml"
    path.parent.mkdir()
    path.write_text('[machine]\nid="Main"\n[stores.FF_AC]\nseries="D:/data/FF_AC"\nraw="D:/data/raw"\nlegacy_results="//nas/share/old"\nresults="E:/products"\n', encoding="utf-8")
    return path, read_store_map(path, locality_provider=lambda _: LocalFilesystemFacts("fixture-local", True, "synthetic proof"), sync_roots=[])


def test_read_only_map_and_cross_site_locator_leave_inputs_unchanged(stores):
    path, store = stores
    before = path.read_bytes()
    manifest = {"schema_version": 1, "series_id": "FF_AC", "roots": {
        "series": ".", "raw": "//NAS/source/raw", "legacy_results": "//NAS/source/results"}}
    roots = manifest_roots("//NAS/source/FF_AC/DarSIA", manifest)
    ref = input_reference("//NAS/source/raw/AC52/Ω image.JPG", roots,
                          sha256="a" * 64, size_bytes=12, role="image")
    assert ref == {"root_alias": "raw", "relative_path": "AC52/Ω image.JPG",
                   "sha256": "a" * 64, "bytes": 12, "role": "image"}
    assert resolve_input_reference(ref, store, "ff-ac") == r"D:\data\raw\AC52\Ω image.JPG"
    assert path.read_bytes() == before and store.machine_id == "Main"
    assert store.source["size_bytes"] == len(before)


def test_packaged_store_map_is_resolved_without_copying_or_changing_settings(tmp_path):
    packaged = tmp_path / "Packages" / "fixture-app" / "LocalCache/Local/DarSIA/stores.toml"
    packaged.parent.mkdir(parents=True)
    raw = b'[machine]\nid="fixture"\n[stores.sample]\nseries="D:/series"\nresults="D:/results"\n'
    packaged.write_bytes(raw)
    environment = {"LOCALAPPDATA": str(tmp_path)}
    assert default_store_map_path(environment=environment, platform="nt") == packaged.resolve()
    assert packaged.read_bytes() == raw
    assert not (tmp_path / "DarSIA/stores.toml").exists()

    other = tmp_path / "Packages" / "other-app" / "LocalCache/Local/DarSIA/stores.toml"
    other.parent.mkdir(parents=True)
    other.write_bytes(raw)
    with pytest.raises(StoreMapError, match="Several packaged"):
        default_store_map_path(environment=environment, platform="nt")
    primary = tmp_path / "DarSIA/stores.toml"
    primary.parent.mkdir()
    primary.write_bytes(raw)
    assert default_store_map_path(environment=environment, platform="nt") == primary.resolve()


@pytest.mark.parametrize("relative", ["../image.jpg", "/image.jpg", "C:/image.jpg", "C:image.jpg", "a/../b", "a//b", "./b", r"a\b", "a:stream", "a/NUL.txt", "trailing.", "bad?x"])
def test_input_path_cannot_escape_or_depend_on_machine_spelling(relative):
    ref = {"root_alias": "raw", "relative_path": relative, "sha256": "a" * 64,
           "bytes": 1, "role": "image"}
    with pytest.raises(StoreMapError):
        validate_input_reference(ref)


def test_longest_boundary_root_and_ambiguous_or_unknown_source_refuse():
    roots = {"series": "/datasets/FF_AC", "raw": "/datasets/FF_AC/data/raw"}
    kwargs = dict(sha256="a" * 64, size_bytes=0, role="configuration")
    assert input_reference("/datasets/FF_AC/data/raw/run/file", roots, **kwargs)["root_alias"] == "raw"
    for source, aliases in (("/datasets/FF_AC_extra/x", roots),
                            ("/datasets/FF_AC/data/raw/x", dict(roots, duplicate=roots["raw"]))):
        with pytest.raises(StoreMapError):
            input_reference(source, aliases, **kwargs)


def test_missing_local_alias_is_unavailable_not_original_path_fallback(stores):
    _path, store = stores
    ref = {"root_alias": "missing", "relative_path": "file.json", "sha256": "b" * 64,
           "bytes": 1, "role": "model"}
    with pytest.raises(StoreMapError, match="no configured root alias"):
        resolve_input_reference(ref, store, "FF_AC")


def test_store_map_outside_repo_and_series_is_required(tmp_path):
    for marker in (".git", "series.toml"):
        root = tmp_path / marker.replace(".", "_")
        root.mkdir()
        (root / marker).write_text("marker")
        with pytest.raises(StoreMapError, match="outside"):
            read_store_map(root / "stores.toml")


def test_duplicate_canonical_series_refuses_and_explicit_unc_results_are_recorded(stores):
    path, _store = stores
    raw = path.read_text()
    path.write_text(raw + '[stores.ffac]\nseries="D:/x"\nresults="D:/out"\n')
    with pytest.raises(StoreMapError, match="Duplicate"):
        read_store_map(path, locality_provider=lambda _: LocalFilesystemFacts("fixture-local", True, "synthetic proof"), sync_roots=[])
    path.write_text(raw.replace('results="E:/products"', 'results="//nas/share/out"'))
    parsed=read_store_map(path, locality_provider=lambda _: LocalFilesystemFacts("fixture-local", True, "synthetic proof"), sync_roots=[])
    assert parsed.roots("FF_AC")["results"].replace("\\","/")=="//nas/share/out"
    # Parsing records explicit DATA; actual output transport is separately admitted.


def test_converter_proposal_comes_only_from_literal_manifest_data():
    doc = {"legacy": {"raw_inputs": "../actual-raw", "previous_results": "//site/actual/results"}}
    assert manifest_root_proposal(doc) == {"series": ".", "raw": "../actual-raw", "legacy_results": "//site/actual/results"}
    assert "roots" not in doc
    with pytest.raises(StoreMapError, match="explicit bounded"):
        manifest_roots("/series", {"schema_version": 1, "series_id": "example"})


def test_availability_is_explicit_directory_only_and_never_creates(stores):
    path, store = stores
    before = path.read_bytes()
    seen = []
    report = root_availability(store, "FF_AC", is_directory=lambda path: seen.append(str(path)) or False)
    assert len(seen) == 4 and all(not row["available"] for row in report["roots"])
    assert all(not row["directory_created"] and not row["input_hashes_verified"] for row in report["roots"])
    assert path.read_bytes() == before


def test_real_containment_rejects_link_escape(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    (outside / "file").write_text("not a scientific input")
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Local fixture account cannot create a symlink")
    with pytest.raises(StoreMapError, match="through a link"):
        require_real_containment(root, root / "link/file")


def test_defaults_do_not_depend_on_repository_or_series():
    assert str(default_store_map_path(environment={"LOCALAPPDATA": "C:/Users/test/AppData/Local"}, platform="nt")).replace("\\", "/").endswith("/DarSIA/stores.toml")
    assert default_store_map_path(environment={"XDG_CONFIG_HOME": "/local/config"}, platform="posix") == Path("/local/config/darsia/stores.toml")


@pytest.mark.parametrize("local", [False, None])
def test_local_file_read_refuses_remote_or_unknown_before_open(tmp_path, local):
    path = tmp_path / "not-created" / "stores.toml"
    with pytest.raises(StoreMapError, match="positively identified local"):
        read_store_map(path, locality_provider=lambda _: LocalFilesystemFacts("fixture", local, "not local"), sync_roots=[])
    assert not path.parent.exists()


def test_missing_locally_admitted_map_remains_a_missing_file(tmp_path):
    path = tmp_path / "stores.toml"
    with pytest.raises(FileNotFoundError):
        read_store_map(path, locality_provider=lambda _: LocalFilesystemFacts("fixture", True, "local"), sync_roots=[])
    assert not path.exists()


def test_parser_does_no_filesystem_admission_and_preserves_unknown_values(monkeypatch):
    import darsia.fluidflower.store_map as module
    monkeypatch.setattr(module, "inspect_local_file", lambda *_a, **_k: pytest.fail("pure parse probed filesystem"))
    raw = b'[machine]\nid="Main"\nother=17\n[stores.S]\nseries="/series"\nresults="/products"\n[future]\nunknown=[1,2]\n'
    parsed = parse_store_map(raw)
    assert parsed.document["future"]["unknown"] == [1, 2]
    assert parsed.document["machine"]["other"] == 17
    assert "path" not in parsed.source
    with pytest.raises(StoreMapError, match="1 MiB"):
        parse_store_map(b" " * (1024 * 1024 + 1))


def test_explicit_save_preserves_exact_text_and_previous_bytes_and_refuses_stale(stores):
    path, store = stores
    before = path.read_bytes()
    after = before + b'\n[future]\ncustom = "keep me"\n'
    options = dict(locality_provider=lambda _: LocalFilesystemFacts("fixture", True, "local"), sync_roots=[])
    result = save_store_map_text(path, after, expected_bytes=before, **options)
    assert path.read_bytes() == after
    assert Path(result["previous_copy"]).read_bytes() == before
    assert result["sha256"] != store.source["sha256"]
    with pytest.raises(StoreMapError, match="changed since"):
        save_store_map_text(path, before, expected_bytes=before, **options)
    assert path.read_bytes() == after


def test_new_local_map_is_explicit_and_unknown_fs_never_creates_directory(tmp_path):
    path = tmp_path / "new-settings" / "stores.toml"
    raw = b'[machine]\nid="Main"\n[stores.S]\nseries="/series"\nresults="/products"\n'
    with pytest.raises(StoreMapError, match="positively identified local"):
        save_store_map_text(path, raw, expected_bytes=None, locality_provider=lambda _: LocalFilesystemFacts("unknown", None, "unproved"), sync_roots=[])
    assert not path.parent.exists()
    receipt = save_store_map_text(path, raw, expected_bytes=None, locality_provider=lambda _: LocalFilesystemFacts("fixture", True, "local"), sync_roots=[])
    assert path.read_bytes() == raw and receipt["previous_copy"] is None


def test_roots_proposal_appends_only_literal_sources_and_never_writes(tmp_path):
    raw = b'# keep exactly\r\nschema_version=1\r\nseries_id="S"\r\n[legacy]\r\nraw_inputs="../raw"\r\nprevious_results="/archive"\r\n[unknown]\r\ncustom=42\r\n'
    before = tuple(tmp_path.iterdir())
    result = propose_manifest_roots(tmp_path, raw)
    assert result.before == raw and result.after.startswith(raw)
    assert result.receipt["proposed_roots"] == {"series": ".", "raw": "../raw", "legacy_results": "/archive"}
    assert result.receipt["state"] == "proposal_only" and not result.receipt["applied"]
    assert tuple(tmp_path.iterdir()) == before
    second = propose_manifest_roots(tmp_path, result.after)
    assert second.after == result.after and second.receipt["state"] == "already_declared"
