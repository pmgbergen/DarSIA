"""Fault injection for the two original cleanup sites; no real series or jobs.

Compile-only delivery. Default False must retain the original deletion behavior;
True preserves only the failed Store staging/unindexed artifact directory.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as subject


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("site", ["canonical", "materialized_validation"])
def test_failure_cleanup_is_opt_in_and_never_publishes_current(tmp_path, monkeypatch, preserve, site):
    project = SimpleNamespace(root=tmp_path / "project", manifest={})
    depth = tmp_path / "original-depth.npz"
    depth.write_bytes(b"source bytes, copied without decode in this fault test")
    config = subject.default_geometry_config(project)
    config.update(depth_path=str(depth), labels_path=str(tmp_path / "labels.npz"))
    validation_calls = []

    def validate(value):
        validation_calls.append(value)
        errors = int(site == "materialized_validation" and len(validation_calls) == 2)
        return {"config_sha256": subject.geometry_config_digest(value),
                "summary": {"error_count": errors}}

    def canonical(_value):
        if site == "canonical":
            raise RuntimeError("exact canonical fault")
        return np.array([[0]], dtype=np.int32), []

    monkeypatch.setattr(subject, "validate_geometry_config", validate)
    monkeypatch.setattr(subject, "_canonical_facies", canonical)
    monkeypatch.setattr(subject, "_darsia_metadata", lambda _path: {})
    store = subject.GeometryRevisionStore(project)
    expected = "exact canonical fault" if site == "canonical" else "Materialized geometry revision failed validation"
    error_type = RuntimeError if site == "canonical" else ValueError
    with pytest.raises(error_type, match=expected):
        store.save_revision(config, actor="synthetic", reason="fault test",
                            preserve_failed_artifacts=preserve)
    children = list(store.artifact_root.iterdir())
    assert len(children) == int(preserve)
    if preserve:
        partial = children[0]
        assert (partial / "depth.npz").read_bytes() == depth.read_bytes()
        if site == "canonical":
            assert partial.name.startswith(".") and partial.name.endswith(".tmp")
        else:
            assert len(partial.name) == 32 and (partial / "facies.npz").is_file()
        assert not (partial / "revision.json").exists()
    assert not store.current_path.exists() and not store.revisions_path.exists()
    assert project.manifest == {} and depth.read_bytes().startswith(b"source bytes")


def test_omitted_option_keeps_original_failure_cleanup(tmp_path, monkeypatch):
    project = SimpleNamespace(root=tmp_path / "project", manifest={})
    depth = tmp_path / "depth.npz"
    depth.write_bytes(b"original")
    config = subject.default_geometry_config(project)
    config["depth_path"] = str(depth)
    monkeypatch.setattr(subject, "validate_geometry_config", lambda value: {
        "config_sha256": subject.geometry_config_digest(value), "summary": {"error_count": 0}})

    def fail(_value):
        raise RuntimeError("original default fault")

    monkeypatch.setattr(subject, "_canonical_facies", fail)
    store = subject.GeometryRevisionStore(project)
    with pytest.raises(RuntimeError, match="original default fault"):
        store.save_revision(config, actor="synthetic", reason="default")
    assert list(store.artifact_root.iterdir()) == []
    assert not store.current_path.exists() and not store.revisions_path.exists()


# Reuse the original tiny 2x3 imported-raster fixture, not a campaign or a
# replacement validation report. Importing only these names adds no collection.
from tests.unit.test_geometry_imported_rasters import imported, save, snapshot


@pytest.mark.parametrize("preserve", [False, True])
def test_oversized_revision_preserves_arrays_and_refuses_all_metadata_publication(imported, monkeypatch, preserve):
    previous = save(imported)
    store = subject.GeometryRevisionStore(imported.project)
    before = snapshot(imported.project.root)
    source_before = snapshot(imported.original)
    previous_directories = set(store.artifact_root.iterdir())
    metadata_before = list(imported.project.metadata)
    audit_before = list(imported.project.audit)
    writes = []

    def forbidden(kind):
        def refuse(*args, **kwargs):
            writes.append(kind)
            pytest.fail("Oversized revision reached " + kind)
        return refuse

    original_write_text = Path.write_text

    def revision_write(path, *args, **kwargs):
        if path.name == "revision.json":
            return forbidden("revision.json")(path, *args, **kwargs)
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", revision_write)
    monkeypatch.setattr(subject, "_append_jsonl", forbidden("revision history"))
    monkeypatch.setattr(subject, "_atomic_json", forbidden("current.json"))
    monkeypatch.setattr(imported.project, "update_metadata", forbidden("project metadata"))
    from darsia.fluidflower.geometry_workspace_frame import METADATA_LIMIT

    with pytest.raises(ValueError, match="original 1 MiB metadata limit"):
        store.save_revision(imported.config, actor="synthetic owner",
            reason="x" * METADATA_LIMIT, preserve_failed_artifacts=preserve)

    assert writes == []
    current = snapshot(imported.project.root)
    assert all(current[path] == identity for path, identity in before.items())
    assert snapshot(imported.original) == source_before
    assert imported.project.metadata == metadata_before and imported.project.audit == audit_before
    fresh = set(store.artifact_root.iterdir()) - previous_directories
    assert len(fresh) == 1
    owned = fresh.pop()
    assert len(owned.name) == 32 and not owned.name.startswith(".")
    assert not (owned / "revision.json").exists()
    assert (owned / Path(previous["artifacts"]["depth_path"]).name).read_bytes() == Path(imported.config["depth_path"]).read_bytes()
    assert (owned / "labels.npz").is_file() and (owned / "facies.npz").is_file()
    assert (owned / "facies_properties.csv").read_bytes() == Path(imported.config["facies_properties_source_path"]).read_bytes()


def test_small_revision_keeps_original_serializer_bytes_and_sources(imported, tmp_path):
    import json
    from darsia.fluidflower.geometry_workspace_frame import METADATA_LIMIT

    before = snapshot(imported.original)
    revision = save(imported)
    expected = tmp_path / "original-writer.json"
    expected.write_text(json.dumps(revision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw = expected.read_bytes()
    store = subject.GeometryRevisionStore(imported.project)
    assert len(raw) <= METADATA_LIMIT
    assert (store.artifact_root / revision["revision_id"] / "revision.json").read_bytes() == raw
    assert store.current_path.read_bytes() == raw
    assert json.loads(store.revisions_path.read_text(encoding="utf-8")) == revision
    assert snapshot(imported.original) == before
    assert len(imported.project.metadata) == 1 and len(imported.project.audit) == 1


@pytest.mark.parametrize("room", [0, -1])
def test_metadata_limit_counts_utf8_and_original_native_newlines(tmp_path, monkeypatch, room):
    import json
    import os
    from darsia.fluidflower import geometry_workspace_frame as frame

    revision = {"actor": "synthetic \u00e6\u03bb", "nested": [{"original": True}, {"note": "line\nline"}]}
    text = json.dumps(revision, indent=2, ensure_ascii=False) + "\n"
    original = tmp_path / "original-text-writer.json"
    original.write_text(text, encoding="utf-8")
    raw = original.read_bytes()
    if os.linesep == "\r\n":
        assert len(raw) > len(text.encode("utf-8"))
    monkeypatch.setattr(frame, "METADATA_LIMIT", len(raw) + room)
    if room == 0:
        assert subject._bounded_revision_text(revision) == text
    else:
        with pytest.raises(ValueError, match="original 1 MiB metadata limit"):
            subject._bounded_revision_text(revision)
