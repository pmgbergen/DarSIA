"""Source-phase archive coverage; no training, image loading or numerical work."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower import native_phase_model_archive as archive
from darsia.fluidflower import phase_model_library as library
from darsia.fluidflower.series_method import MethodSelection


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def model(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    (series / "series.toml").write_text('[series]\nid="ffac"\n', encoding="utf-8")
    root = series / "models" / "phase-model-fixture"
    write(root / "summary.json", dict(model_id=root.name, inventory={"datasets": ["ffac"]}))
    write(root / "training_request.json", dict(model_id=root.name, settings={"precision": "float32"}))
    write(root / "validation" / "boxes.json", {"regions": [{"box": [1, 2, 3, 4]}]})
    pointer = library.index_model(series, root)
    entry = dict(root=str(root), model_id=root.name, library_reference=str(pointer),
                 library_source_identity=library.source_identity(root), library_available=True)
    return series, root, pointer, entry


def choose(series, reference):
    (series / "series.toml").write_text(
        '[series]\nid="ffac"\n[method]\nactive="phase_model"\n'
        '[method.phase_model]\nmodel_version=' + json.dumps(str(reference))
        + '\nmass_calibration="saved-mass"\n', encoding="utf-8")


def test_archive_restore_preserves_science_pointers_boxes_and_choice(model):
    series, root, pointer, entry = model
    before = {p: p.read_bytes() for p in series.rglob("*") if p.is_file()}
    result = archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    assert archive.filter_entries(series, [entry]) == []
    assert all(p.read_bytes() == value for p, value in before.items())
    assert library.resolve_reference(pointer) == root
    assert library.library_records(series)[0]["root"] == str(root)
    archive.restore_model(series, entry, "Reviewer", "Compare the earlier candidate")
    assert archive.filter_entries(series, [entry]) == [entry]
    saved = json.loads(result.read_bytes())["models"][0]
    assert saved["archived"] is False
    assert [row["action"] for row in saved["history"]] == [
        "archived from model lists", "restored to model lists"]
    assert saved["history"][0]["person"] == "Reviewer"
    assert saved["history"][0]["reason"] == "Earlier candidate"
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("kind", ["id", "root", "pointer", "relative_root"])
def test_saved_series_model_is_protected(model, kind):
    series, root, pointer, entry = model
    reference = {"id": root.name, "root": root, "pointer": pointer,
                 "relative_root": root.relative_to(series).as_posix()}[kind]
    choose(series, reference)
    with pytest.raises(ValueError, match="saved for this series"):
        archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    assert not archive.archive_path(series).exists()


def test_active_model_cannot_be_archived_and_stays_visible_if_later_active(model):
    series, root, pointer, entry = model
    with pytest.raises(ValueError, match="active review or training"):
        archive.archive_model(series, entry, "Reviewer", "Earlier candidate", [root])
    archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    assert archive.filter_entries(series, [entry], [root / "."]) == [entry]
    choose(series, root.name)
    assert archive.filter_entries(series, [entry]) == [entry]


@pytest.mark.parametrize("broken", ['[method]\nactive="phase_model"\n', '[method\n'])
def test_unreadable_or_missing_saved_reference_fails_closed(model, broken):
    series, root, pointer, entry = model
    (series / "series.toml").write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError):
        archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    with pytest.raises(ValueError):
        archive.filter_entries(series, [entry])
    assert not (series / "models" / "phase" / "ui" / "browser_archive.json").exists()


def test_stale_source_and_unbound_catalogue_are_refused(model):
    series, root, pointer, entry = model
    unbound = dict(entry)
    unbound.pop("library_source_identity")
    with pytest.raises(ValueError, match="Import or reindex"):
        archive.archive_model(series, unbound, "Reviewer", "Earlier candidate")
    write(root / "summary.json", dict(model_id=root.name, changed=True))
    with pytest.raises(ValueError, match="identity changed"):
        archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    assert not archive.archive_path(series).exists()


def test_archived_identity_cannot_hide_a_replaced_model_or_lose_history(model):
    series, root, pointer, entry = model
    path = archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    original = path.read_bytes()
    write(root / "summary.json", dict(model_id=root.name, changed=True))
    fresh = dict(entry, source_identity=library.source_identity(root))
    fresh.pop("library_source_identity")
    fresh.pop("library_reference")
    with pytest.raises(ValueError, match="identity changed"):
        archive.filter_entries(series, [fresh])
    with pytest.raises(ValueError, match="identity changed"):
        archive.restore_model(series, fresh, "Reviewer", "Compare")
    assert path.read_bytes() == original


def test_retargeted_pointer_does_not_archive_the_old_selected_folder(model):
    series, root, pointer, entry = model
    other = root.with_name("phase-model-copy")
    for name in ("summary.json", "training_request.json"):
        write(other / name, json.loads((root / name).read_bytes()))
    value = json.loads(pointer.read_bytes())
    value["target"] = dict(relative_to="series_root", path=other.relative_to(series).as_posix())
    write(pointer, value)
    with pytest.raises(ValueError, match="changed its model target"):
        archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    assert not archive.archive_path(series).exists()


def test_repeat_action_is_idempotent_and_needs_person_and_reason(model):
    series, root, pointer, entry = model
    with pytest.raises(ValueError, match="person and reason"):
        archive.archive_model(series, entry, "", "Earlier candidate")
    with pytest.raises(ValueError, match="person and reason"):
        archive.archive_model(series, entry, "Reviewer", " ")
    path = archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    original = path.read_bytes()
    archive.archive_model(series, entry, "Reviewer", "Still archived")
    assert path.read_bytes() == original


def test_concurrent_saved_choice_change_does_not_save_overlay(model, monkeypatch):
    series, root, pointer, entry = model
    original_load = archive._load

    def load(path):
        result = original_load(path)
        choose(series, root.name)
        return result

    monkeypatch.setattr(archive, "_load", load)
    with pytest.raises(RuntimeError, match="series choice changed"):
        archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    assert not archive.archive_path(series).exists()


def test_concurrent_overlay_change_is_preserved(model, monkeypatch):
    series, root, pointer, entry = model
    path = archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    external = deepcopy(json.loads(path.read_bytes()))
    external["external_note"] = "Keep this external change"
    original_checked = archive._checked_entry
    calls = []

    def checked(*args, **kwargs):
        result = original_checked(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            write(path, external)
        return result

    monkeypatch.setattr(archive, "_checked_entry", checked)
    with pytest.raises(RuntimeError, match="external edit was preserved"):
        archive.restore_model(series, entry, "Reviewer", "Compare")
    assert json.loads(path.read_bytes()) == external


def test_malformed_overlay_does_not_hide_entries_or_get_overwritten(model):
    series, root, pointer, entry = model
    path = archive.archive_path(series)
    write(path, dict(format="unknown", models=[]))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="another format"):
        archive.filter_entries(series, [entry])
    with pytest.raises(ValueError, match="another format"):
        archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    assert path.read_bytes() == original


def test_shared_library_overlay_cannot_hide_another_series_models(model, tmp_path):
    series, root, pointer, entry = model
    archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    other = tmp_path / "other"
    other.mkdir()
    (other / "series.toml").write_text(
        '[series]\nid="other"\n[paths]\nphase_model_library='
        + json.dumps(str(library.library_root(series))) + '\n', encoding="utf-8")
    with pytest.raises(ValueError, match="another series"):
        archive.filter_entries(other, [entry])
    with pytest.raises(ValueError, match="another series"):
        archive.restore_model(other, entry, "Reviewer", "Compare")


def bomb_filesystem_filter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached model filtering must never access the filesystem")

    for name in ("read_bytes", "read_text", "open", "stat", "lstat", "resolve",
                 "samefile", "exists", "is_file", "is_dir"):
        monkeypatch.setattr(Path, name, forbidden)
    for name in ("source_identity", "resolve_reference", "resolve_review_path",
                 "read_series_method", "library_root", "_load", "_root", "_same_root"):
        monkeypatch.setattr(archive, name, forbidden)


def test_cached_filter_uses_only_exact_metadata_with_all_file_methods_blocked(model, monkeypatch):
    series, root, pointer, entry = model
    archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    snapshot = archive.read_archive_snapshot(series)
    original = deepcopy(snapshot)
    different_root = dict(entry, root=str(root.with_name("phase-model-copy")))
    with monkeypatch.context() as guarded:
        bomb_filesystem_filter(guarded)
        assert archive.filter_snapshot(snapshot, [entry], MethodSelection()) == []
        assert archive.filter_snapshot(snapshot, [different_root], MethodSelection()) == [different_root]
    assert snapshot == original


@pytest.mark.parametrize("kind", ["id", "root", "pointer", "relative_root", "active"])
def test_cached_saved_and_active_models_remain_visible_without_filesystem(model, monkeypatch, kind):
    series, root, pointer, entry = model
    archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    snapshot = archive.read_archive_snapshot(series)
    reference = {"id": root.name, "root": str(root), "pointer": str(pointer),
                 "relative_root": root.relative_to(series).as_posix(), "active": ""}[kind]
    saved = MethodSelection(active="phase_model" if reference else "", model_version=reference)
    active = [root] if kind == "active" else []
    # Being protected must also keep a stale catalogue entry visible.
    entry = dict(entry)
    entry.pop("library_source_identity")
    with monkeypatch.context() as guarded:
        bomb_filesystem_filter(guarded)
        assert archive.filter_snapshot(snapshot, [entry], saved, active) == [entry]


@pytest.mark.parametrize("kind", ["unknown", "stale", "unavailable"])
def test_cached_filter_refuses_unknown_or_stale_identity_without_filesystem(model, monkeypatch, kind):
    series, root, pointer, entry = model
    archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    snapshot = archive.read_archive_snapshot(series)
    entry = deepcopy(entry)
    if kind == "unknown":
        entry.pop("library_source_identity")
    elif kind == "stale":
        entry["library_source_identity"]["version_sha256"] = "0" * 64
    else:
        entry["library_available"] = False
    with monkeypatch.context() as guarded:
        bomb_filesystem_filter(guarded)
        with pytest.raises(ValueError, match="unknown or changed"):
            archive.filter_snapshot(snapshot, [entry], MethodSelection())


def test_snapshot_read_does_not_inspect_model_sources_or_saved_choice(model, monkeypatch):
    series, root, pointer, entry = model
    archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    path = archive.archive_path(series)

    def forbidden(*args, **kwargs):
        raise AssertionError("Snapshot loading may only read the configured overlay")

    monkeypatch.setattr(archive, "archive_path", lambda selected: path)
    for name in ("source_identity", "resolve_reference", "read_series_method", "_root", "_same_root"):
        monkeypatch.setattr(archive, name, forbidden)
    snapshot = archive.read_archive_snapshot(series)
    assert snapshot["models"][0]["source_identity"] == entry["library_source_identity"]


def test_absent_overlay_snapshot_does_not_create_files(model):
    series, root, pointer, entry = model
    path = archive.archive_path(series)
    snapshot = archive.read_archive_snapshot(series)
    assert snapshot == dict(format=archive.FORMAT, series_root=str(series), models=[])
    assert archive.filter_snapshot(snapshot, [entry], MethodSelection()) == [entry]
    assert not path.exists()


def test_cached_unreadable_choice_fails_closed_without_filesystem(model, monkeypatch):
    series, root, pointer, entry = model
    archive.archive_model(series, entry, "Reviewer", "Earlier candidate")
    snapshot = archive.read_archive_snapshot(series)
    with monkeypatch.context() as guarded:
        bomb_filesystem_filter(guarded)
        with pytest.raises(ValueError, match="cached metadata"):
            archive.filter_snapshot(snapshot, [entry], MethodSelection(error="Unreadable saved choice"))
