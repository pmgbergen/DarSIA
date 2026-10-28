"""Saved image choices survive another mount spelling or a cloned project.

Local metadata only: no image is opened and no job is created. New saves keep
paths inside the series root relative to it; older absolute saves still read.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tomllib

import pytest
import toml

from darsia.fluidflower import image_selection_proposal as proposal
from darsia.fluidflower import series_image_selection as model


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "user" / "series_registry.toml"
    path.parent.mkdir()
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(path))

    def write(aliases):
        lines = ["schema_version = 1", "[storage.path_aliases]"]
        lines += [
            f"{json.dumps(str(a))} = {json.dumps(str(b))}" for a, b in aliases.items()
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write({})
    return write


def _project(tmp_path, name="nas"):
    """A series project folder whose raw photographs live outside it."""
    root = tmp_path / name / "series"
    raw = tmp_path / name / "raw" / "AC17"
    (root / "config" / "runs").mkdir(parents=True)
    (root / "protocols").mkdir()
    raw.mkdir(parents=True)
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "unit"\n[runs]\nids = ["ac17"]\n',
        encoding="utf-8",
    )
    (root / "protocols" / "imaging.csv").write_text(
        "Path,Datetime\npre.png,2026-01-01 09:59:00\na.png,2026-01-01 10:00:00\n"
        "b.png,2026-01-01 11:00:00\nc.png,2026-01-01 12:00:00\n"
    )
    (root / "protocols" / "injection.csv").write_text(
        "ID,Location_X,Location_Y,Start,End,Rate_kg/s\n"
        "1,0.1,0.2,2026-01-01 10:00:00,2026-01-01 11:00:00,0.01\n"
    )
    (root / "config" / "runs" / "ac17.toml").write_text(
        f"[data]\nfolders = [{json.dumps(raw.as_posix())}]\n"
        '[protocols]\ninjection = "protocols/injection.csv"\n'
        f'[protocols.imaging]\n{json.dumps(raw.as_posix())} = "protocols/imaging.csv"\n',
        encoding="utf-8",
    )
    return root, raw


def _metadata(root):
    config = root / "config" / "runs" / "ac17.toml"
    return model.load_run_protocol_metadata(root, "ac17", config_paths=[config])


def _save(root, metadata, *, old_format=False, monkeypatch=None):
    paths = [metadata.frames[0]["source_path"], metadata.frames[-1]["source_path"]]
    expected = (root / "series.toml").read_bytes()
    if not old_format:
        return model.save_run_image_selection(
            root, metadata, paths, expected_bytes=expected
        )
    with monkeypatch.context() as patch:
        # The writer before series-relative saves stored every path as given.
        patch.setattr(model, "_stored", lambda value, _root: str(value))
        return model.save_run_image_selection(
            root, metadata, paths, expected_bytes=expected
        )


def _stored_run(root):
    document = tomllib.loads((root / "series.toml").read_bytes().decode("utf-8"))
    return document["image_selection"]["runs"]["ac17"]


def _move(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    shutil.rmtree(source)


def test_new_save_writes_relative_paths_inside_the_root_and_absolute_outside(
    tmp_path, registry
):
    root, raw = _project(tmp_path)
    metadata = _metadata(root)
    table, _raw = _save(root, metadata)
    stored = _stored_run(root)
    assert sorted(row["path"] for row in stored["source_metadata"]) == [
        "config/runs/ac17.toml",
        "protocols/imaging.csv",
        "protocols/injection.csv",
    ]
    manifest_path = stored["selection_manifest"]["path"]
    assert manifest_path.startswith("image_selection/manifests/ac17/")
    assert [row["source_path"] for row in stored["selected"]] == [
        str(raw / "pre.png"),
        str(raw / "c.png"),
    ]
    manifest = json.loads((root / manifest_path).read_bytes())
    assert manifest["source_metadata"] == stored["source_metadata"]
    assert [row["chosen_image"]["source_path"] for row in manifest["frames"]] == [
        str(raw / "pre.png"),
        str(raw / "c.png"),
    ]
    # Readers see paths as this machine reads them.
    located = model.read_series_image_selection(root)[0]["runs"]["ac17"]
    assert located == table["runs"]["ac17"]
    assert located["selection_manifest"]["path"] == str(root / manifest_path)
    assert {row["path"] for row in located["source_metadata"]} == {
        str(root / "config/runs/ac17.toml"),
        str(root / "protocols/imaging.csv"),
        str(root / "protocols/injection.csv"),
    }
    resolved = model.resolve_saved_selection(root, "ac17", metadata)
    assert resolved["selected_frame_count"] == 2
    assert all(Path(row["path"]).is_file() for row in resolved["source_metadata"])


def test_old_absolute_selection_still_validates_with_its_exact_text(
    tmp_path, registry, monkeypatch
):
    root, _raw = _project(tmp_path)
    metadata = _metadata(root)
    _save(root, metadata, old_format=True, monkeypatch=monkeypatch)
    stored = _stored_run(root)
    assert Path(stored["selection_manifest"]["path"]).is_absolute()
    assert all(Path(row["path"]).is_absolute() for row in stored["source_metadata"])
    assert model.read_series_image_selection(root)[0]["runs"]["ac17"] == stored
    resolved = model.resolve_saved_selection(root, "ac17", metadata)
    assert resolved["source_metadata"] == stored["source_metadata"]
    assert resolved["selected_frame_count"] == 2


def test_saved_selection_moves_to_declared_results_setup_with_frozen_manifest(tmp_path, registry):
    from darsia.fluidflower.storage_paths import refresh_storage_paths

    root, _raw = _project(tmp_path)
    _save(root, _metadata(root))
    saved = _stored_run(root)["selection_manifest"]
    original = root / saved["path"]
    content = original.read_bytes()
    old, target = root / "image_selection", root.parent / "Results/setup/image_selection"
    target.parent.mkdir(parents=True)
    old.rename(target)
    manifest = root / "series.toml"
    document = tomllib.loads(manifest.read_text())
    document["paths"] = dict(image_selection="../Results/setup/image_selection")
    document["storage"] = dict(path_aliases={str(old): str(target)})
    manifest.write_text(toml.dumps(document), encoding="utf-8")
    refresh_storage_paths(root)
    resolved = model.resolve_saved_selection(root, "ac17", _metadata(root))
    located = model.read_series_image_selection(root)[0]["runs"]["ac17"]["selection_manifest"]
    assert resolved["selected_frame_count"] == 2
    assert Path(located["path"]).parent == target / "manifests/ac17"
    assert Path(located["path"]).read_bytes() == content
    assert located["sha256"] == saved["sha256"]

    # Future ordinary saves use the same declaration, without recreating the old folder.
    _save(root, _metadata(root))
    assert not old.exists()
    assert len(list((target / "manifests/ac17").glob("*.json"))) == 2


def test_cloned_project_validates_a_new_save_without_any_alias(tmp_path, registry):
    root, _raw = _project(tmp_path)
    _save(root, _metadata(root))
    clone = tmp_path / "student" / "clone" / "series"
    _move(root, clone)
    metadata = _metadata(clone)
    resolved = model.resolve_saved_selection(clone, "ac17", metadata)
    assert resolved["selected_frame_count"] == 2
    assert all(
        Path(r["path"]).is_relative_to(clone) for r in resolved["source_metadata"]
    )


def test_old_absolute_selection_validates_through_an_aliased_root_and_raw_folder(
    tmp_path, registry, monkeypatch
):
    root, raw = _project(tmp_path)
    _save(root, _metadata(root), old_format=True, monkeypatch=monkeypatch)
    clone = tmp_path / "student" / "series"
    photos = tmp_path / "student" / "photos" / "AC17"
    _move(root, clone)
    _move(raw, photos)
    registry({root: clone, raw: photos})
    metadata = _metadata(clone)
    assert all(Path(row["source_path"]).parent == photos for row in metadata.frames)
    resolved = model.resolve_saved_selection(clone, "ac17", metadata, check_files=False)
    assert resolved["selected_frame_count"] == 2
    assert [row["source_path"] for row in resolved["selected"]] == [
        str(photos / "pre.png"),
        str(photos / "c.png"),
    ]
    assert all(
        Path(r["path"]).is_relative_to(clone) for r in resolved["source_metadata"]
    )


def test_applied_proposal_survives_moved_root_and_raw_folder_with_aliases(
    tmp_path, registry, monkeypatch
):
    root, raw = _project(tmp_path)
    metadata = _metadata(root)
    generated = proposal.default20_proposal(metadata)
    with monkeypatch.context() as patch:
        patch.setattr(model, "_stored", lambda value, _root: str(value))
        model.save_run_image_selection(
            root,
            metadata,
            [row["source_path"] for row in generated["selected"]],
            expected_bytes=(root / "series.toml").read_bytes(),
            proposal=generated,
            rule=proposal.RULE,
        )
    clone = tmp_path / "student" / "series"
    photos = tmp_path / "student" / "photos" / "AC17"
    _move(root, clone)
    _move(raw, photos)
    registry({root: clone, raw: photos})
    metadata = _metadata(clone)
    resolved = model.resolve_saved_selection(clone, "ac17", metadata)
    assert resolved["selected_frame_count"] == len(generated["selected"])
    assert all(
        Path(row["source_path"]).parent == photos for row in resolved["selected"]
    )


def test_without_its_alias_a_moved_old_selection_is_refused(
    tmp_path, registry, monkeypatch
):
    root, _raw = _project(tmp_path)
    _save(root, _metadata(root), old_format=True, monkeypatch=monkeypatch)
    clone = tmp_path / "student" / "series"
    _move(root, clone)
    with pytest.raises(model.ManifestChangedError, match="different protocol/config"):
        model.resolve_saved_selection(clone, "ac17", _metadata(clone))


def test_resave_on_a_clone_keeps_relative_paths_and_proposal_state(tmp_path, registry):
    root, _raw = _project(tmp_path)
    metadata = _metadata(root)
    generated = proposal.default20_proposal(metadata)
    chosen = [row["source_path"] for row in generated["selected"]]
    model.save_run_image_selection(
        root,
        metadata,
        chosen,
        expected_bytes=(root / "series.toml").read_bytes(),
        proposal=generated,
        rule=proposal.RULE,
    )
    clone = tmp_path / "student" / "series"
    _move(root, clone)
    metadata = _metadata(clone)
    table, _raw = model.save_run_image_selection(
        clone,
        metadata,
        [
            row["source_path"]
            for row in metadata.frames
            if row["source_path"] in {str(Path(p)) for p in chosen}
        ],
        expected_bytes=(clone / "series.toml").read_bytes(),
    )
    saved = table["runs"]["ac17"]
    assert saved["manual_edits_after_apply"] is False
    assert _stored_run(clone)["selection_manifest"]["path"].startswith(
        "image_selection/"
    )
    assert model.resolve_saved_selection(clone, "ac17", metadata)[
        "selected_frame_count"
    ]


def test_changed_manifest_content_is_still_detected_after_relative_saves(
    tmp_path, registry
):
    root, _raw = _project(tmp_path)
    metadata = _metadata(root)
    _save(root, metadata)
    path = root / "series.toml"
    document = tomllib.loads(path.read_bytes().decode())
    saved = document["image_selection"]["runs"]["ac17"]
    manifest_path = root / saved["selection_manifest"]["path"]
    manifest = json.loads(manifest_path.read_bytes())
    manifest["frames"][0]["chosen_image"]["source_path"] = "another.png"
    raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(raw)
    saved["selection_manifest"].update(
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )
    path.write_text(model.toml.dumps(document), encoding="utf-8")
    with pytest.raises(model.ManifestChangedError, match="different choices"):
        model.resolve_saved_selection(root, "ac17", metadata)


def test_located_stored_and_comparable_paths(tmp_path, registry):
    root = tmp_path / "series"
    old = (tmp_path / "old").as_posix()
    registry({old: tmp_path / "new"})
    remap = model.storage_path_remapper()
    assert model._located("config/a.toml", root, remap) == str(root / "config/a.toml")
    assert model._located(old + "/a.toml", root, remap) == str(tmp_path / "new/a.toml")
    outside = str(tmp_path / "raw/a.png")
    assert model._located(outside, root, remap) is outside
    assert model._stored(str(root / "config/a.toml"), root) == "config/a.toml"
    assert model._stored(outside, root) == outside
    assert model._stored("config/a.toml", root) == "config/a.toml"
    value = {
        "path": "config/a.toml",
        "sha256": "x",
        "rows": [{"source_path": old + "/a"}],
    }
    assert model._map_paths(value, str.upper) == {
        "path": "CONFIG/A.TOML",
        "sha256": "x",
        "rows": [{"source_path": (old + "/a").upper()}],
    }
    assert model._comparable(value, root, remap) == model._comparable(
        {
            "path": str(root / "config/a.toml"),
            "sha256": "x",
            "rows": [{"source_path": str(tmp_path / "new/a")}],
        },
        root,
        remap,
    )


def _quality_receipt(path, raw_folder, metadata):
    """An imported quality receipt that names images by their old absolute path."""
    rows = [
        {
            "source_path": str(raw_folder / Path(row["source_path"]).name),
            "elapsed_h": row["elapsed_h"],
            "source_sha256": "0" * 64,
            "quality": {"class": "ok", "class_measurement": proposal.QUALITY_RULE},
        }
        for row in metadata.frames
    ]
    payload = {"format": "fluidflower-quality-camera-grid-v1", "selected": rows}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_imported_receipt_paths_follow_the_storage_aliases(tmp_path, registry):
    root, raw = _project(tmp_path)
    receipt = _quality_receipt(tmp_path / "quality.json", raw, _metadata(root))
    saved_receipt = receipt.read_bytes()
    photos = tmp_path / "student" / "photos" / "AC17"
    _move(raw, photos)
    registry({raw: photos})
    metadata = _metadata(root)
    assert all(Path(row["source_path"]).parent == photos for row in metadata.frames)
    quality = proposal.load_saved_quality(receipt, metadata)
    # The receipt still names the old folder; its classes key the images here.
    assert set(quality["classes"]) == {
        proposal._key(row["source_path"]) for row in metadata.frames
    }
    assert receipt.read_bytes() == saved_receipt
    assert all(
        Path(row["source_path"]).parent == raw
        for row in json.loads(saved_receipt)["selected"]
    )
