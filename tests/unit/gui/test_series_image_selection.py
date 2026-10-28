"""Series image choices and Q6 Default20: local metadata/offscreen Qt only."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tomllib

import pytest
from PySide6.QtCore import Qt
from darsia.fluidflower import series_image_selection as model
from darsia.fluidflower import image_selection_widget as ui
from darsia.fluidflower import image_selection_proposal as proposal_model

proposal = proposal_model


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        """schema_version = 1
series_id = "unit"
name = "Local only"
[runs]
ids = ["ac17", "ac18"]
[method]
active = "phase_model"
[method.phase_model]
model_version = "existing-model"
mass_calibration = "frozen.json"
[unrelated]
datetime = 2026-01-02T03:04:05Z
integer = 9007199254740993
float = 0.125
array = [1, 2, 3]
""",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def metadata(root):
    times = [-0.25] + [i / 16 for i in range(17)] + [1 + i / 4 for i in range(1, 401)]
    frames = tuple(
        dict(
            source_path=str(root / f"image_{i:04d}.png"),
            image_id=f"image_{i:04d}.png",
            frame_index=i,
            elapsed_h=t,
        )
        for i, t in enumerate(times)
    )
    return model.RunImageMetadata("ac17", frames, 0, 1, (), (), str(root))


def test_no_rule_assumed_before_owner_clarification(metadata):
    result = model.suggest_target_hours(metadata, interpretation="")
    assert (
        len(result) == 20
        and result[0] == -0.25
        and result[8] == 1
        and result[-1] == 101
    )


def test_legacy_count_interpretation_cannot_bypass_actual_studio_contradiction(
    metadata,
):
    with pytest.raises(ValueError, match="historical count"):
        model.suggest_target_hours(metadata, interpretation=model.DRAFT_EARLY_BLOCK)


@pytest.mark.parametrize(
    "field,value", [("early_count", True), ("early_count", 1), ("rest_count", 1.5)]
)
def test_bad_grid_counts(metadata, field, value):
    with pytest.raises(ValueError):
        model.suggest_target_hours(
            metadata, interpretation=model.DRAFT_EARLY_BLOCK, **{field: value}
        )


@pytest.mark.parametrize(
    "frames,message",
    [
        ((0.0, 1.0, 2.0, 3.0), "before injection"),
        ((-0.1, 0.0, 0.5, 1.0), "post-injection"),
    ],
)
def test_grid_cannot_invent_missing_coverage(metadata, frames, message):
    changed = replace(
        metadata,
        frames=tuple(
            dict(source_path=f"{i}.png", elapsed_h=t) for i, t in enumerate(frames)
        ),
    )
    output = proposal_model.default20_proposal(changed)
    assert output["selected_frame_count"] < 20
    if frames[0] >= 0:
        assert output["no_preinjection_image"]
    else:
        assert any("eleven late" in issue for issue in output["issues"])


def test_existing_resolver_tie_blacklist_duplicate_and_late_semantics(root):
    frames = (
        dict(source_path="pre.png", elapsed_h=-0.1),
        dict(source_path="early.png", elapsed_h=0.8),
        dict(source_path="blacklisted.png", elapsed_h=1, blacklisted=True),
        dict(source_path="late.png", elapsed_h=1.2),
        dict(source_path="late.png", elapsed_h=1.2),
    )
    metadata = model.RunImageMetadata("ac17", frames, 0, 1, series_root=str(root))
    receipt = model.preview_targets(metadata, [-0.1, 1.0], tolerance_seconds=800)
    original = model.resolve_selection(
        {"run": "ac17", "frames": list(frames)},
        {
            "schema_version": 1,
            "purposes": {
                "review": {
                    "mode": "target_hours",
                    "target_hours": [-0.1, 1.0],
                    "tolerance_seconds": 800,
                    "on_unmatched_target": "error",
                }
            },
        },
        "review",
    )
    assert receipt["runs"] == original["runs"]
    assert receipt["runs"][0]["selected"][1]["source_path"] == "early.png"
    failed = model.preview_targets(metadata, [-0.1, 50], tolerance_seconds=800)
    assert failed["status"] == "failed"
    assert any(
        row["code"] == "late_selected_coverage_unmet"
        for row in failed["runs"][0]["issues"]
    )


def test_duplicate_target_matches_are_exposed_not_silently_filled(metadata):
    receipt = model.preview_targets(
        metadata, [-0.25, 0.001, 0.002, 101], tolerance_seconds=300
    )
    assert receipt["requested_frame_count"] == 4
    assert receipt["distinct_selected_frame_count"] == 3
    assert any("distinct" in note for note in receipt["selection_notes"])


def test_manual_selection_is_eligible_unique_chronological(metadata):
    paths = [metadata.frames[7]["source_path"], metadata.frames[0]["source_path"]]
    rows = model.selected_frame_rows(metadata, paths)
    assert [r["source_path"] for r in rows] == paths[::-1]
    with pytest.raises(ValueError, match="twice"):
        model.selected_frame_rows(metadata, paths + [paths[0]])
    with pytest.raises(ValueError, match="eligible"):
        model.selected_frame_rows(metadata, ["unregistered.png"])
    blacklisted = replace(
        metadata,
        frames=({**metadata.frames[0], "blacklisted": True}, *metadata.frames[1:]),
    )
    with pytest.raises(ValueError, match="eligible"):
        model.selected_frame_rows(blacklisted, [paths[1]])


def test_save_only_new_section_and_exact_other_values(root, metadata):
    path = root / "series.toml"
    old = path.read_bytes()
    before = tomllib.loads(old.decode())
    choices = [metadata.frames[0]["source_path"], metadata.frames[-1]["source_path"]]
    saved, raw = model.save_run_image_selection(
        root, metadata, choices, expected_bytes=old
    )
    after = tomllib.loads(raw.decode())
    del after["image_selection"]
    del after["data_time"]
    assert model._same_values(before, after)
    assert (
        path.read_bytes() == raw
        and saved["runs"]["ac17"]["selected"][0]["elapsed_h"] < 0
    )
    assert model.read_series_image_selection(root)[0] == saved


def test_preserve_other_run_and_unknown_selection_fields(root, metadata):
    path = root / "series.toml"
    old = path.read_bytes()
    first, raw = model.save_run_image_selection(
        root, metadata, [metadata.frames[0]["source_path"]], expected_bytes=old
    )
    m2 = replace(metadata, run_id="ac18")
    second, raw2 = model.save_run_image_selection(
        root,
        m2,
        [metadata.frames[0]["source_path"], metadata.frames[2]["source_path"]],
        expected_bytes=raw,
    )
    assert second["runs"]["ac17"] == first["runs"]["ac17"]
    doc = tomllib.loads(raw2.decode())
    doc["image_selection"]["custom_owner_field"] = "retained"
    doc["image_selection"]["runs"]["ac18"]["custom_run_field"] = ["retained"]
    raw3 = model.toml.dumps(doc).encode()
    path.write_bytes(raw3)
    third, _ = model.save_run_image_selection(
        root, m2, [metadata.frames[0]["source_path"]], expected_bytes=raw3
    )
    assert third["custom_owner_field"] == "retained" and third["runs"]["ac18"][
        "custom_run_field"
    ] == ["retained"]


def test_manifest_cas_and_foreign_series_do_not_write(root, metadata, tmp_path):
    path = root / "series.toml"
    old = path.read_bytes()
    changed = old + b"\n# external edit\n"
    path.write_bytes(changed)
    with pytest.raises(model.ManifestChangedError):
        model.save_run_image_selection(
            root, metadata, [metadata.frames[0]["source_path"]], expected_bytes=old
        )
    assert path.read_bytes() == changed
    foreign = tmp_path / "other"
    foreign.mkdir()
    (foreign / "series.toml").write_bytes(changed)
    with pytest.raises(ValueError, match="another series"):
        model.save_run_image_selection(
            foreign,
            metadata,
            [metadata.frames[0]["source_path"]],
            expected_bytes=changed,
        )
    assert (foreign / "series.toml").read_bytes() == changed


def test_changed_protocol_blocks_save_without_write(root, metadata):
    source = root / "protocol.csv"
    source.write_text("old")
    metadata = replace(metadata, sources=(model._descriptor(source),))
    old = (root / "series.toml").read_bytes()
    source.write_text("changed")
    with pytest.raises(model.ManifestChangedError, match="Source protocol"):
        model.save_run_image_selection(
            root, metadata, [metadata.frames[0]["source_path"]], expected_bytes=old
        )
    assert (root / "series.toml").read_bytes() == old


@pytest.mark.parametrize("which", ["empty", "no_pre", "negative_tolerance"])
def test_invalid_manual_save_keeps_bytes(root, metadata, which):
    old = (root / "series.toml").read_bytes()
    paths = (
        []
        if which == "empty"
        else [metadata.frames[2 if which == "no_pre" else 0]["source_path"]]
    )
    with pytest.raises(ValueError):
        model.save_run_image_selection(
            root,
            metadata,
            paths,
            expected_bytes=old,
            tolerance_seconds=-1 if which == "negative_tolerance" else 300,
        )
    assert (root / "series.toml").read_bytes() == old


def test_write_race_keeps_original_and_does_not_publish(root, metadata, monkeypatch):
    path = root / "series.toml"
    old = path.read_bytes()
    original = Path.read_bytes
    calls = 0

    def racing(p):
        nonlocal calls
        if p == path:
            calls += 1
            if calls == 2:
                path.write_bytes(old + b"\n# concurrent\n")
        return original(p)

    monkeypatch.setattr(Path, "read_bytes", racing)
    with pytest.raises(model.ManifestChangedError, match="changed"):
        model.save_run_image_selection(
            root, metadata, [metadata.frames[0]["source_path"]], expected_bytes=old
        )
    assert original(path) == old + b"\n# concurrent\n"
    assert not list(root.glob(".*.tmp"))


def _protocol_fixture(root, *, legacy=False):
    raw = root / "raw"
    raw.mkdir()
    imaging = root / "imaging.csv"
    imaging.write_text(
        "Path,Datetime\npre.png,2026-01-01 09:59:00\na.png,2026-01-01 10:00:00\nb.png,2026-01-01 11:00:00\nc.png,2026-01-01 12:00:00\n"
    )
    injection = root / "injection.csv"
    injection.write_text(
        "ID,Location_X,Location_Y,Start,End,Rate_kg/s\n1,0.1,0.2,2026-01-01 10:00:00,2026-01-01 11:00:00,0.01\n"
    )
    blacklist = root / "blacklist.csv"
    blacklist.write_text("Path\nb.png\n")
    cfg = root / "run.toml"
    cfg.write_text(
        "[data]\n"
        + ('folder = "raw"' if legacy else 'folders = ["raw"]')
        + '\n[protocols]\ninjection = "injection.csv"\n[protocols.imaging]\nraw = "imaging.csv"\n[protocols.blacklist]\nraw = "blacklist.csv"\n'
    )
    return cfg


@pytest.mark.parametrize("legacy", [False, True])
def test_actual_original_protocol_readers_no_raw_scan(root, legacy, monkeypatch):
    cfg = _protocol_fixture(root, legacy=legacy)

    def scan_forbidden(*_a, **_kw):
        raise AssertionError("No raw-folder scan allowed")

    monkeypatch.setattr(Path, "glob", scan_forbidden)
    monkeypatch.setattr(Path, "rglob", scan_forbidden)
    m = model.load_run_protocol_metadata(root, "ac17", config_paths=[cfg])
    assert m.injection_start_h == 0 and m.injection_end_h == 1
    assert [r["elapsed_h"] for r in m.frames] == pytest.approx([-1 / 60, 0, 1, 2])
    assert m.frames[2]["blacklisted"] is True
    assert all(Path(r["source_path"]).parent == root / "raw" for r in m.frames)
    assert len(m.sources) == 5  # manifest, config, injection, imaging, blacklist


def test_bad_saved_choice_schema_is_not_silently_replaced(root):
    path = root / "series.toml"
    path.write_bytes(
        path.read_bytes() + b'\n[image_selection]\nformat="unknown"\nschema_version=1\n'
    )
    with pytest.raises(ValueError, match="unsupported"):
        model.read_series_image_selection(root)


def test_full_unc_imaging_keys_resolve_to_declared_folder_without_scan(root):
    cfg = _protocol_fixture(root)
    doc = tomllib.loads(cfg.read_text())
    folder = r"\\unit-nas\share\raw"
    doc["data"]["folders"] = [folder]
    doc["protocols"]["imaging"] = {folder: "imaging.csv"}
    doc["protocols"]["blacklist"] = {folder: "blacklist.csv"}
    cfg.write_text(model.toml.dumps(doc), encoding="utf-8")
    (root / "imaging.csv").write_text(
        "Path,Datetime\n"
        + folder
        + r"\pre.png,2026-01-01 09:59:00"
        + "\n"
        + folder
        + r"\rest.png,2026-01-01 12:00:00"
        + "\n"
    )
    m = model.load_run_protocol_metadata(root, "ac17", config_paths=[cfg])
    assert Path(m.frames[0]["source_path"]) == Path(folder) / "pre.png"
    assert Path(m.frames[1]["source_path"]) == Path(folder) / "rest.png"


def test_future_job_bridge_returns_bound_explicit_choices_without_images(root):
    cfg = _protocol_fixture(root)
    metadata = model.load_run_protocol_metadata(root, "ac17", config_paths=[cfg])
    old = (root / "series.toml").read_bytes()
    model.save_run_image_selection(
        root,
        metadata,
        [metadata.frames[0]["source_path"], metadata.frames[-1]["source_path"]],
        expected_bytes=old,
    )
    result = model.resolve_saved_selection(root, "ac17", metadata)
    assert result["selected_frame_count"] == 2 and result["check_files"] is False
    assert result["selected"][0]["elapsed_h"] < 0
    assert not Path(result["selected"][0]["source_path"]).exists()
    with pytest.raises(FileNotFoundError):
        model.resolve_saved_selection(root, "ac17", metadata, check_files=True)
    with pytest.raises(ValueError, match="another run"):
        model.resolve_saved_selection(root, "ac18", metadata)
    (root / "imaging.csv").write_text(
        (root / "imaging.csv").read_text() + "extra.png,2026-01-01 13:00:00\n"
    )
    with pytest.raises(model.ManifestChangedError, match="changed after"):
        model.resolve_saved_selection(root, "ac17", metadata)


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication, QSettings

    old = (
        QCoreApplication.organizationName(),
        QCoreApplication.applicationName(),
        QSettings.defaultFormat(),
    )
    application = QApplication.instance() or QApplication([])
    try:
        yield application
    finally:
        QCoreApplication.setOrganizationName(old[0])
        QCoreApplication.setApplicationName(old[1])
        QSettings.setDefaultFormat(old[2])


def test_qt_open_is_passive_manual_add_remove_preview_is_explicit(root, metadata, app):
    calls = []
    pixels = []

    def reader(_root, run):
        calls.append(run)
        return metadata

    def forbidden_image(path):
        pixels.append(path)
        raise AssertionError("Preview not requested")

    old = (root / "series.toml").read_bytes()
    dialog = ui.SeriesImageSelectionDialog(
        root, metadata_reader=reader, image_reader=forbidden_image
    )
    assert calls == pixels == [] and (root / "series.toml").read_bytes() == old
    assert not dialog.suggest_button.isEnabled() and not dialog.save_button.isEnabled()
    dialog.load_run()
    assert calls == ["ac17"] and pixels == []
    from PySide6.QtCore import Qt

    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
    dialog.table.item(4, 0).setCheckState(Qt.CheckState.Checked)
    dialog.table.item(4, 0).setCheckState(Qt.CheckState.Unchecked)
    assert dialog.selected_paths() == [metadata.frames[0]["source_path"]]
    assert (root / "series.toml").read_bytes() == old
    dialog.save_run()
    assert (
        model.read_series_image_selection(root)[0]["runs"]["ac17"]["selected"][0][
            "source_path"
        ]
        == metadata.frames[0]["source_path"]
    )
    assert pixels == []
    dialog.close()


def test_qt_switch_retains_drafts_without_loading_second_run(root, metadata, app):
    calls = []
    dialog = ui.SeriesImageSelectionDialog(
        root,
        metadata_reader=lambda _root, run: (
            calls.append(run) or replace(metadata, run_id=run)
        ),
    )
    dialog.load_run()
    from PySide6.QtCore import Qt

    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
    dialog.run_combo.setCurrentIndex(1)
    assert calls == ["ac17"] and not dialog.save_button.isEnabled()
    dialog.load_run()
    dialog.run_combo.setCurrentIndex(0)
    dialog.load_run()
    assert calls == ["ac17", "ac18", "ac17"] and dialog.selected_paths() == [
        metadata.frames[0]["source_path"]
    ]
    dialog._dirty_runs.clear()
    dialog.close()


def test_qt_failed_save_retains_checkboxes_and_bytes(root, metadata, app):
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: metadata)
    dialog.load_run()
    from PySide6.QtCore import Qt

    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
    path = root / "series.toml"
    changed = path.read_bytes() + b"\n# external\n"
    path.write_bytes(changed)
    with pytest.raises(model.ManifestChangedError):
        dialog.save_run()
    assert (
        dialog.selected_paths() == [metadata.frames[0]["source_path"]]
        and path.read_bytes() == changed
    )
    assert "ac17" in dialog._dirty_runs
    dialog._dirty_runs.clear()
    dialog.close()


def test_qt_preview_reads_only_highlighted_source_when_explicit(root, metadata, app):
    from PySide6.QtGui import QImage

    local = root / "preview.png"
    image = QImage(12, 8, QImage.Format.Format_RGB32)
    image.fill(0xFF77AA)
    assert image.save(str(local))
    m = replace(
        metadata,
        frames=(
            {
                "source_path": str(local),
                "elapsed_h": -0.1,
                "frame_index": 0,
                "image_id": "preview.png",
            },
        ),
    )
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: m)
    dialog.load_run()
    assert dialog.preview_label.pixmap().isNull()
    dialog.table.selectRow(0)
    assert dialog.preview_image() == str(local)
    assert not dialog.preview_label.pixmap().isNull()
    assert "Original source image" in dialog.preview_note.text()
    dialog.close()


def test_qt_stale_saved_choices_remain_visible_until_manually_removed(
    root, metadata, app
):
    old = (root / "series.toml").read_bytes()
    model.save_run_image_selection(
        root,
        metadata,
        [metadata.frames[0]["source_path"], metadata.frames[2]["source_path"]],
        expected_bytes=old,
    )
    changed = replace(metadata, frames=(metadata.frames[0], *metadata.frames[3:]))
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: changed)
    dialog.load_run()
    stale_index = next(i for i, r in enumerate(dialog._rows) if r.get("ineligible"))
    from PySide6.QtCore import Qt

    assert dialog.table.item(stale_index, 0).checkState() == Qt.CheckState.Checked
    assert "Unavailable" in dialog.table.item(stale_index, 1).text()
    with pytest.raises(ValueError, match="eligible"):
        dialog.save_run()
    dialog.table.item(stale_index, 0).setCheckState(Qt.CheckState.Unchecked)
    dialog.save_run()
    assert (
        len(model.read_series_image_selection(root)[0]["runs"]["ac17"]["selected"]) == 1
    )
    dialog.close()


def test_qt_explicit_reload_reads_protocol_again_and_keeps_draft(root, metadata, app):
    calls = []

    def reader(*_):
        calls.append(1)
        return (
            metadata
            if len(calls) == 1
            else replace(
                metadata,
                frames=tuple(
                    {**r, "elapsed_h": r["elapsed_h"] + 0.01} for r in metadata.frames
                ),
            )
        )

    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=reader)
    dialog.load_run()
    from PySide6.QtCore import Qt

    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
    dialog.load_run()
    assert len(calls) == 2 and dialog.selected_paths() == [
        metadata.frames[0]["source_path"]
    ]
    assert (
        dialog.metadata.frames[0]["elapsed_h"] == metadata.frames[0]["elapsed_h"] + 0.01
    )
    dialog._dirty_runs.clear()
    dialog.close()


def _saved_bridge_fixture(root):
    source = root / "timing.csv"
    source.write_text("saved timing source", encoding="utf-8")
    meta = model.RunImageMetadata(
        "ac17",
        (
            {
                "source_path": str(root / "pre.png"),
                "frame_index": 0,
                "image_id": "pre.png",
                "elapsed_h": -0.1,
            },
            {
                "source_path": str(root / "post.png"),
                "frame_index": 1,
                "image_id": "post.png",
                "elapsed_h": 2.0,
            },
        ),
        0,
        1,
        (model._descriptor(source),),
        (),
        str(root),
    )
    paths = [row["source_path"] for row in meta.frames]
    model.save_run_image_selection(
        root, meta, paths, expected_bytes=(root / "series.toml").read_bytes()
    )
    return meta


def test_bridge_rejects_manifest_race_and_binds_exact_original_bytes(root, monkeypatch):
    meta = _saved_bridge_fixture(root)
    raw = (root / "series.toml").read_bytes()
    output = model.resolve_saved_selection(root, "ac17", meta)
    import hashlib

    assert output["manifest"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert output["manifest"]["size_bytes"] == len(raw)
    original = model.selected_frame_rows

    def race(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "series.toml").write_bytes(
            raw + b"\n# external edit during resolution\n"
        )
        return result

    monkeypatch.setattr(model, "selected_frame_rows", race)
    with pytest.raises(model.ManifestChangedError, match="while resolving"):
        model.resolve_saved_selection(root, "ac17", meta)


@pytest.mark.parametrize(
    "field,value", [("frame_index", 44), ("image_id", "foreign.png")]
)
def test_bridge_rejects_saved_protocol_identity_changes(root, field, value):
    meta = _saved_bridge_fixture(root)
    import toml

    path = root / "series.toml"
    document = tomllib.loads(path.read_text())
    document["image_selection"]["runs"]["ac17"]["selected"][0][field] = value
    path.write_text(toml.dumps(document), encoding="utf-8")
    with pytest.raises(model.ManifestChangedError, match="protocol identity"):
        model.resolve_saved_selection(root, "ac17", meta)


def test_bridge_rejects_nonpreinjection_saved_selection(root):
    meta = _saved_bridge_fixture(root)
    import toml

    path = root / "series.toml"
    document = tomllib.loads(path.read_text())
    document["image_selection"]["runs"]["ac17"]["selected"].pop(0)
    path.write_text(toml.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="precede injection"):
        model.resolve_saved_selection(root, "ac17", meta)


def test_explicit_save_adds_original_data_time_format_and_keeps_old_entries(
    root, metadata
):
    import toml
    from darsia.presets.workflows.config.time_data import ImageTimeData

    path = root / "series.toml"
    doc = tomllib.loads(path.read_bytes().decode())
    old_entry = {"name": "existing untouched", "times": [1.25], "tol": "00:01:00"}
    doc["data_time"] = [old_entry]
    path.write_bytes(toml.dumps(doc).encode())
    saved, raw = model.save_run_image_selection(
        root,
        metadata,
        [metadata.frames[0]["source_path"], metadata.frames[-1]["source_path"]],
        expected_bytes=path.read_bytes(),
    )
    updated = tomllib.loads(raw.decode())
    assert updated["data_time"][0] == old_entry
    names = saved["runs"]["ac17"]["data_time_names"]
    chosen = [e for e in updated["data_time"] if e["name"] in names]
    assert len(chosen) == 2
    parsed = [ImageTimeData().load({"time": {e["name"]: e}}) for e in chosen]
    assert [t for item in parsed for t in item.times] == [
        metadata.frames[0]["elapsed_h"],
        metadata.frames[-1]["elapsed_h"],
    ]
    assert chosen[0]["tol"] == "00:00:15" and chosen[1]["tol"] == "00:02:30"
    assert updated["method"] == doc["method"]


def test_fresh_per_run_manifests_are_immutable_and_honest_about_missing_quality(
    root, metadata
):
    import json

    path = root / "series.toml"
    choices = [metadata.frames[0]["source_path"]]
    first, raw = model.save_run_image_selection(
        root, metadata, choices, expected_bytes=path.read_bytes()
    )
    bound = first["runs"]["ac17"]["selection_manifest"]
    receipt_path = Path(bound["path"])
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    assert receipt["frames"][0]["target_time_h"] == metadata.frames[0]["elapsed_h"]
    assert receipt["frames"][0]["offset_seconds"] == 0
    assert receipt["frames"][0]["camera_time"] == "Not recorded"
    assert receipt["frames"][0]["frame_list_class"] == "Not recorded"
    assert receipt["scientific_computation"] is False
    second, _ = model.save_run_image_selection(
        root, metadata, choices, expected_bytes=raw
    )
    assert second["runs"]["ac17"]["selection_manifest"]["path"] != bound["path"]
    assert receipt_path.read_bytes() == original


def test_resolver_rejects_tampered_per_run_manifest(root):
    meta = _saved_bridge_fixture(root)
    table, _ = model.read_series_image_selection(root)
    p = Path(table["runs"]["ac17"]["selection_manifest"]["path"])
    p.write_bytes(p.read_bytes() + b"\n ")
    with pytest.raises(model.ManifestChangedError, match="selection manifest changed"):
        model.resolve_saved_selection(root, "ac17", meta)


def test_resolver_rejects_tampered_named_time_section(root):
    import toml

    meta = _saved_bridge_fixture(root)
    path = root / "series.toml"
    doc = tomllib.loads(path.read_bytes().decode())
    name = doc["image_selection"]["runs"]["ac17"]["data_time_names"][0]
    next(e for e in doc["data_time"] if e["name"] == name)["times"][0] = -0.2
    path.write_bytes(toml.dumps(doc).encode())
    with pytest.raises(model.ManifestChangedError, match="times differ"):
        model.resolve_saved_selection(root, "ac17", meta)


def test_late_series_cas_failure_preserves_immutable_manifest_and_external_edit(
    root, metadata, monkeypatch
):
    path = root / "series.toml"
    old = path.read_bytes()
    original_read = Path.read_bytes
    calls = 0

    def racing(p):
        nonlocal calls
        if p == path:
            calls += 1
            if calls == 3:
                path.write_bytes(old + b"\n# preserved external edit\n")
        return original_read(p)

    monkeypatch.setattr(Path, "read_bytes", racing)
    with pytest.raises(model.ManifestChangedError):
        model.save_run_image_selection(
            root, metadata, [metadata.frames[0]["source_path"]], expected_bytes=old
        )
    assert original_read(path) == old + b"\n# preserved external edit\n"
    assert (
        len(list((root / "image_selection" / "manifests" / "ac17").glob("*.json"))) == 1
    )
    assert not list(root.glob(".*.tmp"))


def with_times(metadata, times):
    return replace(
        metadata,
        frames=tuple(
            dict(
                source_path=str(Path(metadata.series_root) / f"frame{i}.jpg"),
                image_id=f"frame{i}.jpg",
                frame_index=i,
                elapsed_h=t,
            )
            for i, t in enumerate(times)
        ),
    )


def quality_file(root, metadata, classes):
    rows = []
    for index, status in classes.items():
        row = metadata.frames[index]
        rows.append(
            dict(
                source_path=row["source_path"],
                elapsed_h=row["elapsed_h"],
                source_sha256=f"{index+1:064x}",
                quality=dict(
                    class_measurement=proposal.QUALITY_RULE, **{"class": status}
                ),
            )
        )
    path = root / "saved-quality.json"
    path.write_text(
        json.dumps(dict(format="fluidflower-quality-camera-grid-v1", selected=rows)),
        encoding="utf-8",
    )
    return path, proposal.load_saved_quality(path, metadata)


def native_file(root, metadata, indices):
    path = root / "stored-native-manifest.json"
    doc = dict(
        dataset="unit",
        run=metadata.run_id,
        frame_sources=[
            dict(
                source_path=metadata.frames[i]["source_path"],
                source_sha256=f"{i+1:064x}",
                frame_index=n,
                elapsed_h=metadata.frames[i]["elapsed_h"],
            )
            for n, i in enumerate(indices)
        ],
    )
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def bound_metadata(root, metadata):
    path = root / "protocol-source.csv"
    path.write_text("local timing metadata", encoding="utf-8")
    return replace(metadata, sources=(model._descriptor(path),))


def test_exact_q6_targets_delays_last_and_unknown_quality(metadata):
    result = proposal.default20_proposal(metadata)
    targets = [row["target_time_h"] for row in result["target_matches"]]
    assert len(targets) == 20 and targets[0] == -0.25
    assert targets[1:9] == [k / 8 for k in range(1, 9)]
    assert targets[9:] == pytest.approx(
        [1 + 0.25 * (100 / 0.25) ** (k / 10) for k in range(11)]
    )
    assert targets[-1] == 101 and result["parameters"]["t_last_h"] == 101
    assert result["parameters"]["t_last_class"] == "Not recorded"
    assert all(
        row.get("frame_list_class") == "Not recorded"
        for row in result["target_matches"]
        if row["selected"]
    )
    assert result["rule"] == proposal.RULE and not result["scientific_computation"]


def test_camera_tie_earlier_outside_tolerance_still_selected_and_distinct_next(
    metadata,
):
    changed = with_times(metadata, [-0.25, 0, 0.25, 0.5, 1, 2])
    result = proposal.default20_proposal(changed)
    first = result["target_matches"][1]
    assert first["chosen_image"]["source_path"] == changed.frames[1]["source_path"]
    assert "no image within tolerance" in first["flags"] and first["selected"]
    collision = result["target_matches"][3]
    assert collision["chosen_image"]["source_path"] == changed.frames[3]["source_path"]
    assert "duplicate target: next distinct image" in collision["flags"]
    assert (
        result["selected_frame_count"]
        == len({row["source_path"] for row in result["selected"]})
        < 20
    )
    assert any("fewer than 20" in issue for issue in result["issues"])


def test_saved_reject_last_and_pre_are_not_used_for_endpoints(root, metadata):
    path, quality = quality_file(
        root, metadata, {0: "reject", len(metadata.frames) - 1: "reject"}
    )
    result = proposal.default20_proposal(metadata, quality=quality)
    assert (
        result["no_preinjection_image"]
        and result["target_matches"][0]["chosen_image"]["source_path"]
        == metadata.frames[1]["source_path"]
    )
    assert result["parameters"]["t_last_h"] == metadata.frames[-2]["elapsed_h"]
    assert result["parameters"]["t_last_class"] == "Not recorded"


def test_reject_replaced_only_saved_ok_with_bound_and_offset(root, metadata):
    changed = with_times(metadata, [-0.1, 0.125, 0.125 + 40 / 3600, 0.25, 0.5, 1, 2])
    _, quality = quality_file(root, changed, {1: "reject", 2: "ok"})
    result = proposal.default20_proposal(changed, quality=quality)
    match = result["target_matches"][1]
    assert match["selected"] and match["frame_list_class"] == "ok"
    assert match["offset_seconds"] == pytest.approx(40)
    assert "reject replaced by nearest saved ok image" in match["flags"]
    assert "no image within tolerance" in match["flags"]
    assert match["rejected_image"]["source_path"] == changed.frames[1]["source_path"]
    _, quality = quality_file(root, changed, {1: "reject", 2: "minor"})
    missing = proposal.default20_proposal(changed, quality=quality)["target_matches"][1]
    assert not missing["selected"] and any(
        "without a saved ok" in flag for flag in missing["flags"]
    )


def test_saved_quality_source_change_or_fabricated_class_is_refused(root, metadata):
    path, quality = quality_file(root, metadata, {1: "ok"})
    changed = deepcopy(quality)
    changed["classes"][proposal._key(metadata.frames[1]["source_path"])][
        "class"
    ] = "reject"
    with pytest.raises(ValueError, match="exact source receipt"):
        proposal.default20_proposal(metadata, quality=changed)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(model.ManifestChangedError):
        proposal.default20_proposal(metadata, quality=quality)


@pytest.mark.parametrize("d1", [0, -1, float("nan"), True])
def test_invalid_d1_no_mutation(metadata, d1):
    with pytest.raises(ValueError):
        proposal.default20_proposal(metadata, d1_h=d1)


def test_no_postinjection_or_descending_delay_is_finite_and_explicit(metadata):
    short = with_times(metadata, [-0.1, 0, 0.5, 1])
    result = proposal.default20_proposal(short)
    assert len(result["target_matches"]) == 9 and any(
        "eleven late" in issue for issue in result["issues"]
    )
    result = proposal.default20_proposal(
        with_times(metadata, [-0.1, 0, 0.5, 1, 1.1]), d1_h=0.25
    )
    assert len(result["target_matches"]) == 20 and any(
        "descend" in issue for issue in result["issues"]
    )


def test_no_pre_first_usable_save_resolve_flag_and_manual_post_only_still_refused(
    root, metadata
):
    changed = bound_metadata(root, with_times(metadata, [0, 0.25, 0.5, 1, 2]))
    original = (root / "series.toml").read_bytes()
    saved, raw = model.save_run_image_selection(
        root,
        changed,
        [changed.frames[0]["source_path"], changed.frames[-1]["source_path"]],
        expected_bytes=original,
    )
    assert saved["runs"]["ac17"]["no_preinjection_image"] is True
    assert (
        model.resolve_saved_selection(root, "ac17", changed)["selected_frame_count"]
        == 2
    )
    before = (root / "series.toml").read_bytes()
    with pytest.raises(ValueError, match="precede injection"):
        model.save_run_image_selection(
            root, metadata, [metadata.frames[1]["source_path"]], expected_bytes=before
        )
    assert (root / "series.toml").read_bytes() == before


def test_explicit_native_import_exact_source_sha_and_own_clock_no_pixels_or_writes(
    root, metadata
):
    path = native_file(root, metadata, [0, 3, 10])
    before = {str(p): p.read_bytes() for p in root.iterdir() if p.is_file()}
    imported = proposal.load_stored_native_plan(path, metadata, series_id="unit")
    assert len(imported["selected"]) == 3 and imported["readonly"]
    assert imported["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == {str(p): p.read_bytes() for p in root.iterdir() if p.is_file()}
    doc = json.loads(path.read_bytes())
    doc["frame_sources"][0]["elapsed_h"] += 0.01
    path.write_text(json.dumps(doc))
    with pytest.raises(model.ManifestChangedError, match="metadata changed"):
        proposal.load_stored_native_plan(
            path, metadata, series_id="unit", expected=imported["source"]
        )
    # An explicitly chosen historical timeline owns its recorded clock; this
    # does not rewrite the current protocol's time for the same source image.
    changed = proposal.load_stored_native_plan(path, metadata, series_id="unit")
    assert (
        changed["selected"][0]["historical_elapsed_h"]
        == doc["frame_sources"][0]["elapsed_h"]
    )
    assert (
        changed["selected"][0]["protocol_elapsed_h"] == metadata.frames[0]["elapsed_h"]
    )


def test_legacy_pointer_reads_only_explicit_original_project_no_catalog_guess(root):
    from darsia.fluidflower.project import PROJECT_FORMAT

    project = root / "legacy-series.json"
    project.write_text(
        json.dumps(
            dict(
                format=PROJECT_FORMAT,
                schema=1,
                series_id="unit",
                tools={"source_mapping": {}},
            )
        )
    )
    with (root / "series.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[legacy]\nstudio_project="legacy-series.json"\n')
    before = project.read_bytes()
    result = proposal.legacy_stored_plan_status(root)
    assert not result["available"] and "no bound Studio20" in result["note"]
    assert project.read_bytes() == before


def test_qt_proposal_never_changes_stored_or_manual_until_explicit_apply(
    root, metadata, app
):
    before = (root / "series.toml").read_bytes()
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: metadata)
    dialog.load_run()
    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
    initial = dialog.selected_paths()
    dialog.suggest()
    assert (
        dialog.selected_paths() == initial
        and (root / "series.toml").read_bytes() == before
    )
    assert dialog.proposal_table.rowCount() == 20 and dialog.apply_button.isEnabled()
    dialog.apply_proposal()
    assert (
        len(dialog.selected_paths()) > 1
        and (root / "series.toml").read_bytes() == before
    )
    dialog.save_run()
    doc = tomllib.loads((root / "series.toml").read_bytes().decode())
    saved = doc["image_selection"]["runs"]["ac17"]
    assert saved["rule"] == proposal.RULE and saved["default20_d1_h"] == 0.25
    assert saved["default20_proposal"]["parameters"]["t_end_h"] == 1
    assert saved["manual_edits_after_apply"] is False
    dialog.close()
    dialog.deleteLater()


def test_qt_stored_provider_called_only_explicit_load_and_import_survives_proposal(
    root, metadata, app
):
    path = native_file(root, metadata, [0, 4, 12])
    source = model._descriptor(path)
    calls = []
    before = (root / "series.toml").read_bytes()
    dialog = ui.SeriesImageSelectionDialog(
        root,
        metadata_reader=lambda *_: metadata,
        stored_plan_provider=lambda current: (calls.append(current.run_id) or source),
    )
    assert calls == []
    dialog.load_run()
    assert calls == ["ac17"] and dialog.stored_table.rowCount() == 3
    dialog.apply_stored_plan()
    stored = dialog.selected_paths()
    dialog.suggest()
    assert dialog.selected_paths() == stored and dialog.stored_table.rowCount() == 3
    assert path.read_bytes() and (root / "series.toml").read_bytes() == before
    dialog._dirty_runs.clear()
    dialog.close()
    dialog.deleteLater()


def test_saved_proposal_import_quality_binding_revalidated_for_future_job(
    root, metadata
):
    metadata = bound_metadata(root, metadata)
    path = native_file(root, metadata, [0, 3, 10])
    stored = proposal.load_stored_native_plan(path, metadata, series_id="unit")
    quality_path, quality = quality_file(root, metadata, {0: "ok", 1: "minor"})
    generated = proposal.default20_proposal(metadata, quality=quality)
    table, _ = model.save_run_image_selection(
        root,
        metadata,
        [row["source_path"] for row in generated["selected"]],
        expected_bytes=(root / "series.toml").read_bytes(),
        rule=proposal.RULE,
        proposal=generated,
        stored_plan=stored,
        quality=quality,
    )
    resolved = model.resolve_saved_selection(root, "ac17", metadata)
    assert {row["path"] for row in resolved["source_metadata"]} >= {
        str(path),
        str(quality_path),
    }
    quality_path.write_text("{}", encoding="utf-8")
    with pytest.raises(model.ManifestChangedError, match="Imported selection/quality"):
        model.resolve_saved_selection(root, "ac17", metadata)


def test_only_known_reject_pre_allows_first_usable_no_pre_save_and_resolve(
    root, metadata
):
    metadata = bound_metadata(root, metadata)
    _, quality = quality_file(root, metadata, {0: "reject"})
    generated = proposal.default20_proposal(metadata, quality=quality)
    saved, _ = model.save_run_image_selection(
        root,
        metadata,
        [row["source_path"] for row in generated["selected"]],
        expected_bytes=(root / "series.toml").read_bytes(),
        proposal=generated,
        quality=quality,
        rule=proposal.RULE,
    )
    assert saved["runs"]["ac17"]["no_preinjection_image"]
    assert (
        model.resolve_saved_selection(root, "ac17", metadata)["selected"][0][
            "source_path"
        ]
        == metadata.frames[1]["source_path"]
    )


def test_stale_applied_proposal_refuses_before_any_save(root, metadata):
    generated = proposal.default20_proposal(metadata)
    generated["target_matches"][1]["target_time_h"] += 0.001
    before = (root / "series.toml").read_bytes()
    with pytest.raises(model.ManifestChangedError, match="Default20 proposal"):
        model.save_run_image_selection(
            root,
            metadata,
            [row["source_path"] for row in generated["selected"]],
            expected_bytes=before,
            proposal=generated,
            rule=proposal.RULE,
        )
    assert (root / "series.toml").read_bytes() == before and not (
        root / "image_selection"
    ).exists()


@pytest.mark.parametrize("source_kind", ["quality", "stored_plan"])
@pytest.mark.parametrize("changed_source", [False, True])
def test_newer_import_draft_survives_run_switch_and_revalidates_same_source(
    root, metadata, app, source_kind, changed_source
):
    metadata = bound_metadata(root, metadata)
    old_quality_path, old_quality = quality_file(root, metadata, {0: "ok", 1: "minor"})
    old_quality_path.rename(root / "quality-old.json")
    old_quality = proposal.load_saved_quality(root / "quality-old.json", metadata)
    new_quality_path, new_quality = quality_file(root, metadata, {0: "ok", 1: "ok"})
    old_stored_path = native_file(root, metadata, [0, 4, 12])
    old_stored_path.rename(root / "stored-old.json")
    old_stored = proposal.load_stored_native_plan(
        root / "stored-old.json", metadata, series_id="unit"
    )
    new_stored_path = native_file(root, metadata, [0, 7, 13])
    new_stored = proposal.load_stored_native_plan(
        new_stored_path, metadata, series_id="unit"
    )
    model.save_run_image_selection(
        root,
        metadata,
        [metadata.frames[0]["source_path"]],
        expected_bytes=(root / "series.toml").read_bytes(),
        quality=old_quality,
        stored_plan=old_stored,
    )
    before = (root / "series.toml").read_bytes()
    dialog = ui.SeriesImageSelectionDialog(
        root, metadata_reader=lambda _root, run: replace(metadata, run_id=run)
    )
    dialog.load_run()
    if source_kind == "quality":
        dialog.import_quality(new_quality_path)
        dialog.suggest()
        dialog.apply_proposal()
        source_path, new_source = new_quality_path, new_quality
    else:
        dialog.import_stored_plan(new_stored_path)
        dialog.apply_stored_plan()
        source_path, new_source = new_stored_path, new_stored
    selected = dialog.selected_paths()
    dialog.run_combo.setCurrentIndex(1)
    dialog.load_run()
    if changed_source:
        source_path.write_text("{}", encoding="utf-8")
    dialog.run_combo.setCurrentIndex(0)
    dialog.load_run()
    assert (
        dialog.selected_paths() == selected
        and (root / "series.toml").read_bytes() == before
    )
    if changed_source:
        assert getattr(dialog, source_kind) is None
        assert not dialog.save_button.isEnabled()
        with pytest.raises(ValueError, match="reimport"):
            dialog.save_run()
        dialog.run_combo.setCurrentIndex(1)
        dialog.load_run()
        assert dialog.drafts["ac17"][source_kind]["source"] == new_source["source"]
    else:
        assert getattr(dialog, source_kind)["source"] == new_source["source"]
        assert dialog.save_button.isEnabled()
        if source_kind == "quality":
            assert dialog.applied_proposal is not None
    dialog._dirty_runs.clear()
    dialog.close()
    dialog.deleteLater()


def _offset_proposal_fixture(root, metadata):
    metadata = bound_metadata(
        root,
        with_times(metadata, [-0.1, 0.125, 0.125 + 40 / 3600, 0.25, 0.5, 1, 2, 2.1]),
    )
    _, quality = quality_file(root, metadata, {1: "reject", 2: "ok"})
    generated = proposal.default20_proposal(metadata, quality=quality)
    table, raw = model.save_run_image_selection(
        root,
        metadata,
        [r["source_path"] for r in generated["selected"]],
        expected_bytes=(root / "series.toml").read_bytes(),
        quality=quality,
        proposal=generated,
        rule=proposal.RULE,
    )
    return metadata, quality, generated, table, raw


def test_manifest_applied_targets_offsets_and_replacements_keep_camera_times(
    root, metadata
):
    metadata, quality, generated, table, _raw = _offset_proposal_fixture(root, metadata)
    saved = table["runs"]["ac17"]
    manifest = json.loads(Path(saved["selection_manifest"]["path"]).read_bytes())
    repaired = next(row for row in manifest["frames"] if "rejected_image" in row)
    assert repaired["target_time_h"] == 0.125 and repaired[
        "offset_seconds"
    ] == pytest.approx(40)
    assert (
        repaired["data_time_h"]
        == repaired["camera_elapsed_h"]
        == metadata.frames[2]["elapsed_h"]
    )
    assert repaired["selection_origin"] == "default20_proposal"
    assert (
        "reject replaced" in repaired["replacement"]
        and "no image within tolerance" in repaired["flags"]
    )
    document = tomllib.loads((root / "series.toml").read_bytes().decode())
    sections = {entry["name"]: entry for entry in document["data_time"]}
    assert [
        t for name in saved["data_time_names"] for t in sections[name]["times"]
    ] == [r["camera_elapsed_h"] for r in manifest["frames"]]
    assert model.resolve_saved_selection(root, "ac17", metadata)[
        "selected_frame_count"
    ] == len(generated["selected"])


def test_later_manual_save_preserves_proposal_background_and_marks_added_frame(
    root, metadata
):
    metadata = bound_metadata(root, metadata)
    generated = proposal.default20_proposal(metadata)
    initial = [r["source_path"] for r in generated["selected"]]
    first, raw = model.save_run_image_selection(
        root,
        metadata,
        initial,
        expected_bytes=(root / "series.toml").read_bytes(),
        proposal=generated,
        rule=proposal.RULE,
    )
    removed = initial[-1]
    added = next(
        row["source_path"]
        for row in metadata.frames
        if row["source_path"] not in initial
    )
    table, _ = model.save_run_image_selection(
        root,
        metadata,
        [p for p in initial if p != removed] + [added],
        expected_bytes=raw,
    )
    saved = table["runs"]["ac17"]
    manifest = json.loads(Path(saved["selection_manifest"]["path"]).read_bytes())
    # The manifest and series.toml store the same portable (series-relative) text.
    stored = tomllib.loads((root / "series.toml").read_bytes().decode())
    assert saved["default20_proposal"] == first["runs"]["ac17"]["default20_proposal"]
    assert (
        stored["image_selection"]["runs"]["ac17"]["default20_proposal"]
        == manifest["proposal"]
    )
    assert saved["manual_edits_after_apply"] is True
    manual = next(
        row
        for row in manifest["frames"]
        if root / row["chosen_image"]["source_path"] == Path(added)
    )
    assert manual["selection_origin"] == "manual" and manual["offset_seconds"] == 0
    assert (
        manual["target_time_h"] == manual["camera_elapsed_h"] == manual["data_time_h"]
    )
    assert all(
        row["selection_origin"] == "default20_proposal"
        for row in manifest["frames"]
        if row is not manual
    )
    assert model.resolve_saved_selection(root, "ac17", metadata)[
        "selected_frame_count"
    ] == len(initial)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_time_h", 99.0),
        ("offset_seconds", 0.0),
        ("data_time_h", 99.0),
        ("flags", []),
        ("replacement", "No automatic replacement; explicit manual choice"),
        (
            "chosen_image",
            {
                "source_path": "another.png",
                "image_id": "another.png",
                "frame_index": 99,
            },
        ),
    ],
)
def test_resolver_binds_applied_provenance_even_if_manifest_hash_is_redeclared(
    root, metadata, field, value
):
    metadata, _quality, _generated, table, _raw = _offset_proposal_fixture(
        root, metadata
    )
    path = root / "series.toml"
    document = tomllib.loads(path.read_bytes().decode())
    saved = document["image_selection"]["runs"]["ac17"]
    manifest_path = root / saved["selection_manifest"]["path"]  # series-relative
    manifest = json.loads(manifest_path.read_bytes())
    repaired = next(row for row in manifest["frames"] if "rejected_image" in row)
    repaired[field] = value
    raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(raw)
    saved["selection_manifest"]["sha256"] = hashlib.sha256(raw).hexdigest()
    saved["selection_manifest"]["size_bytes"] = len(raw)
    path.write_text(model.toml.dumps(document), encoding="utf-8")
    with pytest.raises(model.ManifestChangedError, match="different choices"):
        model.resolve_saved_selection(root, "ac17", metadata)


@pytest.mark.parametrize("historical", [0.24720805555555556, 0.24719500000000003])
def test_historical_clock_is_displayed_and_saved_separately_from_protocol(
    root, metadata, app, historical
):
    metadata = bound_metadata(
        root,
        with_times(metadata, [-0.10277777777777777, 0.24722222222222223, 0.5, 1, 2]),
    )
    path = native_file(root, metadata, [0, 1, 4])
    native = json.loads(path.read_bytes())
    native["frame_sources"][1]["elapsed_h"] = historical
    native["frame_sources"][1]["target_time_h"] = 0.25
    path.write_text(json.dumps(native), encoding="utf-8")
    original = path.read_bytes()
    before_series = (root / "series.toml").read_bytes()
    dialog = ui.SeriesImageSelectionDialog(root, metadata_reader=lambda *_: metadata)
    dialog.load_run()
    dialog.import_stored_plan(path)
    assert (
        path.read_bytes() == original
        and (root / "series.toml").read_bytes() == before_series
    )
    assert dialog.stored_table.columnCount() == 4
    assert dialog.stored_table.item(1, 1).text() == str(historical)
    assert dialog.stored_table.item(1, 2).text() == str(metadata.frames[1]["elapsed_h"])
    assert "clock values differ" in dialog.stored_table.item(1, 1).toolTip()
    dialog.apply_stored_plan()
    dialog.save_run()
    saved = model.read_series_image_selection(root)[0]["runs"]["ac17"]
    manifest = json.loads(Path(saved["selection_manifest"]["path"]).read_bytes())
    identity = manifest["stored_plan"]["native_source_identities"][1]
    assert (
        identity["historical_elapsed_h"] == historical
        and identity["elapsed_h"] == historical
    )
    assert (
        identity["historical_target_time_h"] == 0.25
        and identity["clock_difference"] is True
    )
    assert (
        identity["protocol_elapsed_h"]
        == identity["data_time_h"]
        == metadata.frames[1]["elapsed_h"]
    )
    assert identity["source_sha256"] == native["frame_sources"][1]["source_sha256"]
    # Both files store the same portable (series-relative) provenance text.
    stored = tomllib.loads((root / "series.toml").read_bytes().decode())
    assert (
        manifest["stored_plan"]
        == stored["image_selection"]["runs"]["ac17"]["stored_plan_provenance"]
    )
    assert manifest["frames"][1]["data_time_h"] == metadata.frames[1]["elapsed_h"]
    assert (
        model.resolve_saved_selection(root, "ac17", metadata)["selected"][1][
            "elapsed_h"
        ]
        == metadata.frames[1]["elapsed_h"]
    )
    assert path.read_bytes() == original
    dialog.close()
    dialog.deleteLater()


@pytest.mark.parametrize(
    "field", ["historical_elapsed_h", "protocol_elapsed_h", "source_sha256"]
)
def test_resolver_binds_both_stored_clock_identities_even_with_redeclared_sha(
    root, metadata, field
):
    metadata = bound_metadata(root, metadata)
    source = native_file(root, metadata, [0, 3, 10])
    native = json.loads(source.read_bytes())
    native["frame_sources"][1]["elapsed_h"] += 0.001
    source.write_text(json.dumps(native), encoding="utf-8")
    stored = proposal.load_stored_native_plan(source, metadata, series_id="unit")
    table, _ = model.save_run_image_selection(
        root,
        metadata,
        [row["source_path"] for row in stored["selected"]],
        expected_bytes=(root / "series.toml").read_bytes(),
        stored_plan=stored,
    )
    series = root / "series.toml"
    document = tomllib.loads(series.read_bytes().decode())
    saved = document["image_selection"]["runs"]["ac17"]
    manifest_path = root / saved["selection_manifest"]["path"]  # series-relative
    manifest = json.loads(manifest_path.read_bytes())
    wrong = "f" * 64 if field == "source_sha256" else 99.0
    manifest["stored_plan"]["native_source_identities"][1][field] = wrong
    saved["stored_plan_provenance"]["native_source_identities"][1][field] = wrong
    raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(raw)
    saved["selection_manifest"]["sha256"] = hashlib.sha256(raw).hexdigest()
    saved["selection_manifest"]["size_bytes"] = len(raw)
    series.write_text(model.toml.dumps(document), encoding="utf-8")
    with pytest.raises(
        model.ManifestChangedError, match="native/protocol clock provenance"
    ):
        model.resolve_saved_selection(root, "ac17", metadata)
