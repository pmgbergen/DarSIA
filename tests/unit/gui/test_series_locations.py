"""Normal location choices round-trip to native readers and portable workers."""
from pathlib import Path
import tomllib

import pytest
import toml

from darsia.fluidflower import series_creation as creation, series_locations as locations
from darsia.fluidflower.series_paths import series_folder_path
from darsia.fluidflower.store_map import manifest_roots, parse_store_map
from darsia.fluidflower.store_map_ui import StoreSettingsDialog
from darsia.fluidflower.series_wizard import NewSeriesFolderWizard
from darsia.presets.workflows.config.data import DataConfig
from darsia.presets.workflows.config.rig import RigConfig


@pytest.fixture
def draft(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    source = tmp_path / "photographs" / "experiment"
    source.mkdir(parents=True)
    (source / "baseline.jpg").write_bytes(b"metadata tests never decode this image")
    project = tmp_path / "project"
    project.mkdir()
    return creation.SeriesDraft("Study", "Study", "medium", project / "DarSIA",
        [creation.RunDraft("r1", [source], baseline="baseline.jpg")])


@pytest.mark.parametrize("custom_results", [False, True])
def test_wizard_choices_feed_shared_manifest_native_output_and_worker_roots(draft, qtbot, custom_results):
    wizard = NewSeriesFolderWizard(experiment_folder=draft.runs[0].folders[0].parent)
    qtbot.addWidget(wizard)
    page = wizard.series_page
    page.name.setText(draft.name)
    page.series_id.setText(draft.series_id)
    page.rig_class.setCurrentIndex(page.rig_class.findData("medium"))
    page.physical_rig.setCurrentIndex(1)
    page.project_folder.setText(str(draft.darsia_folder.parent))
    expected = draft.darsia_folder.parent / ("chosen outputs" if custom_results else "Results")
    if custom_results:
        page.results_folder.setText(str(expected))
    wizard.experiments_page.runs = draft.runs
    root = wizard.create()
    assert root == draft.darsia_folder
    manifest = tomllib.loads((root / "series.toml").read_text())
    assert series_folder_path(root, "models").resolve() == expected / "calibration/ExpertKnowledge/PhaseModels/Models"
    assert series_folder_path(root, "geometry_facies").resolve() == expected / "setup/Geometry"
    assert series_folder_path(root, "data").resolve() == expected / "setup/inputs"
    assert not (expected / "setup/inputs").exists()
    assert series_folder_path(root, "calibration").resolve() == expected / "calibration"
    assert series_folder_path(root, "image_selection").resolve() == expected / "setup/image_selection"
    assert series_folder_path(root, "results").resolve() == expected
    roots = manifest_roots(root, manifest)
    assert Path(roots["results"]) == expected
    assert any(Path(value) == draft.runs[0].folders[0] for value in roots.values())
    config = root / "config/runs/r1.toml"
    data = DataConfig().load(config, require_data=False, require_results=False)
    assert data.results == expected / "r1"
    assert data.folders == draft.runs[0].folders and data.cache is None
    rig = RigConfig().load([root / "config/common.toml", config], results=data.results)
    assert rig.path == expected / "r1/setup/rig"
    assert series_folder_path(root, "jobs").resolve() == expected / "jobs"
    assert (expected / "jobs").is_dir() and not (root / "jobs").exists()
    assert not (root / "results").exists()
    assert (draft.runs[0].folders[0] / "baseline.jpg").read_bytes().startswith(b"metadata tests")


def test_explicit_setup_and_results_choices_survive_project_field_changes(draft, qtbot):
    wizard = NewSeriesFolderWizard(experiment_folder=draft.darsia_folder.parent)
    qtbot.addWidget(wizard)
    page = wizard.series_page
    chosen_setup = draft.darsia_folder.parent / "chosen setup"
    chosen_results = draft.darsia_folder.parent / "chosen outputs"
    page.setup_folder.setText(str(chosen_setup))
    page.results_folder.setText(str(chosen_results))
    page.project_folder.setText(str(draft.darsia_folder.parent / "another project"))
    assert wizard.draft().darsia_folder == chosen_setup
    assert creation.results_folder(wizard.draft()) == chosen_results


def test_output_change_preserves_custom_outputs_and_implicit_inputs(draft):
    from darsia.presets.workflows.config.color_embedding_registry import ColorEmbeddingRegistry

    root = creation.create_series_folder(draft)
    original_results = root.parent / "Results"
    common_path = root / "config/common.toml"
    common = tomllib.loads(common_path.read_text())
    common["color_path"] = [dict(name="main", mode="relative", basis="labels")]
    common_path.write_text(toml.dumps(common))
    manifest_path = root / "series.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    manifest["runs"]["ids"].append("r2")
    manifest["runs"]["config_files"]["r2"] = "config/runs/r2.toml"
    manifest["future"] = dict(keep="unchanged")
    manifest_path.write_text(toml.dumps(manifest))
    own = root / "config/runs/r1.toml"
    custom = root.parent / "custom output"
    second = tomllib.loads(own.read_text())
    second["data"]["results"] = str(custom)
    (root / "config/runs/r2.toml").write_text(toml.dumps(second))
    marker = original_results / "keep.txt"
    marker.write_bytes(b"existing product")
    loaded = locations.read_series_locations(root)
    target = root.parent / "new outputs"
    result = locations.save_series_locations(root, root.parent, target, expected_bytes=loaded["manifest_bytes"])
    assert result["retained_runs"] == ["r2"]
    for run, expected_output, old in (("r1", target / "r1", original_results / "r1"), ("r2", custom, custom)):
        config = root / "config/runs" / (run + ".toml")
        data = DataConfig().load(config, require_data=False, require_results=False)
        assert data.results == expected_output
        rig = RigConfig().load([common_path, config], results=data.results)
        assert rig.path == old / "setup/rig"
        color = ColorEmbeddingRegistry().load([common_path, config], data=data.folder, results=data.results)
        assert color.resolve("main").root == old / "color/color_path/main"
    assert marker.read_bytes() == b"existing product"
    assert tomllib.loads(manifest_path.read_text())["future"] == {"keep": "unchanged"}
    assert series_folder_path(root, "results").resolve() == target


def test_shared_editor_save_and_machine_copy_keep_caches_and_unknown_settings(draft, qtbot, tmp_path):
    root = creation.create_series_folder(draft)
    path = tmp_path / "stores.toml"
    document = {"machine": {"id": "local"}, "stores": {"Study": {
        "series": str(root), "results": str(tmp_path / "old results"), "source_cache": str(tmp_path / "local cache")}},
        "future": {"keep": 17}}
    raw = toml.dumps(document).encode()
    path.write_bytes(raw)
    def reader(selected, **kwargs):
        return parse_store_map(path.read_bytes(), source={"path": str(path)})
    writes = []
    def writer(selected, data, **kwargs):
        writes.append(data)
        return {"path": str(path), "sha256": "a" * 64, "previous_copy": "previous"}
    dialog = StoreSettingsDialog(root, path=path, reader=reader, writer=writer)
    qtbot.addWidget(dialog)
    dialog.load_locations()
    qtbot.waitUntil(lambda: dialog._thread is None)
    moved = []
    dialog.set_results_migration_handler(moved.append)
    assert dialog.move_results_button.isEnabled()
    dialog.request_results_migration()
    assert moved[0]["series_root"] == root and not writes
    selected = root.parent / "selected results"
    dialog.results_location.setText(str(selected))
    dialog.save_locations()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert locations.read_series_locations(root)["results_folder"] == selected
    dialog.load()
    qtbot.waitUntil(lambda: dialog._thread is None)
    dialog.use_series_locations()
    qtbot.waitUntil(lambda: dialog._thread is None)
    candidate = parse_store_map(dialog.editor.toPlainText().encode())
    assert candidate.roots("Study")["results"] == str(selected)
    assert candidate.roots("Study")["source_cache"] == str(tmp_path / "local cache")
    assert candidate.document["future"] == document["future"]
    assert not writes and path.read_bytes() == raw
    dialog.save()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert len(writes) == 1 and parse_store_map(writes[0]).roots("Study")["results"] == str(selected)


def test_external_results_save_hook_finds_the_registered_setup_folder(draft, monkeypatch):
    from darsia.fluidflower.mass_calibration_library import series_root_for_results

    root = creation.create_series_folder(draft)
    monkeypatch.setenv("DARSIA_STUDY_PROJECT_ROOT", str(root))
    assert series_root_for_results(root.parent / "Results/concentration_mass") == root


def test_results_move_requires_review_and_explicit_apply(draft, qtbot, monkeypatch):
    import sys
    from types import ModuleType

    root = creation.create_series_folder(draft)
    old = root.parent / "Results"
    target = root.parent / "moved results"
    original = (root / "series.toml").read_bytes()
    calls = []
    backend = ModuleType("darsia.fluidflower.series_storage_migration")
    def preview(series_root, project, results, *, expected_bytes, replace_custom_results=False):
        calls.append("preview")
        assert expected_bytes == original
        assert replace_custom_results is False
        return dict(old_results_folder=str(old), new_results_folder=str(results))
    def apply(plan, *, writers_quiescent):
        calls.append("apply")
        assert writers_quiescent and plan["new_results_folder"] == str(target)
        return locations.save_series_locations(root, root.parent, target, preserve_inputs=False,
            path_aliases={str(old): str(target)})
    backend.preview_results_migration, backend.apply_results_migration = preview, apply
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    dialog = StoreSettingsDialog(root)
    qtbot.addWidget(dialog)
    dialog.load_locations()
    qtbot.waitUntil(lambda: dialog._thread is None)
    dialog.results_location.setText(str(target))
    dialog.request_results_migration()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert calls == ["preview"] and str(old) in dialog.preview.toPlainText()
    assert (root / "series.toml").read_bytes() == original
    assert not dialog.apply_move_button.isEnabled()
    dialog.move_writers_closed.setChecked(True)
    assert dialog.apply_move_button.isEnabled()
    dialog.replace_run_results.setChecked(True)
    assert not dialog.apply_move_button.isEnabled()
    dialog.replace_run_results.setChecked(False)
    dialog.apply_move_button.click()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert calls == ["preview", "apply"]
    saved = tomllib.loads((root / "series.toml").read_text())
    assert saved["storage"]["path_aliases"][str(old)] == str(target)
    assert saved["paths"]["results"] == "../moved results"


@pytest.mark.parametrize("foreign_edit", [False, True])
def test_late_save_failure_restores_only_own_successful_metadata_writes(draft, monkeypatch, foreign_edit):
    from darsia.fluidflower import series_method

    root = creation.create_series_folder(draft)
    run = root / "config/runs/r1.toml"
    original = {path: path.read_bytes() for path in (run, root / "series.toml")}
    write = series_method._write_manifest
    concurrent = b'# another editor owns this change\n[data]\nresults="other"\n'
    def late_failure(path, before, updated):
        if path.name == "series.toml":
            if foreign_edit:
                run.write_bytes(concurrent)
            raise OSError("forced late failure")
        return write(path, before, updated)
    monkeypatch.setattr(series_method, "_write_manifest", late_failure)
    with pytest.raises(OSError, match="forced late failure"):
        locations.save_series_locations(root, root.parent, root.parent / "new outputs")
    assert (root / "series.toml").read_bytes() == original[root / "series.toml"]
    assert run.read_bytes() == (concurrent if foreign_edit else original[run])
    assert not list(root.rglob("*.rollback"))
