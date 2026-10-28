"""One saved Jobs queue is shared by Setup and Jobs, with machine-local mounts."""
from pathlib import Path
import tomllib

import toml

from darsia.fluidflower import series_creation as creation, series_locations as locations
from darsia.fluidflower.series_paths import series_folder_path
from darsia.fluidflower.series_wizard import NewSeriesFolderWizard
from .test_series_locations import draft  # noqa: F401
from .test_series_setup_jobs_ui import owner  # noqa: F401


def test_early_setup_job_folder_is_saved_and_custom_choice_survives_results_change(draft, qtbot):
    wizard = NewSeriesFolderWizard(experiment_folder=draft.darsia_folder.parent)
    qtbot.addWidget(wizard)
    page = wizard.series_page
    page.name.setText(draft.name)
    page.series_id.setText(draft.series_id)
    page.rig_class.setCurrentIndex(page.rig_class.findData('medium'))
    page.physical_rig.setCurrentIndex(1)
    page.project_folder.setText(str(draft.darsia_folder.parent))
    chosen = draft.darsia_folder.parent / 'shared queue'
    page.jobs_folder.setText(str(chosen))
    page.results_folder.setText(str(draft.darsia_folder.parent / 'outputs'))
    wizard.experiments_page.runs = draft.runs
    root = wizard.create()
    assert root == draft.darsia_folder
    assert series_folder_path(root, 'jobs').resolve() == chosen
    assert chosen.is_dir()
    assert tomllib.loads((root/'series.toml').read_text())['roots']['jobs']


def test_two_machine_mounts_preserve_one_shared_location(draft, tmp_path):
    root = creation.create_series_folder(draft)
    chosen = root.parent / 'shared queue'
    loaded = locations.read_series_locations(root)
    result = locations.save_series_jobs_folder(root, chosen, expected_bytes=loaded['manifest_bytes'])
    assert result['jobs_folder'] == chosen and not chosen.exists()
    before = (root/'series.toml').read_bytes()
    for number in (1, 2):
        mount = tmp_path / f'machine{number}' / 'mounted jobs'
        path = tmp_path / f'stores{number}.toml'
        path.write_text(toml.dumps(dict(machine=dict(id=f'machine{number}'), stores=dict(Study=dict(
            series=str(root), results=str(root.parent/'Results'), jobs=str(mount))))), encoding='utf-8')
        assert locations.machine_jobs_folder(root, chosen, path) == mount
    assert (root/'series.toml').read_bytes() == before
    assert locations.machine_jobs_folder(root, root.parent/'other queue', path) == root.parent/'other queue'


def test_jobs_save_and_setup_settings_round_trip_one_queue(owner, qtbot, tmp_path):
    shared = owner.jobs.shared_jobs
    chosen = tmp_path/'chosen shared jobs'
    shared.folder.setText(str(chosen))
    shared.save_folder_button.click()
    qtbot.waitUntil(lambda: shared._thread is None, timeout=10000)
    assert locations.read_series_locations(owner.workspace._loaded_root)['jobs_folder'] == chosen
    assert not chosen.exists(), 'Saving queue settings starts no job and creates no queue data'
    editor = owner.jobs.inline_layout.store_editor
    qtbot.waitUntil(lambda: editor._thread is None, timeout=10000)
    assert editor.jobs_location.text() == str(chosen)
    second = tmp_path/'edited from setup locations'
    editor.jobs_location.setText(str(second))
    editor.save_locations()
    qtbot.waitUntil(lambda: editor._thread is None, timeout=10000)
    qtbot.waitUntil(owner.jobs.can_close, timeout=10000)
    assert shared.folder.text() == str(second)
    assert series_folder_path(owner.workspace._loaded_root, 'jobs') == second
