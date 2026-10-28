"""Start folder controls preserve scientific files and share established settings."""
from pathlib import Path
import tomllib

import pytest
import toml

from darsia.fluidflower.series_folders_ui import (
    SeriesFoldersPanel, read_image_locations, save_image_locations,
    select_existing_setup, remove_series_registration,
)
from darsia.fluidflower.series_method import ManifestChangedError
from darsia.fluidflower.series_paths import configured_series, register_series
from .test_native_sidebar import native_window  # noqa:F401


@pytest.fixture
def selected(tmp_path, monkeypatch):
    registry=tmp_path/'registry.toml'
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY',str(registry))
    root=tmp_path/'project'/'DarSIA'; root.mkdir(parents=True)
    images=tmp_path/'images';images.mkdir()
    (images/'baseline.JPG').write_bytes(b'unchanged scientific input')
    document=dict(schema_version=1,series_id='study',name='Study',
        roots=dict(series='.',results='../Results',raw=str(images)),
        paths=dict(project='..',results='../Results',jobs='../Results/jobs'),runs=dict(ids=[]),future=dict(keep=17))
    (root/'series.toml').write_text(toml.dumps(document),encoding='utf-8')
    register_series(root)
    return root,images,registry


def test_image_location_save_retains_inputs_and_other_manifest_fields(selected,tmp_path):
    root,images,_=selected
    snapshot=read_image_locations(root)
    moved=tmp_path/'moved images';moved.mkdir()
    (moved/'baseline.JPG').write_bytes((images/'baseline.JPG').read_bytes())
    result=save_image_locations(snapshot,{'raw':str(moved)})
    document=tomllib.loads((root/'series.toml').read_text())
    assert document['storage']['path_aliases'][str(images)]==str(moved)
    assert document['future']=={'keep':17}
    assert (images/'baseline.JPG').read_bytes()==b'unchanged scientific input'
    assert result['images'][0]['folder']==str(moved)
    with pytest.raises(ManifestChangedError):save_image_locations(snapshot,{'raw':str(images)})


def test_existing_setup_selection_and_removal_keep_all_project_files(selected,tmp_path):
    root,images,registry=selected
    other=tmp_path/'other DarSIA';other.mkdir()
    original=(root/'series.toml').read_bytes()
    (other/'series.toml').write_bytes(original)
    assert select_existing_setup(root,other)==other
    assert configured_series()[0][2]==other
    with pytest.raises(ValueError):
        foreign=tmp_path/'foreign';foreign.mkdir()
        (foreign/'series.toml').write_text('schema_version=1\nseries_id="another"\n')
        select_existing_setup(other,foreign)
    remove_series_registration(other)
    assert not configured_series()
    assert (root/'series.toml').read_bytes()==original
    assert (other/'series.toml').read_bytes()==original
    assert (images/'baseline.JPG').read_bytes()==b'unchanged scientific input'


def test_panel_is_embedded_and_reuses_shared_settings(selected,qtbot):
    root,_,_=selected
    panel=SeriesFoldersPanel(root)
    qtbot.addWidget(panel)
    qtbot.waitUntil(panel.can_close,timeout=10000)
    assert not panel.settings.isWindow()
    assert panel.settings.series_root==root
    assert panel.images.rowCount()==1
    assert panel.settings.results_location.text()==str(root.parent/'Results')
    assert panel.settings.jobs_location.text()==str(root.parent/'Results/jobs')
    assert hasattr(panel.settings,'working_location')


@pytest.mark.parametrize('native_window',[('series-a','series-b')],indirect=True)
def test_start_selection_and_folder_open_do_not_load_series_or_start_jobs(native_window,monkeypatch,qtbot):
    window=native_window
    def forbidden(*args,**kwargs):pytest.fail('Selecting folders must not start series loading or a job')
    monkeypatch.setattr(window,'_select_series',forbidden)
    monkeypatch.setattr(window._series_loader,'request',forbidden,raising=False)
    monkeypatch.setattr(window.process_runner,'start_workflow_process',forbidden)
    table=window.home_page.table
    target=next(i for i in range(table.rowCount()) if table.item(i,1).text()=='series-b')
    table.selectRow(target)
    table.cellClicked.emit(target,0)
    assert window.content_stack.currentWidget() is window.home_page
    window.home_page.folders_button.click()
    panel=window._series_folders_panel
    qtbot.waitUntil(panel.can_close,timeout=10000)
    assert window.content_stack.currentWidget() is panel
    assert panel.root==window.home_page.series_registry.series[1][2]

def test_image_alias_cycle_refuses_before_writing(selected,tmp_path):
    root,images,_=selected
    other=tmp_path/'other images';other.mkdir()
    first=read_image_locations(root)
    save_image_locations(first,{'raw':str(other)})
    second=read_image_locations(root)
    before=(root/'series.toml').read_bytes()
    with pytest.raises(ValueError,match='Cyclic'):
        save_image_locations(second,{'raw':str(images)})
    assert (root/'series.toml').read_bytes()==before


@pytest.mark.parametrize('native_window',[('series-a','series-b')],indirect=True)
def test_remove_start_action_does_not_touch_workers_or_project_files(native_window,monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    from darsia.fluidflower import series_folders_ui as folders
    window=native_window
    table=window.home_page.table
    target=next(i for i in range(table.rowCount()) if table.item(i,1).text()=='series-b')
    table.selectRow(target)
    root=Path(window.home_page._selected_project_path())
    before=(root/'series.toml').read_bytes()
    state=dict(running=True,progress=12)
    window.jobs_workspace=SimpleNamespace(can_close=lambda:True,state=state)
    unchanged=deepcopy(state)
    removed=[]; refreshed=[]
    monkeypatch.setattr(folders,'remove_series_registration',lambda path:removed.append(path))
    monkeypatch.setattr(window,'_refresh_series',lambda **kwargs:refreshed.append(kwargs))
    window.home_page.remove_series_button.click()
    assert removed==[str(root)] and refreshed==[dict(show_errors=True)]
    assert window.jobs_workspace.state==unchanged
    assert (root/'series.toml').read_bytes()==before
    window.jobs_workspace=None

def test_legacy_project_removal_keeps_source_files_and_other_registrations(tmp_path):
    import json
    from darsia.fluidflower.project import ProjectRegistry,REGISTRY_FORMAT
    root=tmp_path/'legacy';root.mkdir()
    (root/'series.json').write_bytes(b'original project')
    path=tmp_path/'projects.json'
    document=dict(schema=1,format=REGISTRY_FORMAT,future=dict(keep=17),projects=[
        dict(path=str(root),project_id='one'),dict(path=str(tmp_path/'other'),project_id='two')])
    path.write_text(json.dumps(document))
    registry=ProjectRegistry(path)
    assert registry.forget_path(root)
    kept=json.loads(path.read_text())
    assert kept['projects']==[document['projects'][1]] and kept['future']==document['future']
    assert (root/'series.json').read_bytes()==b'original project'
    assert registry.forget_path(root) is False

def test_image_draft_rebases_only_unrelated_saved_folder_changes(selected,tmp_path):
    from darsia.fluidflower.series_locations import save_series_jobs_folder
    root,images,_=selected
    snapshot=read_image_locations(root)
    save_series_jobs_folder(root,tmp_path/'new jobs')
    destination=tmp_path/'chosen images';destination.mkdir()
    save_image_locations(snapshot,dict(raw=str(destination)))
    document=tomllib.loads((root/'series.toml').read_text())
    assert document['paths']['jobs']=='../../new jobs'
    assert document['storage']['path_aliases'][str(images)]==str(destination)


def test_panel_automatically_loads_existing_working_settings(selected,qtbot,tmp_path):
    from darsia.fluidflower.store_map_ui import StoreSettingsDialog
    from darsia.fluidflower.store_map import parse_store_map
    root,_,_=selected
    stores=tmp_path/'stores.toml'
    working=tmp_path/'working'
    stores.write_text(toml.dumps(dict(machine=dict(id='Here'),stores=dict(study=dict(
        series=str(root),results=str(root.parent/'Results'),working=str(working))))))
    def settings(root,parent):
        return StoreSettingsDialog(root,parent,path=stores,
            reader=lambda path,**kwargs:parse_store_map(stores.read_bytes(),source=dict(path=str(stores))))
    panel=SeriesFoldersPanel(root,settings_factory=settings)
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda:panel.settings._loaded and panel.can_close(),timeout=10000)
    assert panel.settings.working_location.text()==str(working)
    assert not panel.settings.working_location.isReadOnly()
