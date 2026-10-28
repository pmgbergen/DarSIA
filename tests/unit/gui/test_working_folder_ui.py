"""Working folders are editable per machine without changing shared result choices."""
import hashlib
import pytest
import toml
from darsia.fluidflower.store_map import parse_store_map
from darsia.fluidflower.store_map_ui import StoreSettingsDialog


@pytest.mark.parametrize('shared_work', [False, True])
def test_working_folder_round_trips_through_local_settings(qtbot,tmp_path,shared_work):
    series=tmp_path/'series'
    result=r'\\server\share\Results'
    document=dict(machine=dict(id='Main'),stores=dict(Study=dict(series=str(series),results=result)),future=dict(keep=17))
    raw=toml.dumps(document).encode()
    path=tmp_path/'stores.toml';path.write_bytes(raw)
    def read(selected,**kwargs):return parse_store_map(path.read_bytes(),source=dict(path=str(path)))
    saved=[]
    def write(selected,content,**kwargs):
        saved.append(content)
        return dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),previous_copy='old')
    editor=StoreSettingsDialog(series,path=path,reader=read,writer=write)
    qtbot.addWidget(editor)
    editor.load();qtbot.waitUntil(lambda:editor._thread is None)
    assert editor.working_location.text()==result
    work=r'\\server\share\Working' if shared_work else str(tmp_path/'local work')
    editor.working_location.setText(work)
    editor.working_location.editingFinished.emit()
    candidate=parse_store_map(editor.editor.toPlainText().encode())
    assert candidate.roots('Study')['working']==work
    assert candidate.roots('Study')['results']==result
    assert candidate.document['future']==document['future']
    assert path.read_bytes()==raw and not saved
    editor.save();qtbot.waitUntil(lambda:editor._thread is None)
    assert parse_store_map(saved[0]).roots('Study')['working']==work


@pytest.mark.parametrize('shared_cache', [False, True])
def test_cache_folder_is_a_saved_machine_choice_without_shared_or_data_writes(qtbot,tmp_path,shared_cache):
    series=tmp_path/'series'
    result=r'\\server\share\Results'
    document=dict(machine=dict(id='Main'),stores=dict(Study=dict(series=str(series),results=result)),future=dict(keep=17))
    raw=toml.dumps(document).encode()
    path=tmp_path/'stores.toml';path.write_bytes(raw)
    saved=[]
    def read(selected,**kwargs):return parse_store_map(raw,source=dict(path=str(path)))
    def write(selected,content,**kwargs):
        assert kwargs['expected_bytes']==raw
        saved.append(content)
        return dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),previous_copy='old')
    editor=StoreSettingsDialog(series,path=path,reader=read,writer=write)
    qtbot.addWidget(editor)
    editor.load();qtbot.waitUntil(lambda:editor._thread is None)
    cache=r'\\server\share\Cache' if shared_cache else str(tmp_path/'local cache')
    editor.cache_location.setText(cache)
    editor.cache_location.editingFinished.emit()
    candidate=parse_store_map(editor.editor.toPlainText().encode())
    assert candidate.roots('Study')['cache']==cache
    assert candidate.roots('Study')['results']==result
    assert candidate.document['future']==document['future']
    assert not (tmp_path/'local cache').exists()
    assert path.read_bytes()==raw and not saved
    editor.save();qtbot.waitUntil(lambda:editor._thread is None)
    assert parse_store_map(saved[0]).roots('Study')['cache']==cache
