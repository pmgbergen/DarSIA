"""Native settings show the same relocated sources as the original protocol reader."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib
import pytest,toml
from PySide6.QtWidgets import QWidget,QFormLayout,QLineEdit
from darsia.gui.ui.settings import SettingsFactory,unwrap_composite_widget
from darsia.fluidflower.settings_storage_aliases import display_storage_aliases
from darsia.presets.workflows.config.protocols import ProtocolsConfig


@pytest.fixture
def panel(tmp_path,monkeypatch,qtbot):
    old=tmp_path/'old';new=tmp_path/'current';new.mkdir()
    data=new/'photos';data.mkdir()
    for name in ['imaging.csv','injection.csv','pressure.csv','changed.csv']:
        (new/name).write_text('time,value\n0,1\n')
    registry=tmp_path/'registry.toml'
    registry.write_text(toml.dumps(dict(schema_version=1,storage=dict(path_aliases={str(old):str(new)}))))
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY',str(registry))
    document=dict(data=dict(folders=[str(old/'photos')]), protocols=dict(
        imaging={str(old/'photos'):str(old/'imaging.csv')},injection=str(old/'injection.csv'),
        pressure_temperature=str(old/'pressure.csv'),imaging_mode='Load from CSV'))
    path=tmp_path/'run.toml'
    path.write_bytes(b'# authored references remain byte-for-byte\r\n'+toml.dumps(document).replace('\n','\r\n').encode())
    window=QWidget();qtbot.addWidget(window)
    window.config_dict=deepcopy(document);window.config_file=str(path);window.settings_inputs={};window.chosen_files={}
    window._last_settings_view=None
    window.print_log=lambda *args:None
    window.settings_factory=SettingsFactory(window)
    display_storage_aliases(window)
    factory=window.settings_factory
    form=QFormLayout(window)
    fields={}
    for key in ['protocols.imaging','protocols.injection','protocols.pressure_temperature']:
        spec=dict(key=key,name=key,type='path_map' if key.endswith('.imaging') else 'file')
        if spec['type']=='path_map':spec['key_source']='data.folders'
        label,result=factory.create_setting_edit(spec,form_context=dict(form=form))
        form.addRow(label,result['widget'] if isinstance(result,dict) else result)
        window.settings_inputs[key]=factory._wrap_multi_row_result(result) if isinstance(result,dict) else result
        fields[key]=result
    return SimpleNamespace(window=window,factory=factory,fields=fields,path=path,new=new,old=old,document=document)


def test_shown_protocol_paths_and_folder_keys_equal_the_execution_reader(panel):
    actual=ProtocolsConfig().load(panel.path)
    key,value=panel.fields['protocols.imaging']['rows'][0]
    assert (Path(key.text()),Path(value.text()))==next(iter(actual.imaging.items()))
    assert Path(unwrap_composite_widget(panel.fields['protocols.injection']).text())==actual.injection
    assert Path(unwrap_composite_widget(panel.fields['protocols.pressure_temperature']).text())==actual.pressure_temperature
    assert panel.window.config_dict==panel.document
    original=panel.path.read_bytes()
    panel.factory._sync_settings_inputs_to_config_dict()
    assert panel.window.config_dict==panel.document
    panel.factory.save_settings()
    assert panel.path.read_bytes()==original


def test_changed_protocol_uses_normal_writer_and_matches_after_reload(panel):
    field=unwrap_composite_widget(panel.fields['protocols.injection'])
    field.setText(str(panel.new/'changed.csv'))
    panel.factory.save_settings()
    saved=tomllib.loads(panel.path.read_text())
    assert saved['protocols']['injection']==str(panel.new/'changed.csv')
    assert saved['protocols']['imaging']==panel.document['protocols']['imaging']
    assert saved['data']==panel.document['data']
    actual=ProtocolsConfig().load(panel.path)
    assert actual.injection==Path(field.text())
    panel.window.config_dict=saved
    assert panel.factory.get_value(saved,'protocols.injection')==str(actual.injection)


def test_editing_one_map_row_preserves_its_authored_key(panel):
    key,value=panel.fields['protocols.imaging']['rows'][0]
    value.setText(str(panel.new/'changed.csv'))
    panel.factory.save_settings()
    actual=ProtocolsConfig().load(panel.path)
    assert actual.imaging=={Path(key.text()):Path(value.text())}
    saved=tomllib.loads(panel.path.read_text())
    assert saved['protocols']['imaging']=={str(panel.old/'photos'):str(panel.new/'changed.csv')}


def test_missing_relocated_file_stays_the_selected_path(panel):
    (panel.new/'injection.csv').unlink()
    actual=ProtocolsConfig().load(panel.path)
    shown=unwrap_composite_widget(panel.fields['protocols.injection']).text()
    assert actual.injection==Path(shown)
    with pytest.raises(FileNotFoundError):actual.injection.read_bytes()


def test_saving_workbook_map_preserves_file_and_worksheet_pair(panel):
    key,value=panel.fields['protocols.imaging']['rows'][0]
    pair=[str(panel.new/'imaging.xlsx'),'run-a']
    value.setText(repr(pair))
    panel.factory.save_settings()
    saved=tomllib.loads(panel.path.read_text())
    assert saved['protocols']['imaging']=={str(panel.old/'photos'):pair}
    assert ProtocolsConfig().load(panel.path).imaging=={Path(key.text()):(Path(pair[0]),pair[1])}


def test_relative_workbooks_show_execution_paths_and_unchanged_save_keeps_bytes(panel):
    (panel.path.parent/'series.toml').write_text('schema_version = 1\n')
    document=deepcopy(panel.document)
    document['protocols']['imaging']={str(panel.old/'photos'):['current/imaging.xlsx','run-a']}
    document['protocols']['injection']=['current/injection.xlsx','run-a']
    original=b'# authored relative workbook locators\n'+toml.dumps(document).encode()
    panel.path.write_bytes(original)
    panel.window.config_dict=deepcopy(document)
    for name in ('protocols.imaging','protocols.injection','protocols.pressure_temperature'):
        shown=panel.factory.get_value(panel.window.config_dict,name)
        if name.endswith('.imaging'):
            key,value=panel.fields[name]['rows'][0]
            value.setText(repr(next(iter(shown.values()))))
        else:
            unwrap_composite_widget(panel.fields[name]).setText(str(shown))
    actual=ProtocolsConfig().load(panel.path)
    import ast
    shown=ast.literal_eval(panel.fields['protocols.imaging']['rows'][0][1].text())
    assert (Path(shown[0]),shown[1])==next(iter(actual.imaging.values()))
    injection=ast.literal_eval(unwrap_composite_widget(panel.fields['protocols.injection']).text())
    assert (Path(injection[0]),injection[1])==actual.injection
    panel.factory.save_settings()
    assert panel.path.read_bytes()==original


def test_alias_save_respects_a_newer_explicit_editor_sync_guard(panel, monkeypatch):
    """A custom editor must not be bypassed by the earlier alias save closure."""
    field=unwrap_composite_widget(panel.fields['protocols.injection'])
    field.setText(str(panel.new/'changed.csv'))
    calls=[]
    monkeypatch.setattr(panel.factory,'_sync_settings_inputs_to_config_dict',lambda:calls.append(True))
    panel.window.config_dict['pending_other_setting']='retained'
    panel.factory.save_settings()
    saved=tomllib.loads(panel.path.read_text())
    assert calls
    assert saved['protocols']['injection']==panel.document['protocols']['injection']
    assert saved['pending_other_setting']=='retained'
