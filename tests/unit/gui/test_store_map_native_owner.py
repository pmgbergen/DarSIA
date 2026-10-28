"""Owner context/close contract only; no map, NAS, job or resource inspection."""
from types import SimpleNamespace

from PySide6.QtWidgets import QWidget
from darsia.fluidflower.native_jobs_workspace import NativeJobsWorkspace


class FakeStores(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.contexts=[]
        self.idle=True
    def set_context(self,root,series_id):
        self.contexts.append((root,series_id))
    def can_close(self):
        return self.idle


def test_jobs_owner_forwards_manifest_identity_without_running_availability_and_respects_close(qtbot,tmp_path):
    series=tmp_path/"series";series.mkdir()
    (series/"series.toml").write_bytes(b'schema_version=1\nseries_id="Actual"\n')
    registry=SimpleNamespace(known_jobs=lambda:[], jobs=lambda *_:[], plans=lambda *_:[])
    workspace=NativeJobsWorkspace(registry=registry,local_machine="Main",store_panel_factory=FakeStores)
    qtbot.addWidget(workspace)
    assert workspace.stores.contexts == [(None,None)]
    workspace.set_series("display alias",series)
    assert workspace.stores.contexts[-1] == (series,"Actual")
    workspace.stores.idle=False
    assert not workspace.can_close()
    workspace.stores.idle=True
    workspace._clear_context()
    assert workspace.stores.contexts[-1] == (None,None)
    assert workspace.can_close()
