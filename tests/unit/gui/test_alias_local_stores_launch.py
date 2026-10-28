"""Chosen local map travels only in a fake process argv, never shared metadata.

Only synthetic local job metadata is written. No inventory query, worker,
remote connection, scientific executor or payload is called.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QWidget

from darsia.fluidflower import job_filesystem, jobs_panel as ui
from darsia.fluidflower import native_jobs_workspace as owner
from darsia.fluidflower import remote_worker_launch as remote
from darsia.fluidflower.pipeline import job_queue as jobs
from .test_jobs_panel import app
from .test_native_jobs_owner import local_registry


def alias_job(tmp_path):
    root=tmp_path/"shared_job"
    facts=dict(format="darsia-alias-job-v1",format_version=1,execution_ready=True,
               runtime_support="darsia-role-io-runtime-v1")
    jobs.write_immutable(root/"job.json",facts)
    return root,facts


class FakeProcess:
    ProcessChannelMode=ui.QProcess.ProcessChannelMode
    instances=[]
    starts=True
    def __init__(self):
        self.__class__.instances.append(self)
    def setProcessEnvironment(self,value): self.environment=value
    def setProgram(self,value): self.program=value
    def setArguments(self,value): self.arguments=value
    def setWorkingDirectory(self,value): self.cwd=value
    def setProcessChannelMode(self,value): self.channel=value
    def setStandardOutputFile(self,value): self.console=value
    def startDetached(self): return self.starts,101 if self.starts else 0
    def errorString(self): return "literal synthetic launch failure"


@pytest.mark.parametrize("started",[True,False])
def test_explicit_local_cli_uses_chosen_map_without_persisting_it(
    app,tmp_path,monkeypatch,started
):
    root,_=alias_job(tmp_path)
    chosen=tmp_path/"machine_private"/"Chosen stores ü.toml"
    # This test checks argv/privacy, not store-policy qualification or a worker.
    from darsia.fluidflower import store_map
    from darsia_cache import adapters
    import os

    selected_map = object()
    metadata_calls = []
    environment_calls = []
    declared_environment = {"TEST_ONLY_SELECTED_WORKER_ENV": "chosen-map"}

    def selected_metadata(path):
        assert Path(path) == chosen
        metadata_calls.append(Path(path))
        return selected_map

    def selected_environment(stores, *, machine, series_id, explicit_stores,
                             parent_environment):
        assert stores is selected_map
        assert machine == "Main" and series_id is None and explicit_stores is True
        assert parent_environment is os.environ
        environment_calls.append(dict(declared_environment))
        return dict(declared_environment)

    monkeypatch.setattr(store_map, "read_store_map", selected_metadata)
    monkeypatch.setattr(adapters, "independent_worker_environment", selected_environment)
    monkeypatch.setattr(FakeProcess,"instances",[])
    monkeypatch.setattr(FakeProcess,"starts",started)
    monkeypatch.setattr(ui,"QProcess",FakeProcess)
    # Positive SMB inventory is a pure unit fixture, not a real OS probe.
    checks=[]
    def admitted(path):
        checks.append(Path(path))
        return job_filesystem.SharedFolderAdmission(True,"smb","fixture","synthetic SMB fact")
    monkeypatch.setattr(job_filesystem,"inspect_shared_folder",admitted)
    folder=root/"ui/launches/explicit"
    kwargs=dict(backend="cpu",workers=2,machine="Main",python="explicit-python",
                cwd=tmp_path/"checkout",launch_directory=folder,stores=chosen)
    if started:
        record=ui.launch_local_worker(root,**kwargs)
    else:
        with pytest.raises(RuntimeError,match="literal synthetic launch failure"):
            ui.launch_local_worker(root,**kwargs)
        record=jobs.read_json(folder/"launch.json")
    process=FakeProcess.instances[-1]
    assert metadata_calls == [chosen]
    assert environment_calls == [declared_environment]
    assert process.environment.keys() == list(declared_environment)
    assert process.environment.value("TEST_ONLY_SELECTED_WORKER_ENV") == "chosen-map"
    index=process.arguments.index("--stores")
    assert process.arguments[index+1]==str(chosen)
    assert checks==[root]
    assert record==jobs.read_json(folder/"launch.json")
    raw=(folder/"launch.json").read_text(encoding="utf-8")
    assert str(chosen) not in raw and "Chosen stores" not in raw
    assert "command" not in record and "cwd" not in record and "job_root" not in record
    assert record["console"]=="ui/launches/explicit/console.log"
    assert record["started"] is started
    assert record["worker_liveness_verified"] is False
    assert record["numerical_progress_claimed"] is False
    assert record["error"]==("" if started else "literal synthetic launch failure")


def test_panel_reads_map_at_explicit_start_and_retains_unknown_launch_reservation(
    app,tmp_path,monkeypatch
):
    root,facts=alias_job(tmp_path)
    selected=[tmp_path/"old.toml"]
    calls=[]
    panel=ui.JobsPanel(local_machine="Main",local_stores_path_provider=lambda:selected[0],
        local_launcher=lambda *args,**kwargs:(calls.append((args,kwargs)) or {"started":True}))
    panel.refresh=lambda:None
    try:
        panel.current=dict(root=str(root),job_facts=facts,errors=[],launch_requests=[])
        # Changing the visible settings path is passive and does not call Start.
        selected[0]=tmp_path/"new chosen.toml"
        assert not calls
        panel.start_local_workers()
        assert len(calls)==1 and calls[0][1]["stores"]==selected[0]
        request=panel._session_requests[-1]
        raw=Path(request["source"]["path"]).read_text(encoding="utf-8")
        assert "new chosen.toml" not in raw
        panel.start_local_workers()
        assert len(calls)==1, "The existing pending reservation still blocks a second service"
    finally:
        assert panel.can_close()
        panel.close();panel.deleteLater();app.processEvents()


def test_native_owner_provider_tracks_actual_settings_panel_path(app,tmp_path,local_registry):
    class StorePanel(QWidget):
        def __init__(self,parent):
            super().__init__(parent);self.path=tmp_path/"first.toml"
        def set_context(self,*args): pass
        def can_close(self): return True
    widget=owner.NativeJobsWorkspace(registry=local_registry,local_machine="Main",
                                    store_panel_factory=StorePanel)
    try:
        provider=widget.panel.local_stores_path_provider
        assert provider()==widget.stores.path
        widget.stores.path=tmp_path/"second ü.toml"
        assert provider()==widget.stores.path
        assert not local_registry.root.exists(), "Reading selected settings must not register a job"
        assert not widget.panel._session_requests
    finally:
        assert widget.can_close()
        widget.close();widget.deleteLater();app.processEvents()


def test_remote_worker_cli_uses_remote_default_not_the_local_selected_map():
    target=remote.RemoteWorkerTarget(machine="Second",ssh_alias="second",
        python=r"C:\existing\python.exe",checkout=r"C:\existing\repo",
        ssh_executable=r"C:\Windows\System32\OpenSSH\ssh.exe",default_shell="windows-powershell")
    launch=remote.build_remote_worker_launch(r"\\NAS\share\jobs\explicit",target=target,
                                             backend="cpu",workers=1)
    assert "--stores" not in launch.worker_argv
    assert "main_private_stores.toml" not in launch.remote_command
    assert launch.worker_argv[0]==target.python


def test_missing_selected_map_fails_before_reserving_or_launching(app,tmp_path):
    root,facts=alias_job(tmp_path)
    calls=[]
    panel=ui.JobsPanel(local_machine="Main",local_stores_path_provider=lambda:None,
                      local_launcher=lambda *args,**kwargs:calls.append((args,kwargs)))
    panel.refresh=lambda:None
    try:
        panel.current=dict(root=str(root),job_facts=facts,errors=[],launch_requests=[])
        panel.start_local_workers()
        assert not calls and not panel._session_requests
        assert not (root/"ui/launches").exists()
    finally:
        assert panel.can_close()
        panel.close();panel.deleteLater();app.processEvents()


@pytest.mark.parametrize("missing", ["map", "policy"])
def test_real_missing_map_or_policy_refuses_before_local_process(
    app, tmp_path, monkeypatch, missing
):
    """The positive argv double must not relax the actual launch admission."""
    from darsia.fluidflower import store_map
    from darsia_cache.policy import CachePolicyError

    root, _ = alias_job(tmp_path)
    chosen = tmp_path / "selected-stores.toml"
    if missing == "policy":
        chosen.write_text(
            '[machine]\nid = "Main"\n[stores.FF_AC]\n'
            + 'series = ' + json.dumps(str(tmp_path / "series")) + '\n'
            + 'results = ' + json.dumps(str(tmp_path / "results")) + '\n',
            encoding="utf-8")
    monkeypatch.setattr(FakeProcess, "instances", [])
    monkeypatch.setattr(ui, "QProcess", FakeProcess)
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path:
        job_filesystem.SharedFolderAdmission(True, "smb", "fixture", "synthetic SMB fact"))
    folder = root / "ui/launches/refused"
    expected = (OSError, store_map.StoreMapError) if missing == "map" else CachePolicyError
    with pytest.raises(expected):
        ui.launch_local_worker(root, backend="cpu", workers=1, machine="Main",
            python="not-executed", cwd=tmp_path / "checkout",
            launch_directory=folder, stores=chosen)
    assert not FakeProcess.instances
    assert not folder.exists()
