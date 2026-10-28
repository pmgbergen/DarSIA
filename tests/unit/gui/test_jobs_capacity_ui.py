"""Explicit asynchronous expected-space display; no actual OS/job/SCI operations."""
from types import SimpleNamespace
import threading
import pytest
from darsia.fluidflower import jobs_panel as ui,store_map
from darsia.fluidflower.pipeline import capacity


def panel(qtbot,tmp_path):
    value=ui.JobsPanel(local_stores_path_provider=lambda:tmp_path/"stores.toml")
    qtbot.addWidget(value)
    value.current=dict(root=str(tmp_path/"shared-job"),job_facts=dict(format="darsia-alias-job-v1"))
    return value


def test_passive_construction_never_checks_capacity(qtbot,tmp_path,monkeypatch):
    monkeypatch.setattr(capacity,"report_job_root",lambda *a:(_ for _ in ()).throw(AssertionError("Passive probe")))
    value=panel(qtbot,tmp_path)
    assert value._capacity_thread is None and "not checked" in value.capacity_label.text()
    assert value.can_close()


@pytest.mark.parametrize("state,enabled,feedback",[
    ("no-selection",False,"Select a job"),
    ("no-map",False,"local storage settings"),
    ("legacy",False,"Choose a job created from a saved execution plan"),
    ("alias",True,"Check this job's expected space"),
])
def test_passive_capacity_action_explains_prerequisites_without_probes(
    qtbot,tmp_path,monkeypatch,state,enabled,feedback
):
    calls=[]
    def unexpected(*args,**kwargs):
        calls.append((args,kwargs))
        raise AssertionError("Passive capacity display must not inspect storage")
    monkeypatch.setattr(store_map,"read_store_map",unexpected)
    monkeypatch.setattr(capacity,"report_job_root",unexpected)
    value=panel(qtbot,tmp_path)
    value.local_stores_path_provider=None if state=="no-map" else unexpected
    data=None if state=="no-selection" else dict(
        root=str(tmp_path/"shared-job"),
        job_facts=dict(format="legacy-job" if state=="legacy" else "darsia-alias-job-v1"))
    value._show_snapshot(data)
    assert value.capacity_button.isEnabled() is enabled
    assert feedback in value.capacity_button.toolTip()
    assert "not checked" in value.capacity_label.text()
    assert value._capacity_thread is None and value.can_close() and not calls


def test_explicit_background_report_shows_needed_free_and_estimation_scope(qtbot,tmp_path,monkeypatch):
    observed=[]
    monkeypatch.setattr(store_map,"read_store_map",lambda *a,**k:SimpleNamespace())
    def report(*args):
        observed.append(threading.get_ident())
        return dict(message="Expected needed 1250 bytes / free 1249 bytes",sufficient=False)
    monkeypatch.setattr(capacity,"report_job_root",report)
    value=panel(qtbot,tmp_path);value.check_capacity()
    qtbot.waitUntil(lambda:value._capacity_thread is None)
    assert observed and observed[0]!=threading.get_ident()
    assert "1250" in value.capacity_label.text() and "1249" in value.capacity_label.text()
    assert "insufficient" in value.capacity_label.text() and "preserved native" in value.capacity_label.text()
    assert "estimate, not coordinated across hosts" in value.capacity_label.text()
    assert "cache admission is separate" in value.capacity_label.text()
    assert value.can_close()


def test_missing_profile_is_visible_not_zero_capacity_or_success(qtbot,tmp_path,monkeypatch):
    monkeypatch.setattr(store_map,"read_store_map",lambda *a,**k:SimpleNamespace())
    def refuse(*args):raise ValueError("An explicit applicable measured profile is missing")
    monkeypatch.setattr(capacity,"report_job_root",refuse)
    value=panel(qtbot,tmp_path);value.check_capacity()
    qtbot.waitUntil(lambda:value._capacity_thread is None)
    assert "unavailable" in value.capacity_label.text() and "profile is missing" in value.capacity_label.text()
    assert "enough" not in value.capacity_label.text()


def test_stale_context_does_not_publish_capacity_for_another_job(qtbot,tmp_path):
    value=panel(qtbot,tmp_path)
    before=value.capacity_label.text()
    value._capacity_ready(dict(generation=value._generation-1,root=value.current["root"],
        result=dict(message="WRONG SERIES",sufficient=True)))
    assert value.capacity_label.text()==before
