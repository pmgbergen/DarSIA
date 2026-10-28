"""Actual GUI launch methods and Qt signal; all process/IO/refit work is fixture-only."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from darsia.fluidflower.jobs_panel import JobsPanel
from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow


@pytest.fixture
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def launch_fixture(application, tmp_path, monkeypatch):
    events, errors = [], []
    phase_root = str(tmp_path / "phase-one")
    series = str(tmp_path / "series")
    order = ["ac17", "ac18", "ac23"]
    state = dict(accept=True, pipeline="native-phase-statistics", busy=False)

    def start_refit(root, runs):
        events.append(("refit", root, runs))
        if state["busy"] or refit._automatic is not None:
            return False
        refit._automatic = dict(generation=refit._generation, required_runs=list(runs),
                               arguments={"phase_job": root})
        return True

    refit = SimpleNamespace(series_root=series, start_after_phase_one=start_refit,
                            _automatic=None, _generation=1)

    def launch(root, **kwargs):
        events.append(("launcher", root, kwargs))
        if not state["accept"]:
            raise RuntimeError("fixture launch refused")
        return {"started": True}

    panel = JobsPanel(local_machine="Main", local_launcher=launch)
    panel.series_root = series
    panel.current = dict(root=phase_root, errors=[], job_facts=dict(
        pipeline=state["pipeline"], expected_runs=order))
    owner = SimpleNamespace(mass_calibration_display=SimpleNamespace(refit_panel=refit),
                            jobs_workspace=SimpleNamespace(panel=panel))
    monkeypatch.setattr(panel, "_start_allowed", lambda machine: True)
    monkeypatch.setattr(panel, "_begin_request", lambda machine, kind:
                        {"launch_directory": str(tmp_path / "fake-launch")})
    monkeypatch.setattr(panel, "_launch_failed", lambda request, text: errors.append(text))
    monkeypatch.setattr(panel, "_error", errors.append)
    monkeypatch.setattr(panel, "refresh", lambda: None)
    panel.phase_one_launch_accepted.connect(
        lambda event: NativeFluidFlowerMainWindow._phase_one_launch_accepted(owner, event))
    assert Path(__import__(JobsPanel.__module__, fromlist=["__file__"]).__file__).resolve().is_relative_to(
        Path(__file__).resolve().parents[3] / "src")
    yield SimpleNamespace(panel=panel, owner=owner, refit=refit, state=state,
                          events=events, errors=errors, order=order,
                          phase_root=phase_root, series=series)
    panel.close()
    panel.deleteLater()
    application.processEvents()


def test_accepted_original_cpu_p1_hands_exact_root_and_order_once(launch_fixture):
    f = launch_fixture
    before = deepcopy(f.panel.current)
    f.panel.start_local_workers()
    assert [row[0] for row in f.events] == ["launcher", "refit"]
    assert f.events[1] == ("refit", str(Path(f.phase_root).resolve()), f.order)
    assert f.panel.current == before
    assert not f.errors
    # Another service request for this same original P1 must not attach a second refit.
    f.panel.start_local_workers()
    assert [row[0] for row in f.events].count("refit") == 1


@pytest.mark.parametrize("case", ["failed", "phase2", "cuda", "denied"])
def test_unaccepted_or_other_launch_never_arms_refit(launch_fixture, monkeypatch, case):
    f = launch_fixture
    if case == "failed":
        f.state["accept"] = False
    elif case == "phase2":
        f.panel.current["job_facts"]["pipeline"] = "n1"
    elif case == "cuda":
        monkeypatch.setattr(f.panel.backend_combo, "currentData", lambda: "cuda")
    else:
        monkeypatch.setattr(f.panel, "_start_allowed", lambda machine: False)
    f.panel.start_local_workers()
    assert all(row[0] != "refit" for row in f.events)


@pytest.mark.parametrize("case", ["busy", "other_series"])
def test_attachment_failure_keeps_accepted_p1_visible(launch_fixture, case):
    f = launch_fixture
    if case == "busy":
        f.state["busy"] = True
    else:
        f.refit.series_root = f.series + "-different"
    f.panel.start_local_workers()
    assert len(f.panel._launch_records) == 1
    assert f.errors and "Phase map generation was requested" in f.errors[0]
    assert not getattr(f.owner, "_refit_phase_launches", set())


def test_accepted_event_retains_launch_time_identity(launch_fixture):
    f = launch_fixture
    original = f.panel.local_launcher

    def changes_visible_selection(root, **kwargs):
        result = original(root, **kwargs)
        f.panel.current = dict(root=f.phase_root + "-other", job_facts=dict(
            pipeline="n1", expected_runs=["ac99"]))
        return result

    f.panel.local_launcher = changes_visible_selection
    f.panel.start_local_workers()
    assert f.events[1] == ("refit", str(Path(f.phase_root).resolve()), f.order)


@pytest.mark.parametrize("case", ["cleared", "generation_changed"])
def test_dedup_expires_when_original_waiter_clears_or_changes_generation(launch_fixture, case):
    f = launch_fixture
    f.panel.start_local_workers()
    if case == "cleared":
        f.refit._automatic = None
    else:
        f.refit._generation += 1
    f.panel.start_local_workers()
    assert [row[0] for row in f.events].count("refit") == 2
    if case == "cleared":
        assert not f.errors
        assert f.refit._automatic["generation"] == f.refit._generation
    else:
        assert f.errors and "Phase map generation was requested" in f.errors[0]
