"""The actual GUI waiter with finite job metadata and no numerical launch."""
import json
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from darsia.fluidflower import refit_job_panel as module


@pytest.fixture
def panel(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    view = module.RefitJobPanel()
    monkeypatch.setattr(view, "_update_actions", lambda *_args: None)
    view.series_root = str(tmp_path / "series")
    view.roles = {"fit": ["r1", "r2"]}
    view.rules = {"fixture": "finite rules only"}
    view.baseline = {"fixture": "finite baseline only", "sha256": "a" * 64}
    view.jobs_edit.setText(str(tmp_path / "results"))
    view.name_edit.setText("round-fit")
    phase = tmp_path / "phase"
    phase.mkdir()
    (phase / "job.json").write_text(json.dumps({"expected_runs": ["r1", "r2"]}), encoding="utf-8")
    yield view, phase
    view._phase_wait_timer.stop()
    view._timer.stop()
    view.close()
    view.deleteLater()
    app.processEvents()


def test_exact_phase_order_is_captured_and_partial_completion_stays_waiting(panel, monkeypatch):
    view, phase = panel
    monkeypatch.setattr(view, "_poll_phase_one", lambda: None)
    with pytest.raises(ValueError, match="run order"):
        view.start_after_phase_one(phase, ["r2", "r1"])
    assert view._automatic is None
    assert view.start_after_phase_one(phase, ["r1", "r2"])
    assert not view.start_after_phase_one(phase, ["r1", "r2"])
    view._phase_one_waited(["r1"])
    assert view._automatic["state"] == "waiting" and not view.phase_ready
    assert "1 of 2" in view.job_label.text()
    view._phase_one_waited(["r1", "r2"])
    assert view._automatic["state"] == "create" and view.phase_ready
    captured = []
    monkeypatch.setattr(view, "_call", lambda operation, callback: captured.append((operation, callback)))
    view._closed()
    assert view._automatic["state"] == "creating" and len(captured) == 1


def test_retirement_wait_prevents_duplicate_or_early_automatic_start(panel, monkeypatch):
    view, _ = panel
    view._automatic = dict(state="start_ready", generation=view._generation)
    window = SimpleNamespace(process_runner=SimpleNamespace(active=True),
        depth_workflow=SimpleNamespace(sequence=SimpleNamespace(running=False)),
        isEnabled=lambda: True)
    monkeypatch.setattr(view, "window", lambda: window)
    starts = []
    monkeypatch.setattr(view, "start", lambda **kwargs: starts.append(kwargs) or True)
    view._poll_phase_one()
    assert not starts and view._automatic["state"] == "start_ready"
    window.process_runner.active = False
    view._poll_phase_one()
    view._poll_phase_one()
    assert starts == [{"confirm": False}]
    assert view._automatic["state"] == "running"


def test_failed_phase_run_cannot_reach_the_optimizer(panel, monkeypatch):
    view, phase = panel
    monkeypatch.setattr(view, "_poll_phase_one", lambda: None)
    assert view.start_after_phase_one(phase, ["r1", "r2"])
    monkeypatch.setattr(module, "phase_job_readiness", lambda *args: (["r1"], []))
    (phase / "failed").mkdir()
    (phase / "failed" / "r2.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(view, "_call", lambda operation, callback: callback(operation()))
    with pytest.raises(ValueError, match="optimizer has not started"):
        module.RefitJobPanel._poll_phase_one(view)
    assert view._automatic["state"] == "waiting"


def test_overflow_selection_requires_sources_before_arming_and_is_captured_exactly(panel, monkeypatch):
    view, phase = panel
    monkeypatch.setattr(view, "_poll_phase_one", lambda: None)
    view.set_overflow_enabled(True)
    with pytest.raises(ValueError, match="reviewed overflow inputs"):
        view.start_after_phase_one(phase, ["r1", "r2"])
    assert view._automatic is None
    # Finite descriptor-routing fixture only; the real read_request has its
    # original source-byte tests separately. No numerical job is created.
    view.overflow_inputs = dict(path="declared-inputs.json", sha256="a" * 64, size_bytes=17)
    assert view.start_after_phase_one(phase, ["r1", "r2"])
    assert view._automatic["arguments"]["overflow"] == view.overflow_inputs
    view.overflow_inputs["sha256"] = "b" * 64
    assert view._automatic["arguments"]["overflow"]["sha256"] == "a" * 64


def test_overflow_selection_cannot_start_an_open_ordinary_job(panel):
    view, phase = panel
    view.set_overflow_enabled(True)
    view.job_root = phase.parent / view.name_edit.text()
    view.snapshot = dict(start=dict(blocker=""), local_variant_f=False)
    assert "different overflow choice" in view._start_blocker()
