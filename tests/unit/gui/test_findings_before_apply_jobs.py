"""Current findings guard new Setup metadata without controlling accepted jobs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from darsia.fluidflower.series_setup_jobs_ui import SeriesSetupJobsController


class _Workspace(QWidget):
    """Expose the currently loaded series without reading its files."""

    reference_changed = Signal(str)

    def __init__(self, window, root):
        """Retain only the loaded context used by the Setup presentation."""
        super().__init__(window)
        self.project = SimpleNamespace(root=root / "geometry")
        self._loaded_root = root
        self.series_id = self._loaded_series = "campaign"


class _Findings(QObject):
    """Supply the owning controller's exact current-selection reason API."""

    findings_changed = Signal()

    def __init__(self, view, workspace):
        """Track synthetic pending findings without any scientific preparation."""
        super().__init__(view)
        self.view, self.workspace = view, workspace
        self.pending, self.checks = set(), []

    def key(self):
        """Bind this injected reason to the currently selected series and map."""
        return str(self.workspace._loaded_root), self.view.map_version

    def findings_block_reason(self):
        """Return an unresolved reason only for the exact current selection."""
        self.checks.append(self.key())
        return "The current transfer has unresolved findings." if self.key() in self.pending else ""


class _View(QWidget):
    """Use the existing action and busy signals with an injected findings owner."""

    busy_changed = Signal(bool)

    def __init__(self, workspace):
        """Create the existing action before its Setup controller is attached."""
        super().__init__(workspace)
        self.map_version = "map-a"
        self.apply_build_button = QPushButton("Prepare Apply and Build job")
        self.apply_build_reason = QLabel("No Setup context is available")
        self.adoption_controller = _Findings(self, workspace)
        layout = QVBoxLayout(self)
        layout.addWidget(self.apply_build_button)
        layout.addWidget(self.apply_build_reason)

    def busy(self):
        """Leave readiness to the controller while no other operation is active."""
        return False


@pytest.fixture
def owner(tmp_path, qtbot):
    """Provide actual Qt choices and forbid every metadata or worker dispatch."""
    window = QWidget()
    qtbot.addWidget(window)
    workspace = _Workspace(window, tmp_path / "series-a")
    view = _View(workspace)
    effects, contexts = [], []

    def forbidden(*args, **kwargs):
        effects.append((args, kwargs))
        pytest.fail("Unresolved findings must not dispatch or alter a job")

    target, backend = QComboBox(window), QComboBox(window)
    target.addItem("Local target", dict(machine="fixture", local=True))
    backend.addItem("CPU", "cpu")
    accepted = dict(root=str(tmp_path / "accepted-job"), state="Running",
        workers=[dict(state="Running")], launch_requests=[dict(state="Accepted")])
    jobs = QObject(window)
    jobs.series_root, jobs.series_id = workspace._loaded_root, "campaign"
    jobs.new_analysis = SimpleNamespace(form=SimpleNamespace(job_edit=QLineEdit(window)))
    jobs.panel = SimpleNamespace(target_combo=target, backend_combo=backend,
        _start_call=forbidden, _error=forbidden, current=accepted, snapshots=[accepted])
    jobs.stores = SimpleNamespace(check_button=QPushButton(window), settings_button=QPushButton(window))
    jobs.registry, jobs.status_label = SimpleNamespace(register_job=forbidden), QLabel(window)
    window.jobs_workspace = jobs
    window.show_jobs = forbidden

    def context(current):
        contexts.append((current.project, current._loaded_root))
        return dict(project=current.project, series_root=current._loaded_root,
            job_root=tmp_path / "new-setup-job", target_machine="fixture", commit="a" * 40,
            environment_lock=dict(path="requirements-worker.lock", sha256="b" * 64))

    return SimpleNamespace(window=window, workspace=workspace, view=view, jobs=jobs,
        effects=effects, contexts=contexts, context=context, forbidden=forbidden,
        findings=view.adoption_controller, output=tmp_path / "new-setup-job")


def test_direct_prepare_blocks_findings_before_context_or_job_writes(owner):
    """A direct slot call cannot bypass disabled readiness using an old revision."""
    owner.findings.pending.add(owner.findings.key())
    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=owner.context, preparer=owner.forbidden)
    assert not owner.view.apply_build_button.isEnabled()
    owner.view.apply_build_button.setEnabled(True)
    controller.prepare()
    assert not owner.view.apply_build_button.isEnabled()
    assert owner.view.apply_build_reason.text() == (
        "Review gaps and findings before preparing Apply and Build. "
        "The current transfer has unresolved findings.")
    assert owner.contexts == owner.effects == []
    assert controller._context is None and controller._thread is None
    assert not owner.output.exists()


def test_map_and_series_findings_changes_refresh_current_preparation(owner):
    """The exact view's signal rechecks both new map and new series selections."""
    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=owner.context, preparer=owner.forbidden)
    assert owner.view.apply_build_button.isEnabled()
    owner.view.map_version = "map-b"
    owner.findings.pending.add(owner.findings.key())
    owner.findings.findings_changed.emit()
    assert not owner.view.apply_build_button.isEnabled()
    assert controller._context is None
    owner.findings.pending.clear()
    owner.findings.findings_changed.emit()
    assert owner.view.apply_build_button.isEnabled()

    root = owner.workspace._loaded_root.parent / "series-b"
    owner.workspace._loaded_root = owner.jobs.series_root = root
    owner.workspace.project = SimpleNamespace(root=root / "geometry")
    owner.workspace.series_id = owner.workspace._loaded_series = owner.jobs.series_id = "other-campaign"
    owner.findings.pending.add(owner.findings.key())
    owner.findings.findings_changed.emit()
    controller.prepare()
    assert not owner.view.apply_build_button.isEnabled()
    assert owner.findings.checks[-1] == (str(root), "map-b")
    owner.findings.pending.clear()
    owner.findings.findings_changed.emit()
    assert owner.view.apply_build_button.isEnabled()
    assert controller._context["project"] is owner.workspace.project
    assert Path(controller._context["series_root"]) == root
    assert owner.effects == [] and not owner.output.exists()


def test_findings_appearing_during_context_capture_block_dispatch(owner):
    """A newly changed reason is checked again immediately before the call runner."""
    def context(workspace):
        captured = owner.context(workspace)
        if len(owner.contexts) > 1:
            owner.findings.pending.add(owner.findings.key())
        return captured

    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=context, preparer=owner.forbidden)
    assert owner.view.apply_build_button.isEnabled()
    controller.prepare()
    assert len(owner.contexts) == 2
    assert not owner.view.apply_build_button.isEnabled()
    assert "Review gaps and findings" in owner.view.apply_build_reason.text()
    assert controller._thread is None and controller._context is None
    assert owner.effects == [] and not owner.output.exists()


def test_new_findings_leave_an_accepted_job_and_its_workers_unchanged(owner):
    """The new-action guard neither cancels nor modifies an already accepted job."""
    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=owner.context, preparer=owner.forbidden)
    accepted = owner.jobs.panel.current
    before = deepcopy(accepted)
    controller._outcome = "Setup job metadata created previously."
    owner.findings.pending.add(owner.findings.key())
    owner.findings.findings_changed.emit()
    controller.prepare()
    assert owner.jobs.panel.current is accepted
    assert owner.jobs.panel.snapshots == [before] and accepted == before
    assert owner.jobs.status_label.text() == ""
    assert controller._outcome is None and controller._thread is None
    assert owner.effects == [] and not owner.output.exists()
