"""Source-phase GUI fixtures; no original producer, science, worker or SSH calls."""
from pathlib import Path
import threading
from types import SimpleNamespace

from PySide6.QtGui import QCloseEvent

from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
from .test_native_jobs_owner import local_registry, workspace, series  # noqa: F401


def fill(page, tmp_path):
    for field, value in ((page.config_path, tmp_path / "original-comparison.toml"),
                         (page.metadata_folder, tmp_path / "fresh-metadata"),
                         (page.job_root, tmp_path / "fresh-job"),
                         (page.job_id, "comparison-fixture"),
                         (page.actor, "Reviewer"), (page.reason, "Compare the original saved mass")):
        field.setText(str(value))


def prepared(root):
    """UI callback fixture only; deliberately writes no queue or scientific data."""
    return dict(root=str(root), name="comparison-fixture", pipeline="preset-wasserstein",
        job=dict(path=str(root / "job.json"), sha256="a" * 64, size_bytes=1),
        comparison_count=3, requested_times=[1.0], reference_run="run-a")


def test_preparation_is_explicit_and_uses_captured_series_off_ui(workspace, qtbot, tmp_path, monkeypatch):
    owner, page = workspace, workspace.wasserstein
    a, b = series(tmp_path / "a", "A"), series(tmp_path / "b", "B")
    owner.set_series("displayed-alias", a)
    owner.show_view("wasserstein")
    calls, registrations = [], []
    entered, release = threading.Event(), threading.Event()
    gui_thread = threading.get_ident()

    def create(config, metadata, output, **kwargs):
        calls.append((config, metadata, output, kwargs, threading.get_ident()))
        entered.set()
        if not release.wait(10):
            raise RuntimeError("Private producer fixture was not released")
        return prepared(Path(output))

    page.preparer = create
    monkeypatch.setattr(owner.registry, "register_job",
        lambda root, series_id, job_root, **kwargs: registrations.append((root, series_id, job_root)))
    fill(page, tmp_path)
    assert not calls and owner.can_close()
    assert not page.show_jobs_button.isEnabled()
    page.prepare_button.click()
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert not page.can_close() and not owner.panel.can_close() and not owner.can_close()
        page.prepare()  # A second click cannot create another metadata call.
        owner.show_view("jobs")
        assert not owner.can_close()
        event, logs = QCloseEvent(), []
        NativeFluidFlowerMainWindow.closeEvent(SimpleNamespace(jobs_workspace=owner, print_log=logs.append), event)
        assert not event.isAccepted() and logs
        owner.set_series("B", b)
    finally:
        release.set()
        qtbot.waitUntil(owner.can_close, timeout=10000)
    assert len(calls) == 1 and calls[0][-1] != gui_thread
    assert calls[0][3]["series_root"] == a and calls[0][3]["series_id"] == "A"
    assert calls[0][3]["job_id"] == "comparison-fixture"
    assert calls[0][3]["actor"] == "Reviewer" and calls[0][3]["reason"]
    assert registrations == [(str(a), "A", str(tmp_path / "fresh-job"))]
    assert page.registration is None and not page.show_jobs_button.isEnabled()
    assert "earlier series A" in page.status.text()
    assert owner.panel._session_requests == [] and not (tmp_path / "fresh-job").exists()


def test_created_job_returns_to_original_controls_without_starting(workspace, qtbot, tmp_path, monkeypatch):
    owner, page = workspace, workspace.wasserstein
    root = series(tmp_path / "s", "literal")
    owner.set_series("literal", root)
    owner.show_view("wasserstein")
    controls = (owner.panel.target_combo, owner.panel.backend_combo, owner.panel.workers_spin)
    registrations = []
    monkeypatch.setattr(owner.registry, "register_job", lambda *args, **kwargs: registrations.append(args))
    active = []
    monkeypatch.setattr(owner.registry, "set_active", lambda *args: active.append(args))
    page.preparer = lambda config, metadata, output, **kwargs: prepared(Path(output))
    fill(page, tmp_path)
    page.prepare_button.click()
    qtbot.waitUntil(owner.can_close, timeout=10000)
    assert len(registrations) == 1 and page.show_jobs_button.isEnabled()
    assert "No worker started" in page.status.text()
    page.show_jobs_button.click()
    assert active == [(root, "literal", str(tmp_path / "fresh-job"))]
    assert owner.views.currentWidget() is owner.jobs_view
    assert controls == (owner.panel.target_combo, owner.panel.backend_combo, owner.panel.workers_spin)
    assert owner.panel._session_requests == []


def test_failed_producer_or_registration_does_not_offer_a_ready_job(workspace, qtbot, tmp_path, monkeypatch):
    owner, page = workspace, workspace.wasserstein
    owner.set_series("literal", series(tmp_path / "s", "literal"))
    fill(page, tmp_path)
    registrations = []
    monkeypatch.setattr(owner.registry, "register_job", lambda *args, **kwargs: registrations.append(args))

    def fail(*args, **kwargs):
        raise ValueError("Original mass source is unavailable")

    page.preparer = fail
    page.prepare_button.click()
    qtbot.waitUntil(owner.can_close, timeout=10000)
    assert not registrations and "Original mass source is unavailable" in page.status.text()
    assert page.registration is None and not page.show_jobs_button.isEnabled()
    page.preparer = lambda config, metadata, output, **kwargs: prepared(Path(output))
    monkeypatch.setattr(owner.registry, "register_job", fail)
    page.prepare_button.click()
    qtbot.waitUntil(owner.can_close, timeout=10000)
    assert "local registration failed" in page.status.text()
    assert page.registration is None and not page.show_jobs_button.isEnabled()
    assert owner.panel._session_requests == []


def test_folders_are_named_from_the_series_and_the_actor_is_filled(qtbot, tmp_path, monkeypatch):
    import re
    from darsia.fluidflower import series_paths
    from darsia.fluidflower.preset_wasserstein_preparation_ui import PresetWassersteinPreparationPage
    from darsia.fluidflower.series_geometry import default_actor

    page = PresetWassersteinPreparationPage(None)
    qtbot.addWidget(page)
    assert page.job_root.isReadOnly() and page.metadata_folder.isReadOnly()
    assert not page.job_root.text() and page.actor.text() == default_actor()
    folders = {"jobs": Path("//server/share/series/jobs"), "results": Path("//server/share/series/results")}

    def folder(root, key, **kwargs):
        if key not in folders:
            raise KeyError(key)
        return folders[key]

    monkeypatch.setattr(series_paths, "series_folder_path", folder)
    page.set_context(tmp_path, "S")
    job, metadata = page.job_root.text(), page.metadata_folder.text()
    parent, name = job.rsplit("\\", 1)
    assert parent == str(folders["jobs"]) and re.fullmatch("wasserstein-[0-9a-f]{32}", name)
    assert metadata == str(folders["results"] / "comparisons" / "inputs" / name)
    page.suggest_folders()
    assert page.job_root.text() == job  # The shown fresh folders are kept.
    page.suggest_folders(fresh=True)
    assert page.job_root.text() != job
    folders["comparison_inputs"] = Path("//server/share/series/comparison-inputs")
    page.suggest_folders(fresh=True)
    assert page.metadata_folder.text().startswith(str(folders["comparison_inputs"]))
    folders["jobs"] = tmp_path / "jobs"  # No shared Jobs folder: nothing is invented.
    page.suggest_folders()
    assert not page.job_root.text() and not page.metadata_folder.text()
    assert "Jobs > Shared job folder" in page.job_root.placeholderText()
    page.prepare()
    assert "Jobs > Shared job folder" in page.status.text()
