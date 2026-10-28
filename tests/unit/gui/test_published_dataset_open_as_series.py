"""A published dataset added on Start opens read-only, with New analysis pre-filled (offscreen Qt).

The window, the series registry, the dataset and the launcher page are real; the
reproduction runner is the launcher tests' stand-in, and Studio and geometry
pages are empty widgets as in the sidebar tests.
"""

import json
import os
from pathlib import Path
import re
import shutil
import zipfile

import pytest
from PySide6.QtCore import QCoreApplication, QSettings

from darsia.fluidflower.published_dataset import builder, dataset_series, read_only, scan
from darsia.fluidflower.published_dataset.comparison import IDENTICAL
from tests.unit.published_dataset.synthetic import make_series
from tests.unit.published_dataset.test_reproduce import INSTALLED, FakeRunner


def snapshot(folder):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in Path(folder).rglob("*")}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A built R record, unpacked where a user keeps downloads."""
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),))
    series = make_series(tmp_path / "producer-site")
    report = builder.build_record(builder.BuildRequest(series.series, series.job, "R", "S50", tmp_path / "record",
                                                       deny=(), fingerprint=False, peak_memory_bytes=2 * 1024**3))
    assert report.built, report.text()
    folder = tmp_path / "downloads" / "dataset"
    shutil.copytree(tmp_path / "record", folder)
    for name in ("series.zip", "expected.zip"):
        with zipfile.ZipFile(folder / name) as archive:
            archive.extractall(folder)
    return folder.resolve()


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import native_darsia_app as native
    from darsia.fluidflower.published_dataset import dataset_analysis
    from darsia.fluidflower.published_dataset.reproduce_window import ReproduceWindow
    from darsia.gui.ui.process_runner import ProcessRunner
    from darsia.gui.ui.settings import SettingsFactory

    old_format = QSettings.defaultFormat()
    old_organization = QCoreApplication.organizationName()
    old_application = QCoreApplication.applicationName()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "qsettings"))
    QCoreApplication.setOrganizationName("DarSIA-dataset-tests")
    QCoreApplication.setApplicationName(tmp_path.name)
    # Only this test's registry lists series.
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "settings" / "series_registry.toml"))
    for name in list(os.environ):
        if re.fullmatch(r"DARSIA_[A-Z][A-Z0-9_]*_PROJECT_ROOT|DARSIA_FLUIDFLOWER_DATA_ROOT", name):
            monkeypatch.delenv(name)

    class EmptyProjects:
        def entries(self):
            return []

    monkeypatch.setattr(native, "ProjectRegistry", EmptyProjects)
    monkeypatch.setattr(native.PhaseModelWorkspace, "ensure_studio", lambda self: self.studio_placeholder)
    monkeypatch.setattr(native.SeriesGeometryWorkspace, "ensure_page", lambda self: self)
    started, saved = [], []
    monkeypatch.setattr(ProcessRunner, "start_workflow_process", lambda self, *args, **kwargs: started.append(args))
    monkeypatch.setattr(SettingsFactory, "save_settings", lambda self, *args, **kwargs: saved.append(args))
    monkeypatch.setattr(dataset_analysis, "ReproduceWindow", lambda: ReproduceWindow(
        runner_factory=FakeRunner, installed=INSTALLED, fingerprint={}, machine={"cpu": "fixture"},
        inventory={"cpu": {"physical_cores": 4, "logical_cores": 8}, "memory": {"available_bytes": 16 * 1024**3}}))
    window = native.NativeFluidFlowerMainWindow()
    qtbot.addWidget(window)
    window.recorded = {"started": started, "saved": saved}
    window.show()
    qtbot.waitUntil(window.isVisible)
    try:
        yield window
    finally:
        page = window.dataset_analysis_page
        if page is not None:
            qtbot.waitUntil(lambda: not page.busy(), timeout=20000)
        window.close()
        for root in read_only.protected_roots():
            read_only.unprotect(root)
        QCoreApplication.setOrganizationName(old_organization)
        QCoreApplication.setApplicationName(old_application)
        QSettings.setDefaultFormat(old_format)


def test_a_dataset_added_on_start_opens_read_only_with_new_analysis_prefilled(window, dataset, tmp_path, qtbot,
                                                                              monkeypatch):
    from darsia.fluidflower import native_darsia_app as native

    results = tmp_path / "my results"
    answers = iter([str(dataset), str(dataset / "config"), str(results)])
    monkeypatch.setattr(native.QFileDialog, "getExistingDirectory", lambda *args: next(answers))
    warnings = []
    monkeypatch.setattr(native.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    before = snapshot(dataset)

    window._add_existing_series()

    # A results location outside the dataset is required; one inside is refused.
    assert warnings == [("Choose another results location", "The results location must lie outside the dataset folder")]
    assert dataset_series.dataset_results(dataset) == results.resolve()
    assert os.path.normcase(str(dataset)) in read_only.protected_roots()
    assert window.dataset_series_root() == dataset
    page = window.dataset_analysis_page
    assert window.content_stack.currentWidget() is page
    assert window.sidebar.get_selected() == ("jobs", "new_analysis")
    launcher = page.launcher
    assert Path(launcher.dataset_edit.text()) == dataset
    assert Path(launcher.results_edit.text()) == results.resolve() / "analysis-1"
    qtbot.waitUntil(lambda: launcher.reproduction is not None and not launcher.busy(), timeout=20000)
    manifest = launcher.reproduction.check.manifest
    facts = page.facts_label.text()
    assert "Frames: 6 in 2 run(s), as published" in facts and "Stored layer: 1 mm cells" in facts
    assert "Storage profile: full; precision: float32" in facts and "Mass calibration: SHA-256" in facts
    assert page.differences_label.text() == "Changes from the published plan: none."
    assert launcher.threads_spin.value() == manifest["machine"]["threads"]
    assert "measured on the reference machine" in launcher.advice_label.text()

    # Every page that would change the series stays closed; Start and the job list remain.
    for category, section in window.sidebar._sections.items():
        for display_id in list(section._rows):
            action, item_id = window.sidebar._logical_selection(category, display_id)
            window._on_sidebar_selection(action, item_id)
            if action != "start" and (action, item_id) != ("jobs", "jobs"):
                assert window.content_stack.currentWidget() is page, (action, item_id)
                assert not window.toolbar_builder.play_action.isEnabled()
    window._select_row(("jobs", "new_analysis"))
    window.settings_factory.save_settings()
    window.process_runner.start_workflow_process(["python", "-c", "pass"])
    window.open_wizard_for("preprocessing")
    window._open_preprocessing_for_selected_run()
    assert window.recorded == {"started": [], "saved": []}
    assert snapshot(dataset) == before

    # The new analysis runs the published plan into the results location.
    launcher.workers_spin.setValue(1)
    launcher.start()
    qtbot.waitUntil(lambda: launcher.result is not None or "Stopped" in launcher.status_label.text(), timeout=20000)
    assert launcher.status_label.text() == f"Verdict: {IDENTICAL}."
    report = json.loads((results / "analysis-1" / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["differences_from_published_plan"] == [] and report["reproduction"] is True

    # A changed setting is named; opening the page again offers a fresh folder.
    launcher.threads_spin.setValue(manifest["machine"]["threads"] + 1)
    assert "Threads per worker" in page.differences_label.text()
    window._on_sidebar_selection("geometry_facies", "rig_frame")
    assert Path(launcher.results_edit.text()) == results.resolve() / "analysis-2"
    assert snapshot(dataset) == before
    assert not any(path.name == "published_datasets.toml" for path in dataset.rglob("*"))

    # A registered dataset is protected again as soon as DarSIA starts.
    read_only.unprotect(dataset)
    again = native.NativeFluidFlowerMainWindow()
    qtbot.addWidget(again)
    assert os.path.normcase(str(dataset)) in read_only.protected_roots()
    again.close()


def test_a_packed_dataset_is_refused_with_how_to_unpack_it(window, dataset, tmp_path, monkeypatch):
    from darsia.fluidflower import native_darsia_app as native

    packed = tmp_path / "packed"
    packed.mkdir()
    for name in ("dataset_manifest.json", "series.zip", "expected.zip"):
        shutil.copyfile(dataset / name, packed / name)
    monkeypatch.setattr(native.QFileDialog, "getExistingDirectory", lambda *args: str(packed))
    warnings = []
    monkeypatch.setattr(native.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    window._add_existing_series()
    assert warnings == [("Dataset not unpacked",
                         "Unpack series.zip and expected.zip into the dataset folder, then add it again.")]
    assert dataset_series.registered_datasets() == [] and window.dataset_analysis_page is None
