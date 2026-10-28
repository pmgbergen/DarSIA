"""Compact Jobs controls and resource pages on local Qt fixtures.

The job list fills the left pane; the selected job's header, toolbar and detail
tabs fill the right. Sources are collapsible. No scroll area sits inside
another. The same widgets and actions as before; real Qt, no job is started.
"""

import json

import pytest

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QScrollArea, QSplitter, QStyle, QStyleOptionComboBox,
    QStyleOptionSpinBox, QWidget)

from darsia.fluidflower.resource_advisor_panel import ResourceAdvisorPanel
from darsia.fluidflower.measurement_receipts import MeasurementRegistry
from darsia.fluidflower.store_map_ui import StoreAvailabilityPanel, StoreSettingsDialog
from darsia.fluidflower.ui.theme import apply_application_theme

from .test_native_jobs_owner import local_registry, snapshot, ui  # noqa: F401

UNCAPPED = 16777215  # QWIDGETSIZE_MAX

@pytest.fixture(autouse=True)
def native_layout_font(qapp):
    """Give Qt's offscreen backend the same readable font as Windows."""
    from pathlib import Path
    from PySide6.QtGui import QFont, QFontDatabase
    old = qapp.font()
    old_style = qapp.styleSheet()
    from darsia.fluidflower.ui.theme import apply_application_theme
    apply_application_theme(qapp)
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if font.is_file():
        QFontDatabase.addApplicationFont(str(font))
        qapp.setFont(QFont("Segoe UI", 10))
    yield
    qapp.setFont(old)
    qapp.setStyleSheet(old_style)



def _workspace(qtbot, local_registry):
    owner = ui.NativeJobsWorkspace(
        registry=local_registry,
        local_machine="local",
        reader=snapshot,
        inventory_provider=lambda: {},
        resource_factory=ResourceAdvisorPanel,
    )
    qtbot.addWidget(owner)
    return owner


def _ancestors(widget, stop):
    parent = widget.parentWidget()
    while parent is not None and parent is not stop:
        yield parent
        parent = parent.parentWidget()


def test_job_list_and_selected_job_have_separate_native_pages(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    panel, inline = owner.panel, owner.inline_layout
    assert inline.list_results.isAncestorOf(panel.job_table)
    assert inline.list_results.isAncestorOf(panel.create_button)
    assert not panel.isAncestorOf(panel.job_table)
    for widget in (panel.start_button, panel.pause_button, panel.resume_button,
                   panel.stop_button, panel.add_machine_button, panel.declare_stopped_button):
        assert inline.control_results.isAncestorOf(widget)
    assert owner.job_results.isAncestorOf(panel.detail_tabs)


def test_selected_job_has_separate_progress_and_resources_tabs(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    tabs = panel.detail_tabs
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["Progress", "Log", "Errors", "Files", "Resources"]
    assert tabs.widget(1) is panel.log_text and tabs.widget(2) is panel.error_text
    for widget in (panel.machine_table, panel.worker_table):
        assert tabs.widget(0) in _ancestors(widget, panel)
    assert tabs.widget(4) in _ancestors(panel.resource_panel, panel)
    assert not panel.resource_panel.findChildren(QScrollArea)
    for widget in (panel.capacity_button, panel.capacity_label):
        assert tabs.widget(3) in _ancestors(widget, panel)


def test_job_controls_use_compact_rows_and_fit_a_normal_window(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    owner.resize(1280, 900)
    owner.show()
    qtbot.waitUntil(owner.isVisible)
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    owner.resize(1280, 900)
    panel = owner.panel
    assert owner.width() == 1280
    assert owner.minimumSizeHint().width() <= 1280
    owner.inline_layout.control_section.set_expanded(True)
    assert panel.start_button.mapTo(owner.inline_layout.control_results, panel.start_button.rect().topLeft()).y() < panel.pause_button.mapTo(owner.inline_layout.control_results, panel.pause_button.rect().topLeft()).y()
    assert panel.pause_button.geometry().y() == panel.stop_button.geometry().y()
    assert panel.stop_button.text() == "Stop job after current frame"
    assert panel.resume_button.text() == "Resume job"
    assert owner.new_analysis.start_button.text() == "Start workers for this job"
    for widget in (panel.start_button, panel.pause_button, panel.resume_button,
                   panel.stop_button, panel.add_machine_button, panel.declare_stopped_button):
        assert widget.geometry().right() < widget.parentWidget().width()


def test_start_controls_keep_the_borrowable_target_row(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    row = panel.start_button.parentWidget().layout()
    controls = [panel.target_combo, panel.backend_combo, panel.workers_spin, panel.start_button,
                panel.add_machine_button]
    positions = []
    for widget in controls:
        found = [index for index in range(row.count()) for item in [row.itemAt(index)]
                 if item.layout() is not None and item.layout().indexOf(widget) >= 0]
        positions.append(found)
    assert all(found == positions[0] for found in positions) and positions[0]


def test_empty_refresh_has_visible_feedback(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    qtbot.mouseClick(panel.refresh_button, Qt.MouseButton.LeftButton)
    assert "No jobs registered" in panel.progress_label.text()


def test_resource_actions_explain_refresh_copy_and_apply(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    resources = owner.resources
    rows = [dict(machine="local", inventory=dict(
        cpu=dict(physical_cores=8), memory=dict(available_bytes=120 * 1024**3)), history=[])]
    resources.set_context({"precision": "float32"}, machines_provider=lambda: rows)
    owner.resize(1280, 900)
    owner.show()
    owner.panel.detail_tabs.setCurrentWidget(owner.panel.resource_page)
    qtbot.mouseClick(resources.refresh_button, Qt.MouseButton.LeftButton)
    assert "Loaded 1 saved machine snapshot" in resources.probe_label.text()
    resources.policy_budget.setChecked(True)
    qtbot.mouseClick(resources.suggested_button, Qt.MouseButton.LeftButton)
    assert "Copied 4 worker(s)" in resources.selection_label.text()
    choices = []
    resources.workers_selected.connect(choices.append)
    qtbot.mouseClick(resources.use_button, Qt.MouseButton.LeftButton)
    assert "Applied 4 worker(s)" in resources.selection_label.text()
    assert choices[0]["workers"] == 4 and choices[0]["starts_job"] is False


def test_main_tables_log_and_error_text_have_no_height_cap(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    for widget in (panel.job_table, panel.machine_table, panel.worker_table, panel.log_text, panel.error_text):
        assert widget.maximumHeight() == UNCAPPED


def test_sources_are_collapsible_and_keep_their_actions(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    owner.show()
    assert owner.sources.title() == "Advanced — imports and paths" and owner.sources.toggle.isCheckable()
    assert owner.shared_jobs_section.isAncestorOf(owner.sources)
    assert not owner.sources.isChecked() and not owner.sources_body.isVisible()
    for widget in (owner.open_job_button, owner.import_plan_button):
        assert owner.sources_body in _ancestors(widget, owner)
    owner.sources.setChecked(True)
    assert owner.shared_jobs_section.is_expanded()
    assert owner.sources_body.isVisible() and owner.open_job_button.isVisible()
    assert not owner.stores.isVisible()
    assert owner.inline_layout.store_results.isAncestorOf(owner.stores)


def test_no_scroll_area_inside_another_on_the_jobs_page(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    for area in owner.findChildren(QScrollArea):
        assert not any(isinstance(parent, QScrollArea) for parent in _ancestors(area, None)), area


def test_store_details_and_preview_have_no_height_cap(qtbot, tmp_path):
    panel = StoreAvailabilityPanel(path=tmp_path / "stores.toml")
    qtbot.addWidget(panel)
    assert panel.details.maximumHeight() == UNCAPPED
    dialog = StoreSettingsDialog(path=tmp_path / "stores.toml")
    qtbot.addWidget(dialog)
    assert dialog.preview.maximumHeight() == UNCAPPED
    assert isinstance(dialog, QWidget)


def test_standalone_resource_choice_reports_selection_without_claiming_application(qtbot, tmp_path):
    """A standalone advisor emits the selection without claiming an owner applied it."""
    resources = ResourceAdvisorPanel(measurement_registry=MeasurementRegistry(tmp_path / "measurements"))
    qtbot.addWidget(resources)
    resources.set_context({"precision": "float32"}, machines_provider=lambda: [
        dict(machine="unregistered", inventory={}, history=[])])
    resources.workers_spin.setValue(5)
    choices = []
    resources.workers_selected.connect(choices.append)
    resources.show()
    qtbot.mouseClick(resources.use_button, Qt.MouseButton.LeftButton)
    assert resources.selection_label.text().startswith("Selected 5 worker(s)")
    assert resources.selection_label.wordWrap()
    assert "Applied" not in resources.selection_label.text()
    assert "No worker was started" in resources.selection_label.text()
    assert len(choices) == 1 and choices[0]["workers"] == 5
    assert choices[0]["starts_job"] is False
    assert not resources.measurement_registry.root.exists()


@pytest.mark.parametrize("matches", [0, 1, 2], ids=["missing", "applied", "ambiguous"])
def test_resource_choice_feedback_follows_actual_target_matches(qtbot, local_registry, tmp_path, matches, record_property):
    """The actual owner reports application or refusal while forwarding the unchanged choice."""
    def forbidden(*_args, **_kwargs):
        """Reject any accidental producer, worker launch or hardware inspection."""
        pytest.fail("Resource selection must not start work")

    owner = ui.NativeJobsWorkspace(
        registry=local_registry, local_machine="local", reader=snapshot,
        measurement_registry=MeasurementRegistry(tmp_path / "measurements"),
        inventory_provider=forbidden, job_creator=forbidden,
        local_launcher=forbidden, remote_launcher=forbidden,
        resource_factory=ResourceAdvisorPanel,
    )
    qtbot.addWidget(owner, before_close_func=lambda widget: qtbot.waitUntil(widget.can_close, timeout=10000))
    owner.setFont(QFont("Segoe UI", 10))
    apply_application_theme(owner)
    panel, resources = owner.panel, owner.resources
    qtbot.waitUntil(panel.can_close)
    selected = dict(panel.target_combo.currentData())
    chosen = panel.local_machine if matches else panel.local_machine + "-history"
    if matches == 2:
        panel.target_combo.addItem("Alternative saved target", dict(selected, key="alternative"))
    panel.workers_spin.setMaximum(2)
    before = (panel.target_combo.currentIndex(), panel.workers_spin.value(), panel.backend_combo.currentData())
    resources.set_context({"precision": "float32"}, machines_provider=lambda: [
        dict(machine=chosen, inventory={}, history=[])])
    resources.workers_spin.setValue(5)
    resources.backend_combo.setCurrentIndex(resources.backend_combo.findData("cuda"))
    choices, forwarded = [], []
    resources.workers_selected.connect(choices.append)
    panel.workers_selected.connect(forwarded.append)
    owner.resize(1280, 1000)
    owner.show()
    panel.detail_tabs.setCurrentWidget(panel.resource_page)
    qtbot.mouseClick(resources.use_button, Qt.MouseButton.LeftButton)
    qtbot.wait(20)
    assert len(choices) == 1 and forwarded == choices
    assert forwarded[0]["workers"] == 5 and forwarded[0]["starts_job"] is False
    feedback = resources.selection_label.text()
    assert panel.progress_label.text() == feedback
    assert resources.selection_label.wordWrap() and resources.reason_label.wordWrap()
    assert owner.width() == 1280 and owner.height() == 1000
    assert owner.minimumSizeHint().width() <= 1280
    for label in (resources.selection_label, resources.reason_label):
        assert 0 <= label.mapTo(owner, label.rect().topLeft()).x()
        assert label.mapTo(owner, label.rect().bottomRight()).x() < owner.width()
    assert not resources.selection_label.visibleRegion().isEmpty()
    if matches == 1:
        assert panel.target_combo.currentData() == selected
        assert panel.workers_spin.value() == 2 and panel.backend_combo.currentData() == "cuda"
        assert "Applied 2 worker(s) using CUDA" in feedback
        assert "Applied 5" not in feedback
        screenshot = tmp_path / "resource_worker_count_applied.png"
        assert owner.grab().save(str(screenshot))
        record_property("resource_feedback_screenshot", str(screenshot))
    else:
        assert (panel.target_combo.currentIndex(), panel.workers_spin.value(), panel.backend_combo.currentData()) == before
        assert "Worker count not applied" in feedback
        assert ("no registered launch target" if matches == 0 else "multiple registered launch targets") in feedback
    assert not local_registry.root.exists() and not resources.measurement_registry.root.exists()
    qtbot.waitUntil(owner.can_close, timeout=10000)
    assert owner.close() and not owner.isVisible()


@pytest.mark.parametrize("matches", [True, False], ids=["applied", "missing"])
def test_themed_owner_resource_fields_and_borrowing_fit_1280_by_800(qtbot, local_registry, tmp_path, matches, record_property):
    """Keep actual worker controls and feedback reachable while both owner pages fit."""
    def forbidden(*_args, **_kwargs):
        """Refuse any producer, worker launch or hardware inspection during layout proof."""
        pytest.fail("Changing resource tabs must not start work")

    owner = ui.NativeJobsWorkspace(
        registry=local_registry, local_machine="local", reader=snapshot,
        measurement_registry=MeasurementRegistry(tmp_path / "measurements"),
        inventory_provider=forbidden, job_creator=forbidden,
        local_launcher=forbidden, remote_launcher=forbidden,
        resource_factory=ResourceAdvisorPanel,
    )
    qtbot.addWidget(owner, before_close_func=lambda widget: qtbot.waitUntil(widget.can_close, timeout=10000))
    owner.setFont(QFont("Segoe UI", 10))
    apply_application_theme(owner)
    project = tmp_path / "project"
    project.mkdir()
    (project / "series.toml").write_text('schema_version=1\nseries_id="campaign"\n', encoding="utf-8")
    owner.set_series("campaign", project, series_label="Copy")
    qtbot.waitUntil(owner.can_close, timeout=10000)
    panel, resources, page = owner.panel, owner.resources, owner.new_analysis
    chosen = panel.local_machine if matches else panel.local_machine + "-history"
    resources.set_context({"precision": "float32"}, machines_provider=lambda: [
        dict(machine=chosen, inventory={}, history=[])])
    resources.workers_spin.setValue(3)
    owner.resize(1280, 800)
    owner.show()
    panel.detail_tabs.setCurrentWidget(panel.resource_page)
    assert [resources.details_tabs.tabText(index) for index in range(resources.details_tabs.count())] == [
        "Hardware", "Measurements", "GPU speed"]
    assert resources.details_tabs.currentIndex() == 0
    qtbot.wait(20)
    owner.jobs_view.ensureWidgetVisible(resources.use_button, 0, 0)
    qtbot.wait(20)
    qtbot.mouseClick(resources.use_button, Qt.MouseButton.LeftButton)
    qtbot.wait(20)
    feedback = resources.selection_label.text()
    assert ("Applied 3 worker(s)" if matches else "Worker count not applied") in feedback
    diagnostics = {}
    for stage in ("jobs", "new_analysis", "jobs_returned"):
        if stage == "new_analysis":
            owner.show_view("new_analysis")
            assert page.machine_box.isAncestorOf(resources)
            for widget in (panel.target_combo, panel.backend_combo, panel.workers_spin):
                assert page.target_layout.indexOf(widget) >= 0
        elif stage == "jobs_returned":
            owner.show_view("jobs")
            assert panel.resource_slot.indexOf(resources) >= 0
            for index, widget in enumerate((panel.target_combo, panel.backend_combo, panel.workers_spin)):
                assert panel._controls.indexOf(widget) == index
        qtbot.wait(20)
        active_scroll = page.scroll if stage == "new_analysis" else owner.jobs_view
        active_scroll.ensureWidgetVisible(resources.selection_label, 0, 0)
        qtbot.wait(20)
        diagnostics[stage] = dict(owner=list(owner.minimumSizeHint().toTuple()),
            views=list(owner.views.minimumSizeHint().toTuple()), jobs=list(owner.jobs_view.minimumSizeHint().toTuple()),
            jobs_body=list(owner.jobs_view.widget().minimumSizeHint().toTuple()),
            jobs_body_height=owner.jobs_view.widget().height(), viewport_height=active_scroll.viewport().height(),
            new_analysis=list(page.minimumSizeHint().toTuple()), resources=list(resources.minimumSizeHint().toTuple()),
            details=list(resources.details_tabs.minimumSizeHint().toTuple()),
            resource_height=resources.height(), resource_layout_height=resources.layout().geometry().height(),
            detail_height=resources.details_tabs.height(),
            page_height=resources.details_tabs.currentWidget().height(),
            page_layout_height=resources.details_tabs.currentWidget().layout().geometry().height(), controls={})
        for name in ("machine_combo", "backend_combo", "threads_spin", "workers_spin"):
            control = getattr(resources, name)
            combo = name.endswith("combo")
            option = QStyleOptionComboBox() if combo else QStyleOptionSpinBox()
            control.initStyleOption(option)
            style = control.style()
            content_type = QStyle.ContentsType.CT_ComboBox if combo else QStyle.ContentsType.CT_SpinBox
            font_height = control.fontMetrics().height()
            styled_height = style.sizeFromContents(content_type, option, QSize(1, font_height), control).height()
            frame = style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, option, control)
            required_height = max(control.minimumSizeHint().height(), font_height + 2 * frame, styled_height)
            diagnostics[stage]["controls"][name] = dict(height=control.height(),
                size_hint=list(control.sizeHint().toTuple()), minimum_hint=list(control.minimumSizeHint().toTuple()),
                font_height=font_height, styled_height=styled_height, required_height=required_height,
                fully_visible=control.visibleRegion().contains(control.rect()),
                parent_height=control.parentWidget().height(),
                parent_layout_height=control.parentWidget().layout().geometry().height())
        record_property("resource_fields_layout_minima_" + stage, json.dumps(diagnostics[stage], sort_keys=True))
        screenshot = tmp_path / ("resource_fields_" + ("applied_" if matches else "missing_") + stage + ".png")
        assert owner.grab().save(str(screenshot))
        record_property("resource_fields_layout_screenshot_" + stage, str(screenshot))
        assert owner.width() == 1280 and owner.height() == 800, diagnostics
        assert owner.minimumSizeHint().width() <= 1280 and owner.minimumSizeHint().height() <= 800, diagnostics
        for area in owner.findChildren(QScrollArea):
            assert not any(isinstance(parent, QScrollArea) for parent in _ancestors(area, None))
        for label in (resources.selection_label, resources.reason_label):
            assert label.isVisible() and label.visibleRegion().contains(label.rect()), diagnostics
            assert 0 <= label.mapTo(owner, label.rect().topLeft()).y()
            assert label.mapTo(owner, label.rect().bottomRight()).y() < owner.height()
        for name, metrics in diagnostics[stage]["controls"].items():
            control = getattr(resources, name)
            assert metrics["height"] >= metrics["required_height"], diagnostics
            assert control.isVisible() and metrics["fully_visible"], diagnostics
            assert 0 <= control.mapTo(owner, control.rect().topLeft()).y()
            assert control.mapTo(owner, control.rect().bottomRight()).y() < owner.height()
        assert resources.selection_label.text() == feedback
    assert not local_registry.root.exists() and not resources.measurement_registry.root.exists()
    qtbot.waitUntil(owner.can_close, timeout=10000)
    assert owner.close() and not owner.isVisible()
