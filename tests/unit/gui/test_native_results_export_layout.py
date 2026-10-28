"""Real saved-result widgets keep their behaviour in the native two-column page."""

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea

from darsia.fluidflower.native_results_export_layout import NativeResultsExportLayout
from .test_calibration_review_approval import _page, _series


def _items(layout):
    return [layout.itemAt(i).widget() or layout.itemAt(i).layout()
            for i in range(layout.count())]


@pytest.fixture
def results_page(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    page, _ = _page(qtbot, monkeypatch, root)
    page.native_start_analysis_button = QPushButton("Start analysis…", page)
    page.layout().insertWidget(0, page.native_start_analysis_button)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original = [_items(layout) for layout in (page.layout(), page.scroll.widget().layout(),
                                             page.saved_results.layout())]
    adapter = page.results_export_layout = NativeResultsExportLayout(page)
    try:
        yield page, adapter, original
        assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    finally:
        page._stop_lookups()


def test_results_have_left_foldable_controls_and_original_large_viewer(results_page, qtbot, tmp_path):
    page, adapter, _ = results_page
    saved = page.saved_results
    original_views, original_slider = saved.views, saved.slider
    page.resize(1650, 950)
    page.show_view("results")
    page.show()
    qtbot.waitUntil(saved.views.isVisible)
    QApplication.processEvents()
    workflow = adapter.workflow
    assert workflow.workspace is saved
    assert saved.views is original_views and saved.slider is original_slider
    assert not page.scroll.isVisible()
    assert workflow.controls_scroll.isVisible()
    for control in (page.model, saved.view_experiment, saved.preview_field,
                    saved.export_buttons[0], page.publish_button, page.native_start_analysis_button):
        assert workflow.controls_widget.isAncestorOf(control)
    assert saved.isAncestorOf(saved.slider) and saved.views.isVisible()
    assert workflow.controls_scroll.mapTo(page, QPoint(workflow.controls_scroll.width(), 0)).x() <= saved.mapTo(page, QPoint()).x()
    assert saved.width() > workflow.controls_scroll.width()
    assert saved.views.height() > 450
    assert saved.result is not None and not saved.photo.pixmap().isNull()
    adapter.sections["model"].toggle.click()
    QApplication.processEvents()
    assert page.model.isVisible()
    assert page.model.mapTo(workflow.controls_widget, QPoint(page.model.width(), 0)).x() <= workflow.controls_widget.width()
    adapter.sections["model"].toggle.click()
    qtbot.waitUntil(lambda: adapter.sections["model"].height() < 100)
    section = adapter.sections["results"]
    section.toggle.click()
    assert not section.content.isVisible() and saved.views.isVisible()
    section.toggle.click()
    assert saved.view_experiment.isVisible()
    QApplication.processEvents()
    rendered = page.grab()
    assert not rendered.isNull()
    assert rendered.save(str(tmp_path / "results-and-export.png"))


def test_export_map_and_frame_callbacks_keep_original_values(results_page, monkeypatch):
    page, adapter, _ = results_page
    saved = page.saved_results
    page.show_view("results")
    saved.slider.setValue(0)
    saved.next_image.click()
    assert saved.slider.value() == 1
    saved.previous_image.click()
    assert saved.slider.value() == 0
    selected = saved.result_path
    saved.color_scale.setCurrentIndex(saved.color_scale.findData("power"))
    assert saved.color_scale.currentData() == "power" and saved.result_path == selected
    exported, started = [], []
    monkeypatch.setattr(saved, "export", lambda kind: exported.append(kind))
    page.native_start_analysis_button.clicked.connect(lambda: started.append(True))
    for button in saved.export_buttons:
        button.setEnabled(True)
        button.click()
    assert exported == ["movie", "figure", "csv", "bundle"]
    adapter.sections["export"].toggle.click()
    assert not saved.export_toggle.isChecked()
    adapter.sections["export"].toggle.click()
    assert saved.export_toggle.isChecked()
    assert started == [] and saved.process is None
    assert not page.player.timer.isActive()


def test_review_restores_exact_original_layouts_and_results_folds(results_page, qtbot):
    page, adapter, original = results_page
    saved = page.saved_results
    old_minimum = saved.views.minimumHeight()
    page.show_view("results")
    adapter.sections["map"].toggle.click()
    assert adapter.sections["map"].is_expanded()
    selected = saved.result_path
    page.show_view("review")
    QApplication.processEvents()
    assert adapter.workflow is None
    assert [_items(layout) for layout in (page.layout(), page.scroll.widget().layout(),
                                         saved.layout())] == original
    assert saved.views.minimumHeight() == old_minimum
    assert saved.result_path == selected
    qtbot.waitUntil(lambda: page.findChildren(QScrollArea) == [page.scroll])
    assert page.findChildren(QScrollArea) == [page.scroll]
    assert saved.review_mode and saved.views.isHidden()
    page.show_view("results")
    assert adapter.sections["map"].is_expanded()
    assert not saved.review_mode and saved.result_path == selected
    assert saved.process is None
