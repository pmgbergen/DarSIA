import tomllib

from darsia.fluidflower.native_calibration_image_selection import CalibrationImageSelection
from .test_native_series_method import method_window, _roots, _files  # noqa: F401


def test_calibration_page_saves_only_fit_count_with_feedback(qtbot, tmp_path):
    raw = 'schema_version = 1\n[runs]\nids = ["sample"]\n[image_selection.color_paths]\nmax_images_per_run = 15\n'
    path = tmp_path / "series.toml"
    path.write_text(raw, encoding="utf-8")
    page = CalibrationImageSelection()
    qtbot.addWidget(page)
    page.set_series(tmp_path)
    qtbot.waitUntil(lambda: page.snapshot is not None)
    assert page.count.value() == 30 and path.read_text() == raw
    page.count.setValue(20)
    assert "Unsaved" in page.status.text()
    page.save_button.click()
    qtbot.waitUntil(lambda: "✓" in page.status.text())
    assert page.can_close()
    document = tomllib.loads(path.read_text())
    assert document["calibration"]["mass"]["image_selection"]["max_images_per_run"] == 20
    assert document["image_selection"]["color_paths"]["max_images_per_run"] == 15
    assert "No job started" in page.status.text()


def test_native_calibration_navigation_does_not_open_setup_selector(method_window, qtbot, monkeypatch):
    window = method_window
    monkeypatch.setattr(window, "open_image_selection_plan", lambda: (_ for _ in ()).throw(
        AssertionError("Calibration routed to Setup image selection")))
    before = _files(_roots(window)["series-a"])
    window.open_calibration_image_selection()
    page = window.calibration_image_selection
    assert window.sidebar.get_selected() == ("calibration", "images")
    assert window.content_stack.currentWidget() is page
    qtbot.waitUntil(lambda: page.snapshot is not None)
    assert not window.current_row_scope.run_selector_enabled
    assert not window.toolbar_builder.play_action.isEnabled()
    assert _files(_roots(window)["series-a"]) == before


def test_choice_page_header_and_sidebar_have_no_missing_output_or_start(method_window, qtbot):
    window = method_window
    window.open_calibration_image_selection()
    page = window.calibration_image_selection
    qtbot.waitUntil(lambda: page.snapshot is not None)
    adapter = window.workflow_status
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    assert adapter.states[("calibration", "images")].state == "default_choice"
    assert "Default choice" in adapter.label.text() and "30 calibration target images" in adapter.label.text()
    assert not adapter.start_button.isVisible() and not adapter.open_button.isVisible()
    assert not adapter.missing_label.isVisible()
    assert "Missing:" not in adapter._details(adapter.states[("calibration", "images")])
    page.count.setValue(20)
    page.save_button.click()
    qtbot.waitUntil(lambda: "✓" in page.status.text())
    qtbot.waitUntil(lambda: not window._series_loader.busy())
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    assert adapter.states[("calibration", "images")].state == "saved_choice"
    assert "Saved choice" in adapter.label.text() and "20 calibration target images" in adapter.label.text()
    assert not adapter.start_button.isVisible() and not adapter.missing_label.isVisible()


def test_page_lists_the_phase_models_labelled_training_images(qtbot, tmp_path):
    page = CalibrationImageSelection()
    qtbot.addWidget(page)
    page.root = tmp_path
    page._training_read((tmp_path, dict(model="model-a", bundles=2, unreadable=[],
                                        runs={"run1": dict(times_h=[1.0, 2.5], labels=3)})), "")
    assert page.training_status.text() == "Model model-a: 2 labelled images in 1 runs."
    assert [page.training_table.item(0, column).text() for column in range(4)] == ["run1", "2", "3", "1, 2.5"]
    page._training_read(None, "ValueError: No phase model is named")
    assert "could not be read" in page.training_status.text()
