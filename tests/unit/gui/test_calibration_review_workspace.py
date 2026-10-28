"""Selection and lifecycle tests of the real retained calibration-review page."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea
import pytest

from .calibration_review_fixture import add_saved_results
from .test_calibration_review_approval import _page, _series


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_early_named_choice_preserves_matching_time_and_shared_calibration_between_runs(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    (root / "fits/other.json").write_text('{"frozen": true, "variant": 2}', encoding="utf-8")
    page, experiments = _page(qtbot, monkeypatch, root)
    add_saved_results(page, root, ("ac28", "ac47"), name="Calibration B",
                      reference="fits/other.json", times=(1., 0.))
    before = _files(root)
    saved = page.saved_results
    try:
        page.show()
        page.show_view("review")
        selector = saved.mass_calibration
        a = selector.findData(str((root / "fits/frozen.json").resolve()))
        b = selector.findData(str((root / "fits/other.json").resolve()))
        assert selector.itemText(a) == "Calibration A · Titration"
        assert selector.itemText(b) == "Calibration B · Titration"
        selector.setCurrentIndex(a)
        saved.slider.setValue(1)
        assert saved.result["frames"][saved.slider.value()]["source_name"] == "image-1.0.dng"
        selector.setCurrentIndex(b)
        assert saved.result_path.parts[-3] == "Calibration B"
        assert saved.slider.value() == 0  # Same photograph/time, not old list index.
        experiments.setCurrentIndex(experiments.findData("ac47"))
        assert saved.result_path.parts[-3] == "Calibration B"
        assert saved.result["selected_inputs"]["candidate"]["run"] == "ac47"
        assert page.findChildren(QScrollArea) == [page.scroll]
        assert _files(root) == before
    finally:
        page._stop_lookups()


def test_legacy_result_filters_never_hide_the_selected_review_calibration(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    page, _ = _page(qtbot, monkeypatch, root)
    saved = page.saved_results
    try:
        saved.illumination_filter.setCurrentIndex(saved.illumination_filter.findData("facies"))
        saved.colorchecker_filter.setCurrentIndex(saved.colorchecker_filter.findData("off"))
        saved.model_scope_filter.setCurrentIndex(saved.model_scope_filter.findData("individual"))
        page.show_view("review")
        assert saved.result is not None
        selected = saved.result_path
        assert all(widget.isHidden() for widget in saved._legacy_review_rows)
        assert saved.views.isHidden()
        assert saved.review_header.isVisibleTo(page.scroll.widget())
        contents = [saved.review_panels.panels[index].content for index in range(3)]
        assert all("phase" not in box.itemText(item).lower()
                   for box in contents for item in range(box.count()))
        page.show_view("results")
        assert all(not widget.isHidden() for widget in saved._legacy_review_rows)
        assert not saved.views.isHidden()
        assert saved.result_path == selected
        page.show_view("review")
        assert saved.result_path == selected
    finally:
        page._stop_lookups()


def test_absent_run_in_selected_calibration_cannot_fall_back_to_another_fit(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    (root / "fits/other.json").write_text('{"frozen": true, "variant": 2}', encoding="utf-8")
    page, experiments = _page(qtbot, monkeypatch, root)
    add_saved_results(page, root, ("ac28",), name="Calibration B", reference="fits/other.json")
    try:
        page.show_view("review")
        saved = page.saved_results
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData(str((root / "fits/other.json").resolve())))
        assert experiments.findData("ac47") == -1
        assert not page.run_list.table.item(1, 0).flags() & Qt.ItemFlag.ItemIsSelectable
        assert saved.result_path.parts[-3] == "Calibration B"
    finally:
        page._stop_lookups()


def test_literal_names_do_not_merge_different_saved_calibration_references(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    page, _ = _page(qtbot, monkeypatch, root)
    add_saved_results(page, root, ("ac28",), name="Calibration B", reference="fits/other.json")
    saved = page.saved_results
    for entry in saved.entries:
        entry["review_calibration_name"] = "Same name"
    try:
        page.show_view("review")
        assert saved.mass_calibration.count() == 2
        assert len({saved.mass_calibration.itemData(i) for i in range(2)}) == 2
        assert all(saved.mass_calibration.itemText(i).startswith("Same name · Titration · ") for i in range(2))
    finally:
        page._stop_lookups()


@pytest.mark.parametrize("active", [False, True])
def test_first_default_series_visit_initializes_original_saved_result_discovery(qtbot, monkeypatch, tmp_path, active):
    from darsia.fluidflower import calibration_review_workspace as workspace
    from darsia.fluidflower.tools import phase_studio_series_ui as retained
    from .test_calibration_review_panels import _Signal

    calls = []

    class Lookup:
        def __init__(self, dataset, parent, cache_root):
            self.dataset, self.ready = dataset, _Signal()
        def start(self):
            calls.append(self.dataset)
            self.ready.emit(self.dataset, [], "")

    monkeypatch.setattr(retained, "studio_series_keys", lambda: ("ffum",))
    monkeypatch.setattr(retained, "series_path", lambda dataset, folder: tmp_path / folder)
    monkeypatch.setattr(retained, "SavedSeriesResultLookup", Lookup)
    saved = workspace.CalibrationReviewWorkspace(cache_root=tmp_path / "cache")
    qtbot.addWidget(saved)
    try:
        assert saved.series.currentData() == "ffum" and saved.visible == []
        saved.set_review_mode(active)
        assert calls == ["ffum"] and "ffum" in saved._series_catalog_loaded
        saved.set_review_mode(not active)
        assert calls == ["ffum"]
    finally:
        saved.stop_review_reads()


def test_late_catalogue_adds_runs_without_replacing_viewed_calibration_or_time(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    page, _ = _page(qtbot, monkeypatch, root)
    saved = page.saved_results
    original = list(saved.entries)
    add_saved_results(page, root, ("ac99",))
    late = saved.entries[len(original):]
    saved.entries = original
    saved.refresh()
    try:
        page.show_view("review")
        saved.slider.setValue(1)
        path, calibration = saved.result_path, saved.mass_calibration.currentData()
        saved._series_results_ready(page.series, late, "")
        assert saved.result_path == path and saved.slider.value() == 1
        assert saved.mass_calibration.currentData() == calibration
        assert saved.view_experiment.findData("ac99") >= 0
        assert saved.review_panels.index == 1 and saved.review_panels.path == path
    finally:
        page._stop_lookups()


def test_saved_overflow_action_targets_visible_review_panel_and_retained_results_tab(qtbot, monkeypatch, tmp_path):
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    saved = page.saved_results
    try:
        page.show_view("review")
        saved.show_saved_overflow_comparison()
        assert saved.review_panels.panels[0].content.currentData() == "overflow_curve"
        assert not saved.review_panels.panels[0].isHidden()
        assert saved.views.isHidden()
        page.show_view("results")
        saved.show_saved_overflow_comparison()
        assert saved.views.currentWidget() is saved.overflow_plot
    finally:
        page._stop_lookups()


def test_reused_calibration_path_with_different_recorded_checksums_is_not_one_fit(qtbot, monkeypatch, tmp_path):
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    saved = page.saved_results
    saved.entries[1]["review_calibration_sha256"] = "e" * 64
    try:
        page.show_view("review")
        assert saved.mass_calibration.count() == 2
        keys = {saved.mass_calibration.itemData(index) for index in range(2)}
        assert len(keys) == 2
        assert any(key.endswith("e" * 64) for key in keys)
        assert saved.view_experiment.count() == 1
    finally:
        page._stop_lookups()


def test_refresh_does_not_replace_a_disappeared_calibration_with_another_fit(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    page, _ = _page(qtbot, monkeypatch, root)
    add_saved_results(page, root, ("ac28",), name="Calibration B", reference="fits/other.json")
    saved = page.saved_results
    try:
        page.show_view("review")
        selected = str((root / "fits/other.json").resolve())
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData(selected))
        saved.entries = [entry for entry in saved.entries if entry["review_calibration_id"] != selected]
        saved.refresh(keep_view=True)
        assert saved.mass_calibration.currentData() == selected
        assert "unavailable" in saved.mass_calibration.currentText()
        assert saved.result is None and saved.view_experiment.count() == 0
        assert saved.mass_calibration.count() == 2
    finally:
        page._stop_lookups()


def test_review_uses_worker_panels_instead_of_loading_hidden_legacy_graphs(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower.tools import phase_studio_series_ui as retained
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    calls, original = [], retained.SeriesMassWorkspace._load_saved_figures
    def tracked(workspace, value):
        calls.append(value)
        return original(workspace, value)
    monkeypatch.setattr(retained.SeriesMassWorkspace, "_load_saved_figures", tracked)
    try:
        page.show_view("review")
        assert calls == []
        page.show_view("results")
        assert calls
    finally:
        page._stop_lookups()


def test_first_review_visit_preserves_the_calibration_and_image_already_open_in_results(qtbot, monkeypatch, tmp_path):
    root = _series(tmp_path)
    page, _ = _page(qtbot, monkeypatch, root)
    add_saved_results(page, root, ("ac28",), name="Calibration B", reference="fits/other.json")
    saved = page.saved_results
    try:
        page.show_view("results")
        selected = next(entry["path"] for entry in saved.entries if entry["review_calibration_name"] == "Calibration B")
        saved.load_result(selected)
        saved.slider.setValue(1)
        page.show_view("review")
        assert str(saved.result_path) == selected
        assert saved.slider.value() == 1
        assert saved.mass_calibration.currentData() == str((root / "fits/other.json").resolve())
    finally:
        page._stop_lookups()
