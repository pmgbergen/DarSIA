"""A1 controls with the original finite job API and saved-figure browser.

These are synthetic metadata/Qt cases. The inherited transport and launch
seams never start a process; no campaign calibration or chemistry is fitted.
"""
import json
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QAbstractItemView, QMessageBox, QWidget

from darsia.fluidflower import refit_job_panel as refit, store_map
from darsia.fluidflower.pipeline import refit_jobs
from darsia.fluidflower.series_method import MethodSelection, save_series_method
from .test_refit_job_panel import series, _panel, _idle, _create, _bound, YES  # noqa: F401
from .test_native_series_method import method_window, _roots  # noqa: F401


def test_explanations_follow_actual_rules_and_model_choices_are_honest(qtbot, series):
    panel = _panel(qtbot, series)
    assert panel.name_edit.text() == "" and not panel.start_button.isEnabled()
    assert panel.chemistry_combo.currentData() == "titration"
    for index in (1, 2):
        assert not panel.chemistry_combo.model().item(index).isEnabled()
        assert "Not available yet" in panel.chemistry_combo.itemText(index)
    text = panel.parameter_explanation.text()
    assert all(term in text for term in ("Gain", "Beta", "anchor", "dilution factor", "not another fitted parameter"))
    assert "recipe is not fitted" in panel.measured_inputs.text()
    assert "r1, r2" in panel.fit_images.text() and "Choose the rules" in panel.fit_images.text()
    assert panel.load_rules(series.rules)
    text = panel.fit_images.text()
    assert "0.5-2.375 h" in text and "from 2.5 h" in text and "2.247 h" in text
    assert "gain fixed" in text and "Held-out" in text and "Leave-one-run-out" in text
    assert panel.recipe_table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    document = series.root / "explicit-titration.json"
    document.write_bytes(b'{"chemistry_model":"titration","parameters":{"gain":24,"beta":0.6}}')
    from darsia.fluidflower.mass_calibration_display import read_mass_calibration_display
    shown = read_mass_calibration_display("phase_model", document)
    assert next(value for key, value, _source in shown.rows if key == "Chemistry model") == "Titration"


def test_named_gui_job_persists_titration_and_keeps_name_without_start(qtbot, series, monkeypatch):
    # This original finite specimen predates local Build receipts. Keep its
    # existing job writer while checking that the GUI now requests local mode;
    # the local Build route has its own execution/lifecycle tests.
    create_job = refit.create_job
    def original_fixture_job(root, **options):
        assert options.pop('local') is True
        return create_job(root, **options)
    monkeypatch.setattr(refit, 'create_job', original_fixture_job)
    panel = _panel(qtbot, series)
    panel.load_rules(series.rules)
    panel.check_phase_job(series.phase_job)
    _idle(qtbot, panel)
    panel.jobs_edit.setText(str(series.jobs))
    assert panel._create_blocker() == "Name the calibration first."
    panel.name_edit.setText("my-titration-calibration")
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: YES)
    assert panel.create()
    qtbot.waitUntil(lambda: panel._thread is None and panel.snapshot is not None, timeout=20000)
    assert panel.name_edit.text() == panel.job_root.name == "my-titration-calibration"
    assert panel.start_button.isEnabled()
    plans = sorted((series.root / "calibration/refit/my-titration-calibration").glob("*_plan.json"))
    assert len(plans) == 3
    for path in plans:
        assert json.loads(path.read_bytes())["chemistry_model"] == "titration"
    assert not (panel.job_root / "ui/launches").exists()


@pytest.mark.parametrize("change", ["name", "chemistry"])
def test_start_rechecks_name_and_supported_model_after_its_question(qtbot, series, monkeypatch, change):
    job = _create(series)
    stores = series.tmp / "stores.toml"
    stores.write_bytes(b"# finite settings\n")
    monkeypatch.setattr(store_map, "default_store_map_path", lambda **_kwargs: stores)
    panel = _panel(qtbot, series)
    panel.open_job(job)
    _idle(qtbot, panel)
    assert panel.name_edit.text() == job.name and panel.start_button.isEnabled()

    def change_during_question(*_args, **_kwargs):
        if change == "name":
            panel.name_edit.setText("another-calibration")
        else:
            panel.chemistry_combo.setCurrentIndex(1)  # A direct call must also fail closed.
        return YES

    monkeypatch.setattr(QMessageBox, "question", change_during_question)
    assert not panel.start() and not panel.start_button.isEnabled()
    assert not (job / "ui/launches").exists()


def test_recipe_values_are_read_only_and_only_from_exact_completed_configuration(qtbot, series):
    for run in series.roles["fit"]:
        attempt = refit_jobs.completed_attempt(series.phase_job, run)
        configuration = attempt / "configuration.json"
        configuration.write_bytes(json.dumps(dict(run=run, recipe=dict(alkalinity_M=0.0005, btb_M=0.0001),
            source_metadata_padding=" " * refit.MAX_RULES_BYTES)).encode())
        assert configuration.stat().st_size > refit.MAX_RULES_BYTES
        manifest = attempt / "manifest.json"
        manifest.write_bytes(json.dumps(dict(configuration=_bound(configuration))).encode())
        completion = series.phase_job / "results" / run / "completed.json"
        completion.write_bytes(json.dumps(dict(complete=True, run=run, manifest=_bound(manifest))).encode())
    panel = _panel(qtbot, series)
    panel.check_phase_job(series.phase_job)
    _idle(qtbot, panel)
    assert panel.recipe_table.rowCount() == 2 and panel.phase_ready
    assert panel.recipe_table.item(0, 1).text() == "0.0005"
    assert panel.recipe_table.item(0, 2).text() == "0.0001"
    assert not panel.recipe_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    configuration = refit_jobs.completed_attempt(series.phase_job, "r1") / "configuration.json"
    configuration.write_bytes(b'{"run":"r1","recipe":{"alkalinity_M":1,"btb_M":1}}')
    panel.check_phase_job(series.phase_job)
    _idle(qtbot, panel)
    assert panel.recipe_table.item(0, 1).text() == "Not reported"
    assert "differs" in panel.recipe_table.item(0, 3).text()
    assert not (series.root / "calibration/refit").exists()


def test_embedded_overflow_choice_controls_only_original_saved_figure(method_window, qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower.tools import phase_studio_series_ui as saved_ui

    window = method_window
    qtbot.waitUntil(lambda: not window._series_loading)
    root = _roots(window)["series-a"]
    calibration = root / "frozen.json"
    calibration.write_bytes(b'{"overflow_in_objective":false,"chemistry_model":"titration"}')
    save_series_method(root, MethodSelection(active="phase_model", model_version="m", mass_calibration="frozen.json"))
    window._refresh_method_state(force=True)
    monkeypatch.setattr(saved_ui, "studio_series_keys", lambda: ())  # Finite display inventory only.
    browser = saved_ui.SeriesMassWorkspace(cache_root=tmp_path / "unused-display-cache")
    qtbot.addWidget(browser)
    pixmap = QPixmap(12, 8)
    pixmap.fill(Qt.GlobalColor.blue)
    browser.overflow_plot.set_source_pixmap(pixmap)
    browser.result = dict(overflow_curve="saved-only.png")  # A finite existing-figure DISPLAY specimen.
    host = QWidget()
    qtbot.addWidget(host)
    host.saved_results = browser
    shown = []
    host.show_view = shown.append
    window.mass_analysis_page = host
    window.content_stack.addWidget(host)
    monkeypatch.setattr(window, "ensure_mass_analysis", lambda: host)
    before = (root / "series.toml").read_bytes(), calibration.read_bytes()
    window._show_mass_calibration_values("phase_model")
    choice = window.mass_calibration_display.overflow_choice
    index = browser.views.indexOf(browser.overflow_plot)
    original_tabs = [browser.views.isTabVisible(i) for i in range(browser.views.count())]
    key = browser.overflow_plot._source_pixmap.cacheKey()
    choice.choice["off"].click()
    assert not browser.views.isTabVisible(index)
    assert "hidden" in choice.detail.text() and not choice.comparison_button.isEnabled()
    choice.choice["compare"].click()
    assert browser.views.isTabVisible(index) and "existing saved" in choice.detail.text()
    choice.comparison_button.click()
    assert shown == ["review"] and browser.views.currentWidget() is browser.overflow_plot
    assert browser.overflow_plot._source_pixmap.cacheKey() == key
    assert [browser.views.isTabVisible(i) for i in range(browser.views.count())] == original_tabs
    assert choice.steer.isEnabled()
    choice.steer.click()
    assert window.mass_calibration_display.refit_panel.f_check.isChecked()
    assert ((root / "series.toml").read_bytes(), calibration.read_bytes()) == before
    choice.set_comparison_available(False)
    assert "No saved overflow comparison" in choice.detail.text()
    assert "measured overflow targets" in choice.detail.text()
    browser.close()

