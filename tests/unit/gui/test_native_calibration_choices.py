"""Native choice pages save their own inputs without running a calibration."""
import tomllib
from types import SimpleNamespace

from darsia.fluidflower.series_method import MethodSelection, save_series_method
from .test_native_series_method import method_window, _roots, _files  # noqa: F401


def test_first_method_choice_needs_no_model_or_fit_and_never_opens_studio(method_window, qtbot, monkeypatch):
    window = method_window
    root = _roots(window)["series-a"]
    monkeypatch.setattr(window.phase_model_workspace, "ensure_studio",
                        lambda: (_ for _ in ()).throw(AssertionError("Method choice must not load Studio")))
    before = _files(root)
    window._select_row(("method", "use_for_series"))
    panel = window.method_panel
    assert window.content_stack.currentWidget() is panel
    assert panel.title.text() == "Calibration Method"
    assert not panel.model_version.isVisible() and not panel.mass_calibration.isVisible()
    assert not panel.colour_calibration.isVisible()
    assert _files(root) == before
    panel.phase_model_button.click()
    assert panel.use_button.isEnabled()
    panel.use_button.click()
    qtbot.waitUntil(lambda: panel.can_close() and window._saved_method.active == "phase_model")
    document = tomllib.loads((root / "series.toml").read_text())
    assert document["method"] == {"active": "phase_model"}
    assert document["future"] == {"opaque": [1, 2, 3]}
    assert not panel.calls.timer.isActive()
    assert "saved" in panel.use_button.text().lower()
    panel.colour_path_button.click()
    panel.use_button.click()
    qtbot.waitUntil(lambda: panel.can_close() and window._saved_method.active == "colour_path")
    row = window.sidebar._sections["calibration"].get_row("phase_model:training")
    assert not row.isEnabled() and not row.isHidden()


def test_select_model_records_catalogue_reference_without_mass_calibration(method_window, qtbot):
    window = method_window
    root = _roots(window)["series-a"]
    save_series_method(root, MethodSelection(active="phase_model", model_version="models/old",
                                           mass_calibration="old-fit.json"))
    window._refresh_method_state(force=True)
    outcomes = []
    entry = {"root": "models/new"}
    steps = SimpleNamespace(set_model_selection_result=lambda *args: outcomes.append(args))
    window.phase_model_workspace.studio_page = SimpleNamespace(
        studio=SimpleNamespace(_selected_validation_model=lambda: entry), native_steps=steps)
    try:
        window._select_phase_model_for_calibration("models/other")
        assert outcomes[-1][0] is False
        window._select_phase_model_for_calibration("models/new")
        qtbot.waitUntil(lambda: window.method_panel.can_close() and window._saved_method.model_version == "models/new")
        qtbot.waitUntil(lambda: outcomes[-1][0] is True)
        method = tomllib.loads((root / "series.toml").read_text())["method"]["phase_model"]
        assert method["model_version"] == "models/new"
        assert method["mass_calibration"] == ""
        assert method["previous_mass_calibrations"] == ["old-fit.json"]
    finally:
        window.phase_model_workspace.studio_page = None


def test_saved_calibration_is_separate_and_opens_without_preparing_jobs(method_window, monkeypatch):
    window = method_window
    root = _roots(window)["series-a"]
    before = _files(root)
    monkeypatch.setattr(window.mass_calibration_display.preparation, "set_series",
                        lambda *args: (_ for _ in ()).throw(AssertionError("Saved page cannot prepare inputs")))
    window._select_row(("calibration", "saved"))
    assert window.content_stack.currentWidget() is window.mass_calibration_display.saved_calibration_page
    assert _files(root) == before
    assert window.sidebar._sections["calibration"].get_row("calibration_controls:interactive").isHidden()


def test_saved_calibration_refreshes_on_series_change(method_window, qtbot):
    window = method_window
    root = _roots(window)["series-a"]
    (root / "fit.json").write_text('{"parameters":{"gain":17,"beta":0.4}}')
    save_series_method(root, MethodSelection(active="phase_model", model_version="model", mass_calibration="fit.json"))
    window._refresh_method_state(force=True)
    window._select_row(("calibration", "saved"))
    assert window.mass_calibration_display.parameters.item(0, 1).text() == "17"
    window._select_series("series-b")
    qtbot.waitUntil(lambda: not window._series_loading)
    assert window.content_stack.currentWidget() is window.mass_calibration_display.saved_calibration_page
    assert "17" != window.mass_calibration_display.parameters.item(0, 1).text()
