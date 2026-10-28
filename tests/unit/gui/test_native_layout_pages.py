"""N141 Part 1 layout: the Calibration and Analysis rows and their pages.

Real offscreen Qt on local manifests. The Studio, the saved-result page and
Jobs are stand-ins or local objects; nothing is computed, started or written
except an explicit method save.
"""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox, QScrollArea, QWidget

from darsia.fluidflower.series_method import MethodSelection, save_series_method
from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow

from .test_native_phase_steps import setup  # noqa: F401
from .test_native_series_method import _files, _phase_choice, _roots, method_window  # noqa: F401

# The common method fixture replaces this factory to isolate method changes.
_original_mass_analysis_factory = NativeFluidFlowerMainWindow.ensure_mass_analysis


def _visible(window, category):
    """(display id, label) of the shown rows, in screen order."""
    layout = window.sidebar._sections[category].items_layout
    rows = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if hasattr(widget, "checkbox_id") and not widget.isHidden():
            display = window.sidebar.display_selection(widget.action, widget.checkbox_id)[1]
            rows.append((display, widget.label))
    return rows


def _ancestors(widget):
    parent = widget.parentWidget()
    while parent is not None:
        yield parent
        parent = parent.parentWidget()


def test_main_window_installs_native_mass_and_results_panels(method_window):
    from darsia.fluidflower.native_mass_calibration_layout import NativeMassCalibrationDisplay
    from darsia.fluidflower.native_results_export_layout import NativeResultsExportLayout

    window = method_window
    assert isinstance(window.mass_calibration_display, NativeMassCalibrationDisplay)
    assert len(window.mass_calibration_display.sections) == 5
    page = _original_mass_analysis_factory(window)
    assert isinstance(page.results_export_layout, NativeResultsExportLayout)
    original = page.saved_results
    page.show_view("results")
    assert page.results_export_layout.workflow.workspace is original
    assert "export" in page.results_export_layout.sections
    page.show_view("review")
    assert page.saved_results is original and page.results_export_layout.workflow is None


PHASE_CALIBRATION = [
    ("phase_model:training", "Phase model"),
    ("phase_model:mass_from_phase", "Mass calibration"),
    ("phase_model:calibration_review", "Calibration review"),
    ("phase_model:checks", "Checks"),
]
COLOUR_CALIBRATION = [
    ("color", "Colour path"),
    ("mass", "Mass"),
    ("reset", "Reset mass calibration"),
    ("delete", "Delete all calibrations"),
]
USE = [("method:use_for_series", "Use for this series")]
PHASE_ANALYSIS = [
    ("jobs:new_analysis", "New analysis"),
    ("mass_analysis:frozen", "Results and export"),
    ("jobs:production_results", "Results"),
]
COLOUR_ANALYSIS = ["cropping", "mass", "segmentation", "fingers", "thresholding", "volume"]


def test_sidebar_structure_names_and_order_follow_the_chosen_method(method_window):
    window = method_window
    _phase_choice(_roots(window)["series-a"])
    window._refresh_method_state()
    assert _visible(window, "jobs") == [
        ("jobs:overview", "Jobs and workers"),
        ("jobs:sources", "Shared folders and plans"),
        ("jobs:wasserstein", "Wasserstein comparison"),
    ]
    assert _visible(window, "calibration") == (
        PHASE_CALIBRATION + COLOUR_CALIBRATION + USE
    )
    analysis = _visible(window, "analysis")
    assert analysis[:len(PHASE_ANALYSIS)] == PHASE_ANALYSIS
    assert [row for row, _label in analysis[len(PHASE_ANALYSIS):]] == COLOUR_ANALYSIS
    save_series_method(
        _roots(window)["series-a"],
        MethodSelection(active="colour_path", colour_calibration="colour"),
    )
    window._refresh_method_state()
    # Preparation stays in Setup; Use for this series stays last.
    assert _visible(window, "calibration") == (
        COLOUR_CALIBRATION + PHASE_CALIBRATION + USE
    )
    analysis = _visible(window, "analysis")
    assert [row for row, _label in analysis[:6]] == COLOUR_ANALYSIS
    assert [row for row, _label in analysis[6:]] == [row for row, _label in PHASE_ANALYSIS]


def test_image_preparation_is_shared_setup_route_without_duplicate_or_writes(method_window):
    window = method_window

    def files():
        return {key: _files(root) for key, root in _roots(window).items()}

    original = files()
    logical = ("phase_model", "normalisation")
    category, display = window.sidebar.display_selection(*logical)
    assert category == "setup"
    assert (display, "Image correction") in _visible(window, "setup")
    assert display not in window.sidebar._sections["calibration"]._rows
    assert "starts no calculation" in window.sidebar._sections[category].get_row(display).help_text
    workspace = window.phase_model_workspace
    shown = []
    workspace.studio_page = SimpleNamespace(
        native_numbered_steps=True,
        native_steps=SimpleNamespace(show_section=shown.append),
        show=lambda: None,
    )
    for active in ("phase_model", "colour_path"):
        if active == "phase_model":
            _phase_choice(_roots(window)["series-a"])
        else:
            save_series_method(_roots(window)["series-a"],
                               MethodSelection(active=active, colour_calibration="colour"))
        window._refresh_method_state()
        baseline = files()
        window._select_row(logical)
        assert window.content_stack.currentWidget() is workspace
        assert workspace._requested_section == shown[-1] == "normalisation"
        assert not window.run_combo.isVisible()
        assert files() == baseline
        assert _visible(window, "setup").count((display, "Image correction")) == 1
    assert original.keys() == files().keys()


def test_every_old_logical_row_id_still_resolves_to_its_row(method_window):
    window = method_window
    old = [
        ("calibration", item) for item in ("color", "mass", "mass_optuna", "default_mass", "reset", "delete")
    ] + [
        ("analysis", item) for item in COLOUR_ANALYSIS
    ] + [
        ("phase_model", "training"),
        ("phase_model", "mass_from_phase"),
        ("mass_analysis", "frozen"),
        ("mass_analysis", "review"),
        ("jobs", "overview"),
        ("image_selection", "plan"),
        ("preprocessing", "protocols"),
    ]
    new = [
        ("phase_model", "normalisation"),
        ("phase_model", "calibration_review"),
        ("method", "use_for_series"),
        ("jobs", "new_analysis"),
    ]
    for logical in old + new:
        category, display = window.sidebar.display_selection(*logical)
        row = window.sidebar._sections[category].get_row(display)
        assert row is not None and (row.action, row.checkbox_id) == logical, logical
    hidden = window.sidebar._sections["analysis"].get_row("mass_analysis:review")
    assert hidden.isHidden() and hidden.isEnabled()


def test_hidden_phase_review_row_stays_routable(method_window, monkeypatch):
    window = method_window
    opened = []
    monkeypatch.setattr(window, "_open_existing_phase_review", lambda: opened.append("review"))
    window._select_row(("mass_analysis", "review"))
    assert opened == ["review"]
    assert window.sidebar.get_selected() == ("mass_analysis", "review")


@pytest.mark.parametrize(
    "logical, section",
    [
        (("phase_model", "normalisation"), "normalisation"),
        (("phase_model", "training"), "phase_model"),
        (("method", "use_for_series"), "use"),
    ],
)
def test_calibration_rows_show_their_section_of_the_one_page(method_window, logical, section):
    window = method_window
    workspace = window.phase_model_workspace
    shown = []
    workspace.studio_page = SimpleNamespace(
        native_numbered_steps=True,
        native_steps=SimpleNamespace(show_section=shown.append),
        show=lambda: None,
    )
    window._select_row(logical)
    assert window.content_stack.currentWidget() is workspace
    assert window.sidebar.display_selection(*logical)[0] == (
        "setup" if logical == ("phase_model", "normalisation") else "calibration")
    assert workspace._requested_section == section and shown[-1] == section
    assert window.status_header.line.isVisible()
    assert not window.run_combo.isVisible()  # A series page.


def test_one_page_shows_one_section_with_tabs_and_the_phase_review_button(setup):  # noqa: F811
    steps = setup.steps
    steps.show()
    for name in ("normalisation", "phase_model", "use"):
        steps.show_section(name)
        assert steps.section == name
        assert [key for key, widget in steps.sections.items() if not widget.isHidden()] == [name]
    with pytest.raises(ValueError):
        steps.show_section("mass")
    tabs = steps.phase_tabs
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["Basis", "Label phases", "Blindly validate phase model", "Train"]
    review = steps.groups[3]
    assert review.isAncestorOf(steps.phase_review_button)
    assert steps.groups[2].isAncestorOf(setup.studio.open_model_correction)
    assert steps.phase_review_button.isHidden()
    assert not steps.phase_review_button.isEnabled()  # No window callback in this fixture.
    assert steps.groups[4].isAncestorOf(steps.advanced) and steps.advanced.isCheckable()
    assert steps.advanced.isHidden()
    for area in steps.findChildren(QScrollArea):
        assert not any(isinstance(parent, QScrollArea) for parent in _ancestors(area)), area


class _SavedResultPage(QWidget):
    def __init__(self):
        super().__init__()
        self.views = []

    def show_view(self, view):
        self.views.append(view)


def test_calibration_review_and_results_share_the_saved_result_page(method_window):
    window = method_window
    page = _SavedResultPage()
    window.content_stack.addWidget(page)
    window.ensure_mass_analysis = lambda: page
    window._select_row(("phase_model", "calibration_review"))
    assert window.content_stack.currentWidget() is page and page.views == ["review"]
    assert window.sidebar.display_selection("phase_model", "calibration_review")[0] == "calibration"
    window._select_row(("mass_analysis", "frozen"))
    assert window.content_stack.currentWidget() is page and page.views == ["review", "results"]
    assert window.status_header.line.text().endswith(" · Whole series (0 runs)")


def test_new_analysis_row_and_start_analysis_open_the_page(method_window, monkeypatch):
    window = method_window
    shown = []

    class Jobs(QWidget):
        new_analysis = SimpleNamespace(set_context=lambda **kwargs: shown.append(kwargs))

        def show_view(self, name):
            shown.append(name)

    jobs = Jobs()
    window.content_stack.addWidget(jobs)
    monkeypatch.setattr(window, "ensure_jobs", lambda: jobs)
    window.show_jobs(start_analysis=True)
    assert window.sidebar.get_selected() == ("jobs", "new_analysis")
    assert window.content_stack.currentWidget() is jobs
    assert shown[0]["method_text"].startswith("series-a · Method not chosen") and shown[1] == "new_analysis"
    window.show_jobs()
    assert window.sidebar.get_selected() == ("jobs", "overview") and shown[-1] == "jobs"


def test_saving_the_same_method_choice_twice_gives_identical_series_toml(
    method_window, monkeypatch, qtbot,
):
    window = method_window
    root = _roots(window)["series-a"]
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    window._refresh_method_state(force=True)
    choice = MethodSelection(active="colour_path", colour_calibration="colour")
    panel = window.method_panel
    panel.set_selection(choice)

    def save_current_choice():
        assert panel.calls.context == (str(root), choice, (root / "series.toml").read_bytes())
        with qtbot.waitSignal(panel.calls.action_finished, timeout=10000) as completed:
            window._save_method_choice(choice)
            assert not warnings, warnings
        qtbot.waitUntil(
            lambda: panel.calls.thread is None and not panel.calls.timer.isActive(),
            timeout=10000,
        )
        assert not warnings, warnings
        assert completed.args[0]["current"] and not completed.args[0]["error"]

    save_current_choice()
    first = (root / "series.toml").read_bytes()
    before = _files(root)
    save_current_choice()
    assert (root / "series.toml").read_bytes() == first
    assert _files(root) == before
    assert window._saved_method.active == "colour_path"


def test_saved_result_page_has_two_views_and_no_scroll_area_inside_another(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module

    root = tmp_path / "series-a"
    root.mkdir()
    monkeypatch.setattr(module, "configured_series", lambda: (("series-a", "series-a", root),))
    page = module.MassAnalysisWorkspace(series="series-a", project_root=root)
    qtbot.addWidget(page)
    page.show()
    saved = page.saved_results
    # The page is the one scroll area; the figure tabs hold their figures directly.
    assert page.findChildren(QScrollArea) == [page.scroll]
    assert page.scroll.isAncestorOf(saved)
    assert [type(saved.views.widget(i)).__name__ for i in range(saved.views.count())] == [
        "_FullscreenPixmap", "_FullscreenPixmap", "_FullscreenPixmap", "_FullscreenPixmap",
        "QTableWidget", "_FullscreenPixmap", "_FullscreenPixmap"]
    assert saved.views.widget(saved.photo_index) is saved.photo
    content = page.scroll.widget().layout()
    assert content.stretch(content.indexOf(saved)) == 1
    page.show_view("review")
    assert page.title.text() == "Calibration review"
    assert not saved.setup_toggle.isVisible() and not saved.export_toggle.isVisible()
    assert not saved.export_body.isVisible() and not saved.views.isVisible()
    assert saved.review_panels.isVisible() and saved.review_header.isVisible()
    assert not page.model_controls.isVisible()
    assert content.indexOf(saved.review_header) < content.indexOf(page.approval)
    page.show_view("results")
    assert page.title.text() == "Results and export"
    assert saved.setup_toggle.isVisible() and saved.export_toggle.isVisible()
    assert saved.export_body.isVisible()
    assert saved.views.isVisible() and not saved.review_panels.isVisible()
    assert page.model_controls.isVisible() and not saved.review_header.isVisible()
    assert [button.text() for button in saved.export_buttons] == [
        "Export MP4", "Export current figure", "Export CSV", "Export figures + numerical fields"]
    with pytest.raises(ValueError):
        page.show_view("elsewhere")
    page._stop_lookups()


def test_native_pages_do_not_force_the_window_size(qtbot, tmp_path):
    from darsia.fluidflower.native_phase_subpage import NativePhaseModelStudioSubpage

    page = NativePhaseModelStudioSubpage("ffac", tmp_path)
    qtbot.addWidget(page)
    # The steps scroll themselves; the standalone 700 px minimum is not kept.
    assert page.minimumHeight() == 0


def test_header_line_wraps_instead_of_widening_the_window(method_window):
    line = method_window.status_header.line
    assert line.wordWrap()
    line.setText("FF_AC · " + "Phase-map method · " * 20 + "Whole series (39 runs)")
    assert line.minimumSizeHint().width() < 600
