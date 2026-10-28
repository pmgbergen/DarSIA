"""The header status line (N141): one line on every page, details on demand.

The run selector shows on run pages only, and no page sits in a header scroll
area. Local manifests only; nothing is written or calculated.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea

from darsia.fluidflower.calibration_header import row_category, status_line
from darsia.fluidflower.row_scope import scope_summary

from .test_native_row_scope import scope_window  # noqa: F401
from .test_native_series_method import method_window  # noqa: F401


def _files(window):
    roots = window.phase_model_workspace.series_roots.values()
    return {str(p): p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}


def _ancestors(widget):
    parent = widget.parentWidget()
    while parent is not None:
        yield parent
        parent = parent.parentWidget()


def test_status_line_joins_series_method_and_scope_and_maps_row_categories():
    parts = ["Phase-map method", "model m1", "calibration k1"]
    assert status_line("FF_AC", parts, "Whole series (39 runs)") == (
        "FF_AC · Phase-map method · model m1 · calibration k1 · Whole series (39 runs)"
    )
    assert status_line("FF_AC", ["Method not chosen"], "") == "FF_AC · Method not chosen"
    assert [row_category(action) for action in ("phase_model", "method", "jobs", "mass_analysis")] == [
        "calibration", "calibration", "analysis", "analysis"]
    assert row_category("image_selection") == "setup"
    assert row_category("comparison") == "comparison"


def test_one_header_line_with_the_former_lines_as_details(scope_window, qtbot):
    w = scope_window
    selector = w.run_combo
    w._select_row(("calibration", "color"))
    qtbot.wait(20)
    header = w.status_header
    assert header.line.isVisible()
    assert header.line.text() == status_line(
        "series-a", ["Method not chosen"], scope_summary(w.scope_context, w.current_row_scope))
    assert header.line.text().startswith("series-a · Method not chosen · ")
    assert not header.details.isVisible() and not w.setup_series_label.isVisible()
    for widget in (w.method_badge, w.analysis_method_badge, w.frozen_reference_line,
                   w.scope_label, w.rig_label):
        assert header.details.isAncestorOf(widget)
    assert w.method_badge.text() in header.line.toolTip()
    assert w.scope_label.text() in header.line.toolTip()
    qtbot.mouseClick(header.details_button, Qt.MouseButton.LeftButton)
    assert header.details.isVisible() and w.method_badge.isVisible() and w.scope_label.isVisible()
    # A run page keeps the original selector, enabled.
    assert w.run_combo is selector and w.run_combo.isVisible() and w.run_combo.isEnabled()
    # No page is wrapped in a header scroll area any more.
    assert not any(isinstance(parent, QScrollArea) for parent in _ancestors(w.content_stack))
    assert w.findChild(QScrollArea, "calibrationRowScroll") is None


def test_run_selector_only_on_run_pages_and_no_writes(scope_window, qtbot):
    w = scope_window
    before = _files(w)
    for logical, run_page in [
        (("calibration", "color"), True),
        (("phase_model", "normalisation"), False),
        (("phase_model", "training"), False),
        (("phase_model", "mass_from_phase"), False),
        (("method", "use_for_series"), False),
        (("analysis", "mass"), True),
        (("preprocessing", "protocols"), True),
        (("start", "series"), False),
    ]:
        w._select_row(logical)
        qtbot.wait(20)
        start = logical[0] == "start"
        assert w.status_header.line.isVisible() is not start, logical
        assert w.run_combo.isVisible() is run_page, logical
        assert w.series_header.isVisible() is run_page, logical
        if not start and not run_page:
            assert w.status_header.line.text().endswith(" · Whole series (2 runs)"), logical
        assert w.content_stack.currentWidget() is not None
    assert _files(w) == before


def test_painting_hides_only_the_global_selector_across_refreshes_and_restores_it(scope_window, qtbot):
    w = scope_window
    before = _files(w)
    selector, selected = w.run_combo, w.run_combo.currentData()
    for logical, visible in [
        (("geometry_facies", "layers"), False),
        (("geometry_facies", "rig_frame"), True),
        (("geometry_facies", "layers"), False),
        (("geometry_facies", "export"), True),
        (("preprocessing", "protocols"), True),
        (("geometry_facies", "layers"), False),
        (("calibration", "color"), True),
        (("geometry_facies", "layers"), False),
    ]:
        w._select_row(logical)
        # Both refresh routes run during navigation and config updates. They
        # must not bring back a second chooser while painting is selected.
        for _ in range(3):
            w.status_header.refresh()
            w._refresh_scope_display()
            qtbot.wait(1)
        assert w.run_combo is selector and w.run_combo.currentData() == selected
        assert w.series_header.isVisible() is visible, logical
        for widget in (w.run_label, w.run_combo, w.run_path):
            assert widget.isVisible() is visible, logical
        assert w.status_header.line.isVisible(), logical
    assert _files(w) == before
