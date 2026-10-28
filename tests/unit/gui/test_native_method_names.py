"""The method names chosen in N141 (names A), and grouping that never reads them.

The sidebar headers, the method radio buttons, the badge and the header status
line use the new names; the internal keys and series.toml stay. Local
manifests only; no calculation.
"""

from darsia.fluidflower.series_method import (
    METHOD_DESCRIPTIONS,
    METHOD_HEADERS,
    METHOD_NAMES,
    METHODS,
    SeriesMethodPanel,
)
from darsia.gui.ui.sidebar import GroupHeaderLabel

from .test_native_series_method import _phase_choice, _roots, method_window  # noqa: F401


def _headers(window, category):
    layout = window.sidebar._sections[category].items_layout
    widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
    return [widget for widget in widgets if isinstance(widget, GroupHeaderLabel)]


def test_names_cover_exactly_the_two_internal_method_keys():
    for table in (METHOD_NAMES, METHOD_HEADERS, METHOD_DESCRIPTIONS):
        assert set(table) == set(METHODS)
    assert METHOD_NAMES == {"colour_path": "Direct colour method", "phase_model": "Phase-map method"}
    assert METHOD_HEADERS["colour_path"] == "Direct colour method (DarSIA original)"
    assert METHOD_HEADERS["phase_model"] == "Phase-map method"


def test_sidebar_has_one_common_list_without_parallel_method_headers(method_window):
    window = method_window
    for category in ("calibration", "analysis", "comparison"):
        assert all(not header.text() and header.isHidden() for header in _headers(window, category))
        assert len(window.sidebar._sections[category].groups) == 1
    rows = window.sidebar._sections["calibration"]._rows
    assert sum(row.label == "Mass calibration" and not row.isHidden() for row in rows.values()) == 1


def test_method_availability_uses_logical_keys_not_labels_or_headers(method_window):
    window = method_window
    root = _roots(window)["series-a"]
    _phase_choice(root)
    window._refresh_method_state(force=True)
    section = window.sidebar._sections["calibration"]
    row = section.get_row("calibration_controls:interactive")
    original_label = row.label
    row.label = "TESTONLY renamed label"
    for header in _headers(window, "calibration"):
        header.setText("TESTONLY renamed header")
    order = list(section._rows)
    window._capture_method_groups()
    window._order_method_groups()
    assert list(section._rows) == order
    assert not row.isEnabled() and not row.isHidden()
    assert section.get_row("color").isEnabled()
    assert section.get_row("phase_model:training").isEnabled()
    row.label = original_label


def test_badge_and_header_line_name_the_chosen_method(method_window):
    window = method_window
    _phase_choice(_roots(window)["series-a"])
    window._refresh_method_state()
    assert window.method_badge.text().startswith("Method: phase-map method · model ")
    window._select_row(("calibration", "color"))
    series = window.phase_model_workspace.series_selector.currentText()
    assert window.status_header.line.text().startswith(f"{series} · Phase-map method · model ")


def test_method_buttons_show_the_new_names_with_their_explanation(qtbot):
    panel = SeriesMethodPanel()
    qtbot.addWidget(panel)
    assert "Colour signal + SimpleFlash" in panel.colour_path_button.text()
    assert "Titration + dynamic phase maps" in panel.phase_model_button.text()
    assert "original" in panel.colour_path_button.toolTip().lower()
    assert "Expert Knowledge" in panel.phase_model_button.toolTip()
