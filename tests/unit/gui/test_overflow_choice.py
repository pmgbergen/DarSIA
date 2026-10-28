"""Free gas vs overflow (N141 spec 2.3, Part 2): the calibration's overflow use.

Local calibration documents only; nothing is written and no setting changes.
"""

import json

from darsia.fluidflower.mass_calibration_display import OverflowChoicePage, read_overflow_use
from darsia.fluidflower.series_method import MethodSelection, save_series_method

from .test_native_series_method import _roots, method_window  # noqa: F401


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_read_overflow_use_from_the_referenced_calibration(tmp_path):
    _write(tmp_path / "frozen.json", {"frozen": True, "overflow_in_objective": False})
    _write(tmp_path / "off.json", {"overflow_mode": "off"})
    _write(tmp_path / "nested.json", {"calibration": {"overflow_mode": "constrain"}})
    _write(tmp_path / "silent.json", {"parameters": {}})
    assert read_overflow_use("frozen.json", tmp_path)[0] == "compare"
    assert read_overflow_use("off.json", tmp_path) == ("off", "The referenced calibration records overflow_mode = off.")
    assert read_overflow_use("nested.json", tmp_path)[0] == "constrain"
    assert read_overflow_use("silent.json", tmp_path)[0] is None
    assert read_overflow_use(None, tmp_path) == (None, "No phase-model mass calibration is referenced in series.toml.")
    mode, text = read_overflow_use("missing.json", tmp_path)
    assert mode is None and text.startswith("The referenced calibration could not be read")


def test_page_shows_off_and_compare_read_only_and_steer_not_available(qtbot):
    page = OverflowChoicePage()
    qtbot.addWidget(page)
    assert [button.isEnabled() for button in (*page.choice.values(), page.steer)] == [False, False, False]
    assert "reviewed overflow" in page.steer.text()
    page.set_use("compare", "Compared.")
    assert page.choice["compare"].isChecked() and not page.choice["off"].isChecked()
    page.set_use("off", "Off.")
    assert page.choice["off"].isChecked() and not page.choice["compare"].isChecked()
    page.set_use("constrain", "Constrained.")
    assert not any(button.isChecked() for button in (*page.choice.values(), page.steer))
    assert "legacy comparison method" in page.detail.text()
    page.set_use(None, "Nothing referenced.")
    assert page.mode is None and page.detail.text() == "Nothing referenced."


def test_overflow_row_sits_between_phase_model_and_mass_calibration(method_window, qtbot):  # noqa: F811
    window = method_window
    qtbot.waitUntil(lambda: not window._series_loading)
    root = _roots(window)["series-a"]
    _write(root / "fits/frozen.json", {"frozen": True, "overflow_in_objective": False})
    save_series_method(root, MethodSelection(active="phase_model", model_version="m",
                                             mass_calibration="fits/frozen.json"))
    window._refresh_method_state(force=True)
    before = (root / "series.toml").read_bytes()
    section = window.sidebar._sections["calibration"]
    assert "phase_model:overflow" not in section._rows
    keys = list(section._rows)
    assert keys.index("phase_model:training") < keys.index("phase_model:mass_from_phase")
    window._select_row(("phase_model", "overflow"))
    assert window.content_stack.currentWidget() is window.mass_calibration_display
    assert window.overflow_page is window.mass_calibration_display.overflow_choice
    assert window.overflow_page.choice["off"].isEnabled()
    assert window.overflow_page.choice["compare"].isEnabled()
    assert window.overflow_page.choice["compare"].isChecked()
    assert window.status_header.line.text().endswith(" · Whole series (0 runs)")
    assert (root / "series.toml").read_bytes() == before


def test_frozen_overflow_target_is_distinct_from_historical_compare_only(tmp_path):
    _write(tmp_path / 'target.json', dict(format='darsia-local-overflow-frozen-calibration-v1',
        variant_f=True, calibration=dict(overflow_in_objective=False)))
    assert read_overflow_use('target.json', tmp_path)[0] == 'steer'
