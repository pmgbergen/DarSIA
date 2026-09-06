"""End-to-end tests for the merged ``int_rows`` settings widget.

``create_int_rows_input`` replaced the former ``create_int_group_list_input``
(non-pair, ``list[list[int]]``) and ``create_int_list_map_input`` (``pair=True``,
``dict[int, list[int]]``, optional ``flatten_in_section``). Each test drives the
real ``create_setting_edit`` dispatch and the real save pass in
``_sync_settings_inputs_to_config_dict``.

The widget pre-fills via ``QTimer.singleShot``, so ``qapp.processEvents()`` must
run before ``rows`` is inspected.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QFormLayout, QWidget

from darsia.gui.ui.settings import SettingsFactory


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _MockMainWindow:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.settings_inputs = {}
        self.logs = []

    def print_log(self, message, *_a, **_k):
        self.logs.append(message)


def _build(factory, setting, keepalive, qapp):
    """Render through the real dispatch; run the deferred pre-fill; return result."""
    parent = QWidget()
    form = QFormLayout(parent)
    keepalive.append((parent, form))
    result = factory.create_setting_edit(setting, form_context={"form": form})
    qapp.processEvents()
    return result


def test_non_pair_list_of_int_lists_round_trip(qapp):
    config_dict = {"labeling": {"unite_labels": [[3, 5], [8]]}}
    factory = SettingsFactory(_MockMainWindow(config_dict))
    keep: list = []

    setting = {
        "key": "labeling.unite_labels",
        "name": "Unite labels",
        "type": "int_rows",
        "placeholder": "e.g. 3, 5, 8",
    }
    _, res = _build(factory, setting, keep, qapp)

    assert res["int_rows"] is True
    assert res["pair"] is False
    assert [e.text() for e in res["rows"]] == ["3, 5", "8"]

    # edit: row 0 stays valid (space-separated ok); row 1 becomes non-integer
    res["rows"][0].setText("3 5 7")
    res["rows"][1].setText("not ints")

    factory.main_window.settings_inputs = {
        setting["key"]: factory._wrap_multi_row_result(res)
    }
    factory._sync_settings_inputs_to_config_dict()

    assert config_dict["labeling"]["unite_labels"] == [[3, 5, 7]]  # row 1 skipped
    assert any("not all-integer" in m for m in factory.main_window.logs)


def test_pair_int_to_list_map_round_trip(qapp):
    config_dict = {"x": {"m": {"1": [3, 5]}}}
    factory = SettingsFactory(_MockMainWindow(config_dict))
    keep: list = []

    setting = {"key": "x.m", "name": "Map", "type": "int_rows", "pair": True}
    _, res = _build(factory, setting, keep, qapp)

    assert res["pair"] is True
    assert "flatten_in_section" not in res
    assert [(k.text(), v.text()) for k, v in res["rows"]] == [("1", "3, 5")]

    res["rows"][0][1].setText("3 5 9")
    factory.main_window.settings_inputs = {
        setting["key"]: factory._wrap_multi_row_result(res)
    }
    factory._sync_settings_inputs_to_config_dict()

    assert config_dict["x"]["m"] == {1: [3, 5, 9]}


def test_pair_flatten_in_section_writes_and_prunes_subtables(qapp):
    config_dict = {"facies": {"0": {"labels": [3]}, "1": {"labels": [9]}}}
    factory = SettingsFactory(_MockMainWindow(config_dict))
    keep: list = []

    setting = {
        "key": "facies.facies_to_labels_map",
        "name": "Facies groups",
        "type": "int_rows",
        "pair": True,
        "flatten_in_section": True,
    }
    _, res = _build(factory, setting, keep, qapp)

    assert res["flatten_in_section"] is True
    assert res["section"] == "facies"
    assert sorted((k.text(), v.text()) for k, v in res["rows"]) == [
        ("0", "3"),
        ("1", "9"),
    ]

    # drop id 1 (blank its key) and extend id 0
    res["rows"][0][1].setText("3 4")
    res["rows"][1][0].setText("")

    factory.main_window.settings_inputs = {
        setting["key"]: factory._wrap_multi_row_result(res)
    }
    factory._sync_settings_inputs_to_config_dict()

    assert config_dict["facies"] == {"0": {"labels": [3, 4]}}  # [facies.1] pruned


def test_requires_form_context(qapp):
    factory = SettingsFactory(_MockMainWindow({}))
    with pytest.raises(ValueError):
        factory.create_int_rows_input(
            {"key": "x.y", "type": "int_rows"}, form_context=None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
