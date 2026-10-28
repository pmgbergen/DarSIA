"""Formatting parity only: no spectra, color paths, graphics or scientific inputs."""

from __future__ import annotations

import ast

import pytest

from darsia.signals.color.label_color_path_map import LabelColorPathMap
from darsia.signals.color.label_color_spectrum_map import LabelColorSpectrumMap

# Exact Spectrum method bodies from source SHA355602086fafa4eb98660ca5bd15caab445558542ebd90019ab71e5ad083c2c2.
_ORIGINAL_METHODS = 'def __repr__(self) -> str:\n    str_repr = ""\n    for label, color_spectrum in self.items():\n        str_repr += f"Label {label}: {color_spectrum}\\n"\n    return str_repr\n\ndef __str__(self) -> str:\n    str_str = ""\n    for label, color_spectrum in self.items():\n        str_str += f"Label {label}: {str(color_spectrum)}\\n"\n    return str_str'

def _original_method(name):
    tree = ast.parse(_ORIGINAL_METHODS)
    node = next(item for item in tree.body if item.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<original-format-method>", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("method", ["__repr__", "__str__"])
def test_empty_and_insertion_order_equal_original_methods(method):
    assert LabelColorSpectrumMap is not LabelColorPathMap
    empty = LabelColorSpectrumMap()
    assert getattr(empty, method)() == _original_method(method)(empty) == ""
    mapping = LabelColorSpectrumMap({8: "first", 2: "second", 5: "third"})
    assert getattr(mapping, method)() == _original_method(method)(mapping)
    assert getattr(mapping, method)() == "Label 8: first\nLabel 2: second\nLabel 5: third\n"
    assert list(mapping) == [8, 2, 5]


@pytest.mark.parametrize("method", ["__repr__", "__str__"])
def test_custom_value_str_format_and_mapping_subclass_equal_original(method):
    events = []

    class Value:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            events.append(("str", self.name))
            return "str:" + self.name

        def __format__(self, specification):
            events.append(("format", self.name, specification))
            return "format:" + self.name + ":" + specification

    class ValueSubclass(Value):
        pass

    class ReverseItems(LabelColorSpectrumMap):
        def items(self):
            return reversed(list(dict.items(self)))

    mapping = ReverseItems({8: Value("first"), 2: ValueSubclass("second")})
    actual = getattr(mapping, method)()
    actual_events = list(events)
    events.clear()
    expected = _original_method(method)(mapping)
    assert actual == expected
    assert actual_events == events
    if method == "__repr__":
        assert actual == "Label 2: format:second:\nLabel 8: format:first:\n"
        assert events == [("format", "second", ""), ("format", "first", "")]
    else:
        assert actual == "Label 2: str:second\nLabel 8: str:first\n"
        assert events == [("str", "second"), ("str", "first")]
    assert list(mapping) == [8, 2]


@pytest.mark.parametrize("method", ["__repr__", "__str__"])
def test_spectrum_passes_same_self_to_unbound_path_method(monkeypatch, method):
    mapping = LabelColorSpectrumMap({3: object()})
    seen = []

    def selected(self):
        seen.append(self)
        return "selected-owner"

    monkeypatch.setattr(LabelColorPathMap, method, selected)
    assert getattr(mapping, method)() == "selected-owner"
    assert seen == [mapping]
    assert seen[0] is mapping
    assert type(mapping) is LabelColorSpectrumMap


def test_storage_and_construction_interfaces_remain_distinct():
    assert hasattr(LabelColorSpectrumMap, "save")
    assert hasattr(LabelColorSpectrumMap, "load")
    assert not hasattr(LabelColorSpectrumMap, "save_csv")
    assert hasattr(LabelColorPathMap, "save_csv")
    assert hasattr(LabelColorPathMap, "load_csv")
    assert hasattr(LabelColorPathMap, "refine")
    assert not hasattr(LabelColorPathMap, "save")
