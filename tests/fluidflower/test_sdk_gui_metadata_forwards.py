"""Exact pure dictionary/number forwarding without constructing GUI windows."""
from __future__ import annotations
import ast
from copy import deepcopy
import math
from pathlib import Path
import subprocess
from typing import Any
import pytest

ORIGINAL = "65c0e85015700f764b2c4f5fed6bb558490ef7f0"

def pure(path, name, *, class_name=None, original=False, forwards=None):
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(["git", "show", ORIGINAL + ":" + path], cwd=repo).decode() if original else (repo / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    scope = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    fn = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"math": math, "Any": Any}
    if forwards is not None:
        offset = 1 if ast.get_docstring(fn) else 0
        route = fn.body.pop(offset)
        assert isinstance(route, ast.ImportFrom)
        expected_module, expected_name, target = forwards
        assert route.module == expected_module
        assert route.names[0].name == expected_name
        ns[expected_name] = target
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), ns)
    return ns[name]

@pytest.mark.parametrize("method", ["get_value", "set_value"])
def test_dictionary_forwards_keep_original_behavior_with_fake_owner(method):
    canonical = pure("src/darsia/gui/ui/settings.py", method, class_name="SettingsFactory")
    owner_type = type("FakeSettingsFactory", (), {method: canonical})
    old = pure("src/darsia/gui/gui_main.py", method, class_name="MainWindow", original=True)
    new = pure("src/darsia/gui/gui_main.py", method, class_name="MainWindow", forwards=("darsia.gui.ui.settings", "SettingsFactory", owner_type))
    owner = object()  # Original methods do not access widget state.
    for initial, key in [({"a": {"b": 4}}, "a.b"), ({}, "a.b"), ({"a": 0}, "a.b"), ({"": 3}, "")]:
        left, right = deepcopy(initial), deepcopy(initial)
        args = (key,) if method == "get_value" else (key, "new value")
        def observe(fn, data):
            try: return ("return", fn(owner, data, *args))
            except Exception as error: return ("error", type(error), str(error))
        assert observe(new, right) == observe(old, left)
        assert right == left

@pytest.mark.parametrize("value", [None, True, False, "1,25", " 1.25 ", "bad", -0.0, float("inf"), float("nan")])
def test_comma_number_forward_matches_original(value):
    old = pure("src/darsia/fluidflower/ui/common.py", "_gui_float", original=True)
    target = pure("src/darsia/fluidflower/analysis_domains.py", "_number")
    new = pure("src/darsia/fluidflower/ui/common.py", "_gui_float", forwards=("darsia.fluidflower.analysis_domains", "_number", target))
    assert new(value) == old(value)
    if old(value) == 0.0:
        assert math.copysign(1, new(value)) == math.copysign(1, old(value))

def test_comma_number_preserves_real_string_conversion_failure():
    class Broken:
        def __str__(self): raise RuntimeError("original string failure")
    old = pure("src/darsia/fluidflower/ui/common.py", "_gui_float", original=True)
    target = pure("src/darsia/fluidflower/analysis_domains.py", "_number")
    new = pure("src/darsia/fluidflower/ui/common.py", "_gui_float", forwards=("darsia.fluidflower.analysis_domains", "_number", target))
    for fn in (old, new):
        with pytest.raises(RuntimeError, match="original string failure"): fn(Broken())
