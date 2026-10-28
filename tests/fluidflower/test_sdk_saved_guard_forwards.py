"""Exact guard/stat/type forwards tested as pure code, without loading Studio."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
import pytest

ORIGINAL = "65c0e85015700f764b2c4f5fed6bb558490ef7f0"

def pure(path, name, *, original=False, class_name=None, parent=None, forward=None):
    repo = Path(__file__).resolve().parents[2]
    text = subprocess.check_output(["git", "show", ORIGINAL + ":" + path], cwd=repo).decode() if original else (repo / path).read_text(encoding="utf-8")
    module = ast.parse(text)
    scope = module.body
    if class_name: scope = next(n for n in scope if isinstance(n, ast.ClassDef) and n.name == class_name).body
    if parent: scope = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == parent).body
    fn = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []  # Compare actual method body, preserving caller arguments.
    namespace = {"json": json, "Path": Path, "Any": Any}
    if forward:
        first = 1 if ast.get_docstring(fn) else 0
        if isinstance(fn.body[first], ast.ImportFrom):
            imported = fn.body.pop(first)
            assert len(imported.names) == 1 and imported.names[0].name == forward[0]
        namespace[forward[0]] = forward[1]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


def outcome(fn, *args):
    try: return ("return", fn(*args))
    except Exception as error: return ("error", type(error), str(error))

@pytest.mark.parametrize("source", ["{}", '{"unicode":"β"}', "[]", "null", "2", "{bad"])
def test_json_object_readers_preserve_real_content_and_exact_errors(tmp_path, source):
    path = tmp_path / "input.json"
    path.write_text("\ufeff" + source, encoding="utf-8")
    helper = "src/darsia/fluidflower/phase_saved_review_metadata.py"
    actual_value = pure("src/darsia/fluidflower/execution.py", "_read_json_object")
    actual_type = pure(helper, "read_model_json")
    old_value = pure("src/darsia/fluidflower/execution.py", "read", original=True, parent="_read_json")
    old_value.__globals__["path"] = path
    old_type = pure("src/darsia/fluidflower/phase_learning.py", "_read_json", original=True)
    assert outcome(actual_value, path) == outcome(old_value)
    assert outcome(actual_type, path) == outcome(old_type, path)
    new_type = pure("src/darsia/fluidflower/phase_learning.py", "_read_json", forward=("read_model_json", actual_type))
    assert outcome(new_type, path) == outcome(old_type, path)

@pytest.mark.parametrize("entry,expected", [
    ({"value": [1, "x"], "python_type": "tuple"}, ("return", (1, "x"))),
    ({"value": [], "python_type": "tuple"}, ("return", ())),
    ({"value": True, "python_type": "bool"}, ("return", True)),
    ({"value": True, "python_type": "int"}, ("error", TypeError, "Saved metadata DATA changed its original Python type")),
    ({"value": None, "python_type": "NoneType"}, ("return", None)),
    ({"value": {}, "python_type": "dict"}, ("return", {})),
    ({"value": 1.5, "python_type": "str"}, ("error", TypeError, "Saved metadata DATA changed its original Python type")),
    ({"value": "x"}, ("error", KeyError, "'python_type'")),
    ({}, ("error", KeyError, "'value'")),
    ({"value": None, "python_type": "tuple"}, ("error", TypeError, "'NoneType' object is not iterable")),
])
def test_narrow_typed_data_keeps_original_coercion_and_rejections(entry, expected):
    helper = pure("src/darsia/fluidflower/phase_saved_review_metadata.py", "typed_saved_data_value")
    assert outcome(helper, entry) == expected
    for filename in ("phase_saved_candidate_admission.py", "phase_saved_promotions.py", "phase_saved_transfer_context.py", "phase_saved_transfer_review.py"):
        path = "src/darsia/fluidflower/" + filename
        new = pure(path, "_typed", forward=("typed_saved_data_value", helper))
        assert outcome(new, entry) == expected
        if expected[0] == "return" and entry["python_type"] in {"dict", "NoneType", "bool"}:
            assert new(entry) is entry["value"]


def test_file_stamp_keeps_all_four_fields_and_no_path_coercion(tmp_path):
    old = pure("src/darsia/fluidflower/phase_blind_review_policy.py", "_signature", original=True)
    helper = pure("src/darsia/fluidflower/phase_saved_review_metadata.py", "metadata_file_stamp")
    path = tmp_path / "metadata.json"
    path.write_text("{}", encoding="utf-8")
    for selected in (path, tmp_path / "missing", None, "not a Path"):
        assert outcome(helper, selected) == outcome(old, selected)
    class FakePath:
        def stat(self):
            return SimpleNamespace(st_size=7, st_mtime_ns=11, st_ctime_ns=13, st_ino=17)
    assert helper(FakePath()) == (7, 11, 13, 17)
