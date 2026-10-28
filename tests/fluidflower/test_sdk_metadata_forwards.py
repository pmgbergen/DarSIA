"""Local parity of SDK metadata forwards; application imports are inspected, not run."""
from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping

import pytest

ORIGINAL_COMMIT = "65c0e85015700f764b2c4f5fed6bb558490ef7f0"
CASES = [('src/darsia/fluidflower/calibration.py', '_json_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/calibration_review.py', '_json_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/mass_analysis.py', '_json_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/phase_interpretation.py', '_json_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/post_analysis.py', '_json_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/wasserstein.py', '_json_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/calibration.py', '_number', 'src/darsia/fluidflower/mass_inputs.py', '_number', 'number'), ('src/darsia/fluidflower/indicator_response.py', '_number', 'src/darsia/fluidflower/mass_inputs.py', '_number', 'number'), ('src/darsia/fluidflower/mass_analysis.py', '_number', 'src/darsia/fluidflower/mass_inputs.py', '_number', 'number'), ('src/darsia/fluidflower/overflow_workflow.py', '_review_float', 'src/darsia/fluidflower/mass_inputs.py', '_number', 'number'), ('src/darsia/fluidflower/post_analysis.py', '_number', 'src/darsia/fluidflower/mass_inputs.py', '_number', 'number'), ('src/darsia/fluidflower/tools/phase_review_history.py', '_finite', 'src/darsia/fluidflower/mass_inputs.py', '_number', 'number'), ('src/darsia/fluidflower/wasserstein.py', '_number', 'src/darsia/fluidflower/mass_inputs.py', '_number', 'number'), ('src/darsia/fluidflower/facies_model.py', '_json_digest', 'src/darsia/fluidflower/facies_learning.py', '_json_digest', 'UTF8_mapping_JSON'), ('src/darsia/fluidflower/frame_quality.py', '_canonical_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/image_correction.py', '_json_digest', 'src/darsia/fluidflower/analysis_domains.py', '_json_digest', 'ASCII_JSON'), ('src/darsia/fluidflower/mass_analysis.py', '_resolved_path', 'src/darsia/fluidflower/mass_inputs.py', '_resolved_path', 'path'), ('src/darsia/fluidflower/wasserstein.py', '_resolved_path', 'src/darsia/fluidflower/mass_inputs.py', '_resolved_path', 'path')]


def _pure_function(path, name, *, original=False, canonical=None):
    """Compile the actual helper with stdlib globals and its declared forward binding."""
    repo = Path(__file__).resolve().parents[2]
    text = (subprocess.check_output(["git", "show", ORIGINAL_COMMIT + ":" + path], cwd=repo).decode("utf-8")
            if original else (repo / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(json=json, hashlib=hashlib, math=math, Path=Path, Any=Any, Mapping=Mapping)
    if canonical is not None:
        first = 1 if ast.get_docstring(node) else 0
        route = node.body[first]
        assert isinstance(route, ast.ImportFrom)
        expected_module = canonical[0].removeprefix("src/").removesuffix(".py").replace("/", ".")
        assert route.module == expected_module and len(route.names) == 1
        assert route.names[0].name == canonical[1]
        namespace[route.names[0].asname] = _pure_function(canonical[0], canonical[1])
        node.body.pop(first)  # Resolve this exact inspected import without importing SDK UI modules.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("path,name,canonical_path,canonical_name,kind", CASES)
def test_each_forward_keeps_exact_declared_signature(path, name, canonical_path, canonical_name, kind):
    repo = Path(__file__).resolve().parents[2]
    old = ast.parse(subprocess.check_output(["git", "show", ORIGINAL_COMMIT + ":" + path], cwd=repo).decode("utf-8"))
    new = ast.parse((repo / path).read_text(encoding="utf-8"))
    original = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
    actual = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(original.args) == ast.dump(actual.args)
    assert (ast.dump(original.returns) if original.returns else None) == (ast.dump(actual.returns) if actual.returns else None)
    assert ast.get_docstring(original) == ast.get_docstring(actual)


@pytest.mark.parametrize("path,name,canonical_path,canonical_name,kind", [r for r in CASES if "JSON" in r[4]])
def test_unicode_nan_and_mapping_digest_bytes_match_original(path, name, canonical_path, canonical_name, kind):
    actual = _pure_function(path, name, canonical=(canonical_path, canonical_name))
    original = _pure_function(path, name, original=True)
    values = [OrderedDict([("unicode", "\u03b2\u00e6"), ("nan", float("nan")), ("infinity", float("inf")), ("negative_zero", -0.0)]), {}, {"nested": [None, True, 1.25]}]
    if kind == "UTF8_mapping_JSON":
        values += [MappingProxyType({"unicode": "\u03b2", "field": 1}), [("field", 2)]]
    else:
        values += [["\u03b2", float("nan")], None]
    for value in values:
        assert actual(value) == original(value)
    ascii_mode = kind == "ASCII_JSON"
    value = values[0]
    payload = value if ascii_mode else dict(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_mode).encode("utf-8")
    assert actual(value) == hashlib.sha256(encoded).hexdigest()
    with pytest.raises(TypeError) as old_error:
        original({"unserializable": object()})
    with pytest.raises(TypeError) as new_error:
        actual({"unserializable": object()})
    assert str(new_error.value) == str(old_error.value)


@pytest.mark.parametrize("path,name,canonical_path,canonical_name,kind", [r for r in CASES if r[4] == "number"])
def test_finite_none_and_conversion_errors_match_original(path, name, canonical_path, canonical_name, kind):
    actual = _pure_function(path, name, canonical=(canonical_path, canonical_name))
    original = _pure_function(path, name, original=True)
    for value in (None, True, False, " 2.5 ", "bad", -0.0, float("nan"), float("inf"), object()):
        expected = original(value)
        result = actual(value)
        assert result == expected
        if expected == 0.0:
            assert math.copysign(1.0, result) == math.copysign(1.0, expected)
    class Overflow:
        def __float__(self):
            raise OverflowError("original conversion failure")
    with pytest.raises(OverflowError) as old_error:
        original(Overflow())
    with pytest.raises(OverflowError) as new_error:
        actual(Overflow())
    assert str(new_error.value) == str(old_error.value)


@pytest.mark.parametrize("path,name,canonical_path,canonical_name,kind", [r for r in CASES if r[4] == "path"])
def test_path_default_relative_absolute_and_errors_match_original(tmp_path, path, name, canonical_path, canonical_name, kind):
    actual = _pure_function(path, name, canonical=(canonical_path, canonical_name))
    original = _pure_function(path, name, original=True)
    for raw, relative in (("missing", None), ("branch/../missing", tmp_path), (tmp_path / "missing", tmp_path), ("~", None), (None, None), (None, tmp_path), ("relative", "not a Path")):
        assert actual(raw, relative_to=relative) == original(raw, relative_to=relative)
    class BadStringConversion:
        def __str__(self):
            raise TypeError("original string conversion failure")
    with pytest.raises(TypeError) as old_error:
        original(BadStringConversion())
    with pytest.raises(TypeError) as new_error:
        actual(BadStringConversion())
    assert str(new_error.value) == str(old_error.value)
