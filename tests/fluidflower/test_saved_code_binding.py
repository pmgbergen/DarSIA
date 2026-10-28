"""Passive saved-source declarations keep exact names/hashes without execution."""
import hashlib
import sys
from pathlib import Path

import pytest

from darsia.fluidflower.phase_saved_review_metadata import declared_code_binding


def _descriptor(path, **extra):
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), **extra)


def _parameters(tmp_path, declaration):
    runner = tmp_path / "runner.py"
    runner.write_text(declaration, encoding="utf-8")
    numerical = tmp_path / "numeric.py"
    numerical.write_text("raise AssertionError('source must not execute')\n", encoding="utf-8")
    names = ["saved_binding_fixture.runner", "saved_binding_fixture.numeric"]
    return dict(code_declaration=_descriptor(runner, function_name="code_binding",
                                             module_name=names[0]),
                declared_code_names=names,
                code_sources={names[0]: _descriptor(runner),
                              names[1]: _descriptor(numerical)})


def test_literal_bindings_are_read_without_import_or_execution(tmp_path):
    parameters = _parameters(tmp_path,
        "raise AssertionError('runner must not execute')\n"
        "def code_binding():\n"
        "    names = (__name__, 'saved_binding_fixture.numeric')\n")
    expected = {name: entry["sha256"] for name, entry in parameters["code_sources"].items()}
    assert declared_code_binding(parameters=parameters) == expected
    assert "saved_binding_fixture.runner" not in sys.modules
    assert "saved_binding_fixture.numeric" not in sys.modules


def test_nonliteral_names_are_rejected_without_execution(tmp_path):
    parameters = _parameters(tmp_path,
        "raise AssertionError('runner must not execute')\n"
        "def code_binding():\n"
        "    names = tuple(discover_numerical_modules())\n")
    with pytest.raises(ValueError, match="literal name tuple"):
        declared_code_binding(parameters=parameters)


def test_declared_names_must_match_the_full_order_and_set(tmp_path):
    parameters = _parameters(tmp_path,
        "def code_binding():\n"
        "    names = (__name__, 'saved_binding_fixture.numeric')\n")
    parameters["declared_code_names"].reverse()
    with pytest.raises(ValueError, match="declared source names changed"):
        declared_code_binding(parameters=parameters)


def test_passive_source_bytes_are_still_bound(tmp_path):
    parameters = _parameters(tmp_path,
        "def code_binding():\n"
        "    names = (__name__, 'saved_binding_fixture.numeric')\n")
    Path(parameters["code_sources"]["saved_binding_fixture.numeric"]["path"]).write_text(
        "changed bytes\n", encoding="utf-8")
    with pytest.raises(ValueError):
        declared_code_binding(parameters=parameters)
