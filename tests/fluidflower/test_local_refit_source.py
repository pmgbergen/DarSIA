"""Source identity guards only; these tests never invoke a numerical fitter.

Synthetic module compilation/execution creates a small test fixture when the
tests run. The identity implementation itself only compiles source for
comparison, and must never execute that compiled source.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType

import pytest

from darsia.fluidflower.mass import n1_refit
from darsia.fluidflower.pipeline import local_refit_source as source
from darsia.fluidflower.pipeline import local_refit, refit_runner
from test_local_refit import local_metadata  # noqa: F401 -- isolated metadata fixture
from test_refit_jobs import series  # noqa: F401 -- dependency of local_metadata


MODULE_TEXT = '''
def numerical_value(value):
    return value * 3

class NumericalRules:
    def value(self, number):
        return number + 2

    @classmethod
    def from_value(cls, number):
        return number + 4
'''


def test_loaded_refit_identity_binds_the_complete_actual_numerical_modules():
    root = Path(source.__file__).resolve().parents[4]
    identity = source.source_identity(root, calibration_sha256="a" * 64)
    binding = identity["source_binding"]
    assert [row["module"] for row in binding["modules"]] == list(source.MODULES)
    assert all(row["functions"] and row["size_bytes"] > 0 for row in binding["modules"])
    assert identity["commit"] == "local-source-sha256:" + binding["sha256"]


@pytest.fixture
def copied_module(tmp_path):
    root = tmp_path / "copied-source"
    path = root / "src" / "fixture_numerics.py"
    path.parent.mkdir(parents=True)
    path.write_text(MODULE_TEXT, encoding="utf-8")
    module = ModuleType("fixture_numerics")
    module.__file__ = str(path)
    exec(compile(MODULE_TEXT, str(path), "exec"), module.__dict__)
    assert not (root / ".git").exists()
    return root, path, module


def test_module_binding_works_without_git_and_ignores_unrelated_documents(copied_module):
    root, path, module = copied_module
    before = source._module_binding(module, root)
    docs = root / "documents"
    docs.mkdir()
    (docs / "untracked-notes.md").write_text("Unrelated user notes", encoding="utf-8")
    (root / "another-unrelated.txt").write_text("No source change", encoding="utf-8")
    assert source._module_binding(module, root) == before
    assert path.read_text(encoding="utf-8") == MODULE_TEXT
    assert not (root / ".git").exists()


def test_full_identity_uses_copied_loaded_module_without_git_or_unrelated_document_checks(
        copied_module, monkeypatch):
    root, path, module = copied_module
    original_import = source.importlib.import_module
    monkeypatch.setattr(source, "MODULES", (module.__name__,))
    monkeypatch.setattr(source.importlib, "import_module",
        lambda name: module if name == module.__name__ else original_import(name))
    first = source.source_identity(root, calibration_sha256="a" * 64)
    (root / "user-notes.md").write_text("Unrelated preserved document", encoding="utf-8")
    second = source.source_identity(root, calibration_sha256="a" * 64)
    assert first["commit"] == second["commit"] == "local-source-sha256:" + first["source_binding"]["sha256"]
    assert first["source_binding"] == second["source_binding"]
    assert first["source_binding"]["format"] == source.FORMAT
    assert first["source_binding"]["modules"] == [source._module_binding(module, root)]
    assert set(first["source_binding"]["runtime"]) == {"python", "implementation", "numpy", "scipy", "threadpoolctl"}
    assert first["calibration_sha256"] == "a" * 64 and first["precision"] == "float32"
    assert "environment_lock" not in first and not (root / ".git").exists()
    assert path.read_text(encoding="utf-8") == MODULE_TEXT


def test_changed_numerical_source_refuses_the_still_loaded_original_function(copied_module):
    root, path, module = copied_module
    source._module_binding(module, root)
    path.write_text(MODULE_TEXT.replace("return value * 3", "return value * 5"), encoding="utf-8")
    with pytest.raises(ValueError):
        source._module_binding(module, root)


@pytest.mark.parametrize("member", ["function", "method", "classmethod"])
def test_changed_loaded_code_refuses_unchanged_source(copied_module, monkeypatch, member):
    root, path, module = copied_module
    source._module_binding(module, root)
    if member == "function":
        function = module.numerical_value
    elif member == "method":
        function = module.NumericalRules.value
    else:
        function = module.NumericalRules.__dict__["from_value"].__func__
    original = function.__code__
    # Change actual loaded bytecode without changing its filename/name/module
    # identity; no numerical function is called by this test.
    replacement = compile("def replacement(*args):\n    return -123\n", str(path), "exec")
    code = next(value for value in replacement.co_consts if hasattr(value, "co_code"))
    monkeypatch.setattr(function, "__code__", code)
    assert function.__code__ is not original
    with pytest.raises(ValueError):
        source._module_binding(module, root)
    assert path.read_text(encoding="utf-8") == MODULE_TEXT


def test_actual_loaded_fitter_code_change_is_detected_without_calling_it(monkeypatch):
    root = Path(n1_refit.__file__).resolve().parents[4]
    source._module_binding(n1_refit, root)
    original = n1_refit.fit_candidate.__code__
    replacement = compile("def replacement(*args, **kwargs):\n    return None\n",
                          "test-only-replacement", "exec")
    code = next(value for value in replacement.co_consts if hasattr(value, "co_code"))
    monkeypatch.setattr(n1_refit.fit_candidate, "__code__", code)
    assert n1_refit.fit_candidate.__code__ is not original
    with pytest.raises(ValueError):
        source._module_binding(n1_refit, root)


@pytest.mark.parametrize("fault", ["digest", "binding"])
def test_source_identity_change_refuses_start_before_launch_or_fit(local_metadata, monkeypatch, fault):
    """Compare persisted/current identity metadata; never run a fitter."""
    fixture = local_metadata
    local_refit.create_local_refit_job(**fixture.arguments)
    before = (fixture.root / "job.json").read_bytes()
    changed = deepcopy(fixture.identity)
    if fault == "digest":
        changed["source_binding"]["sha256"] = "e" * 64
        changed["commit"] = "local-source-sha256:" + "e" * 64
    else:
        changed["source_binding"]["runtime"] = {"synthetic": "changed metadata with an unchanged tag"}
    monkeypatch.setattr(local_refit, "source_identity", lambda *a, **k: deepcopy(changed))
    monkeypatch.setattr(refit_runner, "run_local_refit", lambda *a, **k: pytest.fail("Changed source started fitting"))
    with pytest.raises(ValueError, match="source identity changed"):
        local_refit.run_local_refit_job(fixture.root)
    assert (fixture.root / "job.json").read_bytes() == before
    assert json.loads(before)["source_binding"] == fixture.identity["source_binding"]
    assert not (fixture.root / "launch.json").exists()
    assert not (fixture.root / "results").exists()
