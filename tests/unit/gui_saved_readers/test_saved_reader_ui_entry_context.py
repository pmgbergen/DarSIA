"""Owned UI contexts and captured read tasks; no Qt/science/job execution."""
import ast
from contextvars import ContextVar, copy_context
from copy import deepcopy
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from darsia.fluidflower import series_native_context as context
from darsia.fluidflower.native import config
from test_saved_reader_code_relocation import package, binding


@pytest.fixture
def isolated_native_context(monkeypatch):
    """A fresh original ContextVar; no environment is required at UI startup."""
    monkeypatch.setattr(config, "_configured", ContextVar("unit-ui-native", default=None))
    for key in context._ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_owned_UI_context_binds_exact_runtime_copy_and_restores_original(package, isolated_native_context):
    root, document = package
    before = deepcopy(document)
    original = config.configure_parameters(document)
    try:
        with context.saved_reader_ui_context() as evidence:
            current = config.require_configuration()
            assert current is not document and len(evidence) == 5
            assert current["numeric"] == before["numeric"]
            assert all(Path(row["current_code_source"]["path"]).is_relative_to(root)
                       for row in current["saved_review_metadata"].values())
        assert config.require_configuration() is document and document == before
    finally:
        config.reset_parameters(original)


def test_UI_context_restores_its_original_token_on_error(package, isolated_native_context):
    _root, document = package
    token = config.configure_parameters(document)
    try:
        with pytest.raises(RuntimeError, match="unit-only UI error"):
            with context.saved_reader_ui_context():
                raise RuntimeError("unit-only UI error")
        assert config.require_configuration() is document
    finally:
        config.reset_parameters(token)


def test_absent_native_stays_absent_without_startup_environment(isolated_native_context):
    with context.saved_reader_ui_context() as evidence:
        assert evidence == ()
        with pytest.raises(ValueError, match="explicit native parameter document"):
            config.require_configuration()
    with pytest.raises(ValueError, match="explicit native parameter document"):
        config.require_configuration()


def test_malformed_explicit_child_pair_is_not_treated_as_absence(isolated_native_context, monkeypatch):
    monkeypatch.setenv(context._ENVIRONMENT_KEYS[0], "unit-only-missing-native.json")
    with pytest.raises(ValueError, match="explicit native parameter document"):
        with context.saved_reader_ui_context():
            raise AssertionError("Malformed explicit pair reached UI body")


def test_UI_entry_decorator_owns_context_before_original_body(package, isolated_native_context, monkeypatch):
    _root, document = package
    token = config.configure_parameters(document)
    @context.saved_reader_ui_entrypoint
    def UI_body(value):
        current = config.require_configuration()
        assert current is not document
        assert "nonexistent-old-clone" not in current["saved_review_metadata"]["physics_profiles"]["current_code_source"]["path"]
        return value
    try:
        assert UI_body(17) == 17
        assert config.require_configuration() is document
        # Reach each actual decorated UI entry function, preserving its full
        # source AST. Isolate unrelated display wiring, then stop at the parser;
        # no QApplication, review model or numerical operation is invoked.
        from darsia.fluidflower import phase_model_app, raw_production_review
        installed = []
        def display_reader_registration():
            active = config.require_configuration()
            assert active is not document and active["numeric"] == document["numeric"]
            assert "nonexistent-old-clone" not in active["saved_review_metadata"]["physics_profiles"]["current_code_source"]["path"]
            installed.append(True)
        monkeypatch.setattr(raw_production_review, "install_readers", display_reader_registration)
        for relative in ("tools/calibration_review_desktop.py", "tools/phase_model_comparison_view.py"):
            installed_before = len(installed)
            source = Path(phase_model_app.__file__).parent / relative
            module = ast.parse(source.read_bytes())
            entry = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            assert [ast.unparse(n) for n in entry.decorator_list] == ["saved_reader_ui_entrypoint"]
            seen = []
            class StopAtOriginalParser(Exception):
                pass
            def parser_boundary(*args, **kwargs):
                active = config.require_configuration()
                assert active is not document and active["numeric"] == document["numeric"]
                assert "nonexistent-old-clone" not in active["saved_review_metadata"]["physics_profiles"]["current_code_source"]["path"]
                seen.append(relative)
                raise StopAtOriginalParser()
            namespace = dict(__doc__="unit UI entry boundary only",
                saved_reader_ui_entrypoint=context.saved_reader_ui_entrypoint,
                build_parser=lambda: SimpleNamespace(parse_args=parser_boundary),
                argparse=SimpleNamespace(ArgumentParser=parser_boundary))
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            unit = ast.fix_missing_locations(ast.Module(body=[future, entry], type_ignores=[]))
            exec(compile(unit, str(source) + ":exact-entry-AST", "exec"), namespace)
            with pytest.raises(StopAtOriginalParser):
                namespace["main"]()
            assert seen == [relative] and config.require_configuration() is document
            expected_installs = int(relative == "tools/calibration_review_desktop.py")
            assert len(installed) == installed_before + expected_installs
    finally:
        config.reset_parameters(token)


def test_explicit_phase_studio_context_keeps_original_document_SHA_and_no_required_startup_env(package, isolated_native_context, tmp_path):
    from darsia.fluidflower.phase_model_app import _configuration_context
    _root, document = package
    raw = (json.dumps(document) + "\n").encode()
    path = tmp_path / "native-original.json"
    path.write_bytes(raw)
    expected = binding(path, raw)
    arguments = SimpleNamespace(configuration=path, configuration_sha256=expected["sha256"])
    with _configuration_context(arguments) as descriptor:
        current = config.require_configuration()
        assert descriptor["sha256"] == expected["sha256"]
        assert len(descriptor["current_code_bindings"]) == 5
        assert current["numeric"] == document["numeric"]
        assert "nonexistent-old-clone" not in current["saved_review_metadata"]["physics_profiles"]["current_code_source"]["path"]
    assert path.read_bytes() == raw
    assert not any(key in __import__("os").environ for key in context._ENVIRONMENT_KEYS)
    with pytest.raises(ValueError, match="explicit native parameter document"):
        config.require_configuration()


def test_child_UI_context_relocates_only_runtime_copy_of_existing_bound_environment(package, isolated_native_context, tmp_path, monkeypatch):
    _root, document = package
    raw = (json.dumps(document) + "\n").encode()
    path = tmp_path / "inherited-native.json"
    path.write_bytes(raw)
    original = binding(path, raw)
    monkeypatch.setenv(context._ENVIRONMENT_KEYS[0], str(path))
    monkeypatch.setenv(context._ENVIRONMENT_KEYS[1], original["sha256"])
    with context.saved_reader_ui_context():
        assert "nonexistent-old-clone" not in config.require_configuration()["saved_review_metadata"]["physics_profiles"]["current_code_source"]["path"]
    assert path.read_bytes() == raw
    assert config.require_configuration() == document
    assert __import__("os").environ[context._ENVIRONMENT_KEYS[1]] == original["sha256"]


def test_actual_UI_read_task_base_captures_context_before_thread_start(package, isolated_native_context):
    """Execute the exact tiny base with a non-Qt constructor, not a fake app."""
    # Context.__file__ is a tiny package fixture. The tested source lives in the
    # isolated source tree imported by this test, independent of that fixture.
    from darsia.fluidflower import phase_model_app
    source = Path(phase_model_app.__file__).parent / "tools/calibration_review_desktop.py"
    tree = ast.parse(source.read_bytes())
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_SavedReaderContextTask")
    namespace = {"QRunnable": type("UnitNonQtRunnable", (), {}), "copy_context": copy_context}
    exec(compile(ast.Module(body=[base], type_ignores=[]), str(source) + ":unit-base-only", "exec"), namespace)
    root, document = package
    token = config.configure_parameters(document)
    seen = []
    class ReadTask(namespace["_SavedReaderContextTask"]):
        def _run_saved_read(self):
            seen.append(deepcopy(config.require_configuration()))
    try:
        with context.saved_reader_ui_context():
            task = ReadTask()
        assert config.require_configuration() is document
        thread = Thread(target=task.run, name="unit-owned-UI-context-only")
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(seen) == 1
        assert seen[0]["numeric"] == document["numeric"]
        assert all(Path(row["current_code_source"]["path"]).is_relative_to(root)
                   for row in seen[0]["saved_review_metadata"].values())
        assert config.require_configuration() is document
    finally:
        config.reset_parameters(token)
