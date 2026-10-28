"""Runtime documentation and immutable code contracts for the CUDA provider."""
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types

from darsia.fluidflower.mass import mass_cuda as provider


def _code_fields(code):
    def constant(value):
        if isinstance(value, types.CodeType):
            return _code_fields(value)
        if isinstance(value, tuple):
            return tuple(constant(item) for item in value)
        return value
    return code.co_code, tuple(constant(value) for value in code.co_consts)


def test_runtime_provider_help_keeps_all_numeric_code_and_original_bytes():
    source = Path(provider.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == "ebc695edf36bc308b7f994d589c0de74660c1f692c8d09b25d764ba351a9e5ea"
    reference = {"__name__": "qualified_provider_byte_reference"}
    exec(compile(source, "<qualified-provider-byte-reference>", "exec"), reference)
    seen = 0
    missing_original_docs = 0
    for name in ("CudaPixelKernels", "CudaHostPixelMath"):
        current = getattr(provider, name)
        original = reference[name]
        for method_name, member in vars(original).items():
            if method_name.startswith("_") and method_name != "__init__":
                continue
            method = getattr(original, method_name)
            if not inspect.isfunction(method):
                continue
            target = getattr(current, method_name)
            assert target.__doc__ and target.__doc__.strip(), (name, method_name)
            assert inspect.signature(target) == inspect.signature(method)
            assert _code_fields(inspect.unwrap(target).__code__) == _code_fields(inspect.unwrap(method).__code__)
            if method.__doc__ is None:
                missing_original_docs += 1
            else:
                assert target.__doc__ == method.__doc__
            seen += 1
    assert seen == 25 and missing_original_docs == 23


def test_fresh_cpu_package_import_documents_provider_without_loading_cupy():
    source_root = str(Path(provider.__file__).resolve().parents[3])
    code = """
import sys
from darsia.fluidflower.mass import mass_cuda as provider
assert 'cupy' not in sys.modules
for name in ('CudaPixelKernels', 'CudaHostPixelMath'):
    for method_name, value in vars(getattr(provider, name)).items():
        if method_name.startswith('_') and method_name != '__init__':
            continue
        function = getattr(getattr(provider, name), method_name)
        if callable(function):
            assert function.__doc__, (name, method_name)
assert 'darsia.fluidflower.execution' not in sys.modules
assert 'darsia.fluidflower.ui' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code],
                            env=dict(os.environ, PYTHONPATH=source_root),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
