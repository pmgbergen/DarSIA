"""Canonical method functions import without numerical calibration DATA."""
import subprocess
import sys
from pathlib import Path


def test_photometry_function_import_without_numerical_configuration():
    """Function imports stay inert; actual method calls require explicit parameters."""
    source = Path(__file__).resolve().parents[2] / "src"
    code = f"""
import os, sys, inspect
sys.path.insert(0, {str(source)!r})
os.environ.pop('DARSIA_NATIVE_CONFIGURATION', None)
os.environ.pop('DARSIA_NATIVE_CONFIGURATION_SHA256', None)
from darsia.fluidflower.native.photometry import render_response
from darsia.fluidflower.native import photometry
from darsia.fluidflower.native.config import require_parameters
assert callable(render_response)
assert inspect.signature(render_response).parameters['parameters'].default is inspect.Parameter.empty
assert not hasattr(photometry, 'GRID')
try:
    require_parameters('photometry')
except ValueError as error:
    assert str(error) == 'An explicit native parameter document is required'
else:
    raise AssertionError('Numerical calls still require explicit DATA')
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True)
