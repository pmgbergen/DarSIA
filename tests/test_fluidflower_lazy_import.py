"""A command-line import must not start the optional execution framework."""

import os
from pathlib import Path
import subprocess
import sys


def test_package_import_is_lightweight_and_public_api_is_available():
    """Keep the old public API while avoiding eager controller/GUI imports."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    code = """
import sys
import darsia.fluidflower as f
assert 'darsia.fluidflower.execution' not in sys.modules
assert 'darsia.fluidflower.calibration_review' not in sys.modules
assert 'darsia.fluidflower.mass_materialization' not in sys.modules
assert 'default_analysis_domain_config' in dir(f)
from darsia.fluidflower.analysis_domains import default_analysis_domain_config
assert f.default_analysis_domain_config is default_analysis_domain_config
assert 'darsia.fluidflower.execution' not in sys.modules
try:
    f.nonexistent_public_api
except AttributeError:
    pass
else:
    raise AssertionError('Unknown public names must raise AttributeError')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=root, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
