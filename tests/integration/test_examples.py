import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"


def _run(relative_path: str) -> int:
    """Run an example script from its own directory and return the exit code.

    Sphinx-Gallery executes examples with the working directory set to the
    script's folder, so the scripts use paths relative to that folder
    (``../images/...``); mirror that here.
    """
    script = EXAMPLES / relative_path
    return subprocess.run([sys.executable, script.name], cwd=script.parent).returncode


def test_color_correction():
    assert not _run("corrections/plot_color_correction.py")


def test_co2_and_tracer_analysis():
    assert not _run("analysis/co2_and_tracer_analysis.py")


def test_segmentation():
    assert not _run("segmentation/plot_segmentation.py")


def test_imread():
    assert not _run("io/plot_numpy_images.py")
    assert not _run("corrections/plot_optical_images.py")
    assert not _run("io/plot_reading_images.py")
    # assert not _run("io/dicom_images.py")


def test_imread_vtu_images():
    images_exist = len(sorted((EXAMPLES / "images").glob("fracture_flow*.vtu"))) == 2
    if not images_exist:
        pytest.xfail("Images required for test not available.")
    assert not _run("io/vtu_images.py")


def test_kernel_interpolation():
    images_exist = all(
        len(
            sorted(
                (EXAMPLES / "images").glob(f"kernel_interpolation_example_{app}.npz")
            )
        )
        == 1
        for app in ["base", "test"]
    )
    if not images_exist:
        pytest.xfail("Images required for test not available.")
    assert not _run("analysis/plot_kernel_interpolation.py")
