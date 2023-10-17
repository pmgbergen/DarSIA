import os
from pathlib import Path

import pytest


def test_color_correction():
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/color_correction.py"))}'
        )
    )


def test_co2_and_tracer_analysis():
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/co2_and_tracer_analysis.py"))}'
        )
    )


def test_segmentation():
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/segmentation.py"))}'
        )
    )


def test_imread():
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/numpy_images.py"))}'
        )
    )
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/optical_images.py"))}'
        )
    )
    #    assert not (
    #        os.system(
    #            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/dicom_images.py"))}'
    #        )
    #    )
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/reading_images.py"))}'
        )
    )


def test_imread_vtu_images():

    # Test whether images required for test are available
    folder = Path(f"{os.path.dirname(__file__)}/../../examples/images")
    images_exist = len(list(sorted(folder.glob("fracture_flow*.vtu")))) == 2

    if not images_exist:
        pytest.xfail("Images required for test not available.")

    # If so, run test.
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/vtu_images.py"))}'
        )
    )


def test_kernel_interpolation():

    # Test whether images required for test are available
    folder = Path(f"{os.path.dirname(__file__)}/../../examples/images")
    images_exist = all(
        [
            len(list(sorted(folder.glob(f"kernel_interpolation_example_{app}.npz"))))
            == 1
            for app in ["base", "test"]
        ]
    )

    if not images_exist:
        pytest.xfail("Images required for test not available.")

    # If so, run test.
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/kernel_interpolation.py"))}'
        )
    )
