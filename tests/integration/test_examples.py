import os
from pathlib import Path


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
    #    assert not (
    #        os.system(
    #            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/vtu_images.py"))}'
    #        )
    #    )
    assert not (
        os.system(
            f'python {str(Path(f"{os.path.dirname(__file__)}/../../examples/reading_images.py"))}'
        )
    )
