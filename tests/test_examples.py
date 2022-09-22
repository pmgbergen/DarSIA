from pathlib import Path
import os

def test_color_correction():
    assert not(os.system(f'python {str(Path(f"{os.path.dirname(__file__)}/../examples/color_correction.py"))}'))

def test_co2_and_tracer_analysis():
    assert not(os.system(f'python {str(Path(f"{os.path.dirname(__file__)}/../examples/co2_and_tracer_analysis.py"))}'))

def test_segmentation_watershed():
    assert not(os.system(f'python {str(Path(f"{os.path.dirname(__file__)}/../examples/segmentation_watershed.py"))}'))

